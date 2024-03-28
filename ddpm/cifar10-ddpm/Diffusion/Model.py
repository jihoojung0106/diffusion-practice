import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

'''[batch_size]->[batch_size,dim=512]'''
class TimeEmbedding(nn.Module): 
    def __init__(self, T, d_model, dim):#T=1000,d_model=128(channel),dim=128*4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        #pos[:None](T,1) * emb[None,:] (1,d_model/2)을 곱하면 -> emb=(T,d_model/2)
        emb = pos[:, None] * emb[None, :] #pos=[1,2,3]->pos[:, None]=[[1],[2],[3]],
        #emb = [0.1, 0.2, 0.3]->emb[None, :]=[[0.1, 0.2, 0.3]]
        
        assert list(emb.shape) == [T, d_model // 2]
        #[T, d_model // 2]=>[T, d_model // 2, 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        #[T, d_model // 2, 2]=>[T, d_model]
        emb = emb.view(T, d_model)

        
        self.timembedding = nn.ModuleList([
            # emb=[T, d_model]->t가 들어오면, emb[t]에 해당하는 값을 내뱉는 것.
            #즉 t=[batch_size]->[batch_size,d_model]
            nn.Embedding.from_pretrained(emb),
            #[batch_size,d_model]->[batch_size,dim=512]
            nn.Linear(d_model, dim),
            #[batch_size,d_model]->[batch_size,dim=512]
            Swish(),
            #[batch_size,d_model]->[batch_size,dim=512]
            nn.Linear(dim, dim)]
        )
        self.initialize()
    #가중치 초기화
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        # emb = self.timembedding(t)
        for layer in self.timembedding:
            t=layer(t)
        emb=t
        return emb

'''[B, C, H, W]->[B, C, H/2, W/2]'''
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        #절반으로 줄어듦?
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


'''[B, C, H, W]->[B, C, H, W]'''
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        #x=[B, C, H, W]
        B, C, H, W = x.shape
        h = self.group_norm(x)
        #[B, C, H, W] -> [B, C, H, W] 
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        #[B, C, H, W] -> [B, H, W, C] -> [B, H * W, C]
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        #[B, C, H, W] -> [B, C, H * W]
        k = k.view(B, C, H * W)
        #w=[B, H * W, H * W]
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        #[B, C, H, W] -> [B, H, W, C] -> [B, H * W, C]
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        #[B, H * W, H * W] * [B, H * W, C]->[B, H * W, C]
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        #[B, H * W, C]->[B, H, W, C]->[B, C, H, W]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        #[B, C, H, W]->[B, C, H, W]
        h = self.proj(h)
        #x+h=[B, C, H, W]
        return x + h

'''[batch_size,in_ch,height,width]->[batch_size,out_ch,height,width]'''
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        #[batch_size,in_ch,height,width]->[batch_size,out_ch,height,width]
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        #[batch_size,tdim=512]->[batch_size,out_ch]
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        #[batch_size,out_ch,height,width]->[batch_size,out_ch,height,width]
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        #[batch_size,in_ch,height,width]->[batch_size,out_ch,height,width]
        h = self.block1(x)
        #[batch_size,tdim=512]->[batch_size,out_ch]->[batch_size,out_ch,1,1]
        #[batch_size,out_ch,height,width] + [batch_size,out_ch,1,1](브로드캐스팅)->[batch_size,out_ch,height,width]
        h += self.temb_proj(temb)[:, :, None, None]
        #[batch_size,out_ch,height,width]->[batch_size,out_ch,height,width]
        h = self.block2(h)
        #[batch_size,out_ch,height,width]->[batch_size,out_ch,height,width]
        h = h + self.shortcut(x)
        #[batch_size,out_ch,height,width]->[batch_size,out_ch,height,width]
        h = self.attn(h)
        return h


class UNet(nn.Module):
    #T=1000,ch=128, ch_mult=[1, 2, 2, 2],attn=[2],num_res_blocks=2,dropout=0.15
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        #동일한 사이즈
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult #1) out_ch=128*1,now_ch=128,
            for _ in range(num_res_blocks):
                '''[B, in_ch, H, W]->[B, out_ch, H, W]'''
                self.downblocks.append(
                    ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim,dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch) #chs=[128*4,256*8]

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList() 
        for i, mult in reversed(list(enumerate(ch_mult))): #[2, 2, 2, 1]
            out_ch = ch * mult 
            for _ in range(num_res_blocks + 1): #3번
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        #temb=[batch_size,d_model]->[batch_size,dim=512]
        temb = self.time_embedding(t)
        # Downsampling
        # x = [batch_size,3,32,32] -> h = [batch_size,128,32,32]
        h = self.head(x)
        hs = [h] #hs=[(batch_size,128,32,32)], temb=[8,512]
        for layer in self.downblocks:
            h = layer(h, temb) #h=(batch_size,256,8,8)
            hs.append(h) #hs=[(batch_size,128,32,32)*3,(batch_size,128,16,16),(batch_size,256,16,16)*2,(batch_size,256,8,8)*3,(batch_size,256,4,4)*3]
            
            
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb) #(batch_size,256,8,8)
            
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h) #(batch_size,3,32,32)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, )) #torch.Size([8])
    y = model(x, t)
    print(y.shape)


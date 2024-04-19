import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder
import numpy as np

import pandas as pd

from utils.text_encoder import TokenTextEncoder
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import VOCODERS
class BinarizationError(Exception):
    pass



class BaseBinarizer:
    def __init__(self, processed_data_dir="data/processed"):
        self.processed_data_dir=processed_data_dir
        self.binary_data_dir ="data/binary"
        self.binarization_args = {
            "shuffle": False,
            "with_txt": True,
            "with_wav": False,
            "with_align": True,
            "with_spk_embed": True,
            "with_f0": True,
            "with_f0cwt": True
        }
        self.pre_align_args = {
            "use_tone": True, # for ZH
            "forced_align": "mfa",
            "use_sox": False,
            "txt_processor": "en",
            "allow_no_txt": False,
            "denoise": False
        }
        self.raw_data_dir="data/raw/LJSpeech-1.1"
        self.forced_align = "mfa"
        tg_dir = "data/processed/ljspeech/mfa_outputs"
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2tgfn = {}
        self.item2spk = {}
        self.meta_df = pd.read_csv(f"data/processed/ljspeech/metadata_phone.csv", dtype=str)
        for r_idx, r in self.meta_df.iterrows(): #item_name,spk,txt,txt_raw,ph,wav_fn
            item_name = r['item_name']
            self.item2txt[item_name] = r['txt']
            self.item2ph[item_name] = r['ph']
            self.item2wavfn[item_name] = os.path.join(self.raw_data_dir, 'wavs', os.path.basename(r['wav_fn']).split('_')[1])
            self.item2spk[item_name] = r.get('spk', 'SPK1') #만약 'spk' 키가 있으면 그 값 반환, 없다면 'SPK1' 반환
            self.item2tgfn[item_name] = f"{tg_dir}/{item_name}.TextGrid"
        self.item_names = sorted(list(self.item2txt.keys())) #len=13100
       
        self.test_num=int(len(self.item_names)* 0.05)
        self.valid_num=int(len(self.item_names) * 0.03)
        self.test_item_names=self.item_names[0: self.test_num]
        self.valid_item_names=self.item_names[self.test_num: self.test_num+self.valid_num] 
        self.train_item_names=self.item_names[self.test_num+self.valid_num:]

    

    

    def process(self):
        os.makedirs(self.binary_data_dir, exist_ok=True)
        phone_list = json.load(open("data/binary/phone_set.json"))
        self.phone_encoder= TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}') #file만들기
        lengths = []
        f0s = []
        total_sec = 0
        voice_encoder = VoiceEncoder().cuda()
        
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
            
            
        for item_name in tqdm(item_names, desc="Processing items"):
            ph = self.item2ph[item_name]
            txt = self.item2txt[item_name]
            tg_fn = self.item2tgfn.get(item_name) #.textgrid 디렉토리
            wav_fn = self.item2wavfn[item_name] #.wav 디렉토리
            spk_id = 0
            
            item = self.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, self.phone_encoder)
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])
            if 'wav' in item:
                print("del wav")
                del item['wav']
            builder.add_item(item)
            lengths.append(item['len'])
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder):
        audio_sample_rate=22050
        wav, mel = VOCODERS["pwg"].wav2spec(wav_fn) #fn=filename의 약자
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / audio_sample_rate, 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            cls.get_pitch(wav, mel, res)
            cls.get_f0cwt(res['f0'], res)
            phone_encoded = res['phone'] = encoder.encode(ph) #(109,),(49,) 등 다양한 사이즈
            cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

    @staticmethod
    def get_align(tg_fn, ph, mel, phone_encoded, res):
        mel2ph, dur = get_mel2ph(tg_fn, ph, mel) #ph.split(" ")를 한 것과 res['dur'] shape 같음
        
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")
        res['mel2ph'] = mel2ph #(256,),(348,) 등 사이즈 달라짐
        res['dur'] = dur #(43,)

    @staticmethod
    def get_pitch(wav, mel, res):
        f0, pitch_coarse = get_pitch(wav, mel)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        res['f0'] = f0
        res['pitch'] = pitch_coarse

    @staticmethod
    def get_f0cwt(f0, res): #cwt : 웨이블릿 변환
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        #f0를 적절히 보간하고 정규화 : cont_lf0_lpf_norm
        uv, cont_lf0_lpf = get_cont_lf0(f0) #uv는 0,1 이진배열, cont_lf0_lpf : 적절힌 선형보간하고 log 씌운 f0
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        #정규화한 것을 wavelet 변환 cont_lf0_lpf_norm : (832,)
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        if np.any(np.isnan(Wavelet_lf0)):
            raise BinarizationError("NaN CWT")
        res['cwt_spec'] = Wavelet_lf0
        res['cwt_scales'] = scales
        res['f0_mean'] = logf0s_mean_org
        res['f0_std'] = logf0s_std_org


if __name__ == "__main__":
    BaseBinarizer().process()

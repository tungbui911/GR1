from metric import Align, Correct_Rate
from g2p_en import G2p
from fastapi import FastAPI, Request
import os, torch, librosa, uvicorn, nest_asyncio, gc
import numpy as np
from python_speech_features import fbank
import scipy.io.wavfile as wav
from pyctcdecode import build_ctcdecoder
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


wav2vec2_large_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
wav2vec2_large_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")

phonemes_70 = [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
    'EY2', 'F', 'G', 'HH',
    'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'OW0', 'OW1',
    'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
    'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

ipa_mapping = {
    'y': 'j', 'ng': 'ŋ', 'dh': 'ð', 'w': 'w', 'er': 'ɝ', 'r': 'ɹ', 'm': 'm', 'p': 'p',
    'k': 'k', 'ah': 'ʌ', 'sh': 'ʃ', 't': 't', 'aw': 'aʊ', 'hh': 'h', 'ey': 'eɪ', 'oy': 'ɔɪ',
    'zh': 'ʒ', 'n': 'n', 'th': 'θ', 'z': 'z', 'aa': 'ɑ', 'ao': 'aʊ', 'f': 'f', 'b': 'b', 'ih': 'ɪ',
    'jh': 'dʒ', 's': 's', 'err': '', 'iy': 'i', 'uh': 'ʊ', 'ch': 'tʃ', 'g': 'g', 'ay': 'aɪ', 'l': 'l',
    'ae': 'æ', 'd': 'd', 'v': 'v', 'uw': 'u', 'eh': 'ɛ', 'ow': 'oʊ'
}

map_39 = {}
for phoneme in phonemes_70:
    phoneme_39 = phoneme.lower()
    if phoneme_39[-1].isnumeric():
        phoneme_39 = phoneme_39[:-1]
    map_39[phoneme] = phoneme_39

def text_to_phonemes(text):
    g2p = G2p()
    phonemes = g2p(text)
    word_phoneme_in = []
    phonemes_result = []
    n_word = 0
    for phoneme in phonemes:
        if map_39.get(phoneme, None) is not None:
            phonemes_result.append(map_39[phoneme])
            word_phoneme_in.append(n_word)
        elif len(phoneme.strip()) == 0:
            n_word += 1
    return ' '.join(phonemes_result), word_phoneme_in

dict_vocab = {
    "y": 0, "ng": 1, "dh": 2, "w": 3, "er": 4, "r": 5, "m": 6, "p": 7, "k": 8, "ah": 9, "sh": 10, 
    "t": 11, "aw": 12, "hh": 13, "ey": 14, "oy": 15, "zh": 16, "n": 17, "th": 18, "z": 19, "aa": 20, 
    "ao": 21, "f": 22, "b": 23, "ih": 24, "jh": 25, "s": 26, "err": 27, "iy": 28, "uh": 29, "ch": 30, 
    "g": 31, "ay": 32, "l": 33, "ae": 34, "d": 35, "v": 36, "uw": 37, "eh": 38, "ow": 39
}

device = torch.device('cpu')

model = torch.load('./checkpoints/checkpoint.pth', map_location=torch.device('cpu'))
model.eval()

labels = sorted([w for w in list(dict_vocab.keys())], key=lambda x : dict_vocab[x])
labels = [f'{w} ' for w in labels]

def text_to_tensor(text):
    text = text.lower()
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list

def get_filterbank(path):
    (rate,sig) = wav.read(path)
    filter, energy = fbank(sig,rate, winlen=0.032, winstep = 0.02, nfilt=80)
    filter = filter.reshape(80, -1)
    energy = energy.reshape(1,-1)
    data = np.concatenate((filter,energy))
    return data

wav2vec2_large_submodel = torch.nn.Sequential(*(list(wav2vec2_large_model.children())[:-2])).to(device)
wav2vec2_large_submodel.eval()

def en_phonetic_extract(path):
    with torch.no_grad():
        path = path
        wav, sr = librosa.load(path, sr=16000)
        input_values = wav2vec2_large_processor(wav, return_tensors="pt",sampling_rate=16000, padding="longest").input_values
        input_values = input_values.to(device)     
        outputs = wav2vec2_large_submodel(input_values)
    return outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()

def run_model(text, audio_path):
    with torch.no_grad():
        phonemes, word_phoneme_in = text_to_phonemes(text)

        linguistic = torch.tensor(text_to_tensor(phonemes), device=device).unsqueeze(0)
        phonetic = torch.tensor(en_phonetic_extract(audio_path), device=device, dtype=torch.float)
        fbank = get_filterbank(audio_path)
        rshape_fbank = fbank[:,:phonetic.shape[0]]
        acoustic = torch.tensor(rshape_fbank.T, device=device, dtype=torch.float).unsqueeze(0)
        phonetic = phonetic.unsqueeze(0)
        
        outputs = model(acoustic, phonetic, linguistic)
        x = F.log_softmax(outputs,dim=2).squeeze(0)
        x = x.detach().cpu().numpy()
        torch.cuda.empty_cache()
        gc.collect()
        decoder = build_ctcdecoder(
            labels = labels
        )
        hypothesis = str(decoder.decode(x)).strip()
        print(hypothesis)
        return phonemes, hypothesis, word_phoneme_in

#canonical_text = 'Recommendation we have here suggested would greatly advance the security of the office without any experiment about the fundamental liberty'
canonical_text = 'Some reasonable precautions have been taken to ensure that the solutions are accurate'
#audio_path = 'Rec.wav'
audio_path = 'Acc.wav'
canonical_phoneme, predict_phoneme, word_phoneme_in = (run_model(canonical_text, audio_path))
print("\nÂm vị gốc:")
print(canonical_phoneme)
print("\nÂm vị dự đoán:")
print(predict_phoneme)

#print(Align(canonical_phoneme, predict_phoneme))
print("\nĐộ chính xác:")
print(Correct_Rate(canonical_phoneme, predict_phoneme))
print("\n")


Align_Canonical, Align_predict = Align(canonical_phoneme, predict_phoneme)

for i in range(len(Align_Canonical)):
    if Align_Canonical[i]=="<eps>":
        print("Đọc thừa âm vị " + Align_predict[i])
    elif Align_predict[i] == "<eps>" and Align_Canonical[i]!=" ":
        print("Đọc thiếu âm vị " + Align_Canonical[i])
    elif Align_Canonical[i]!=Align_predict[i] and Align_Canonical[i]!=" " and Align_predict[i]!=" " and (Align_predict[i]!="<eps>" or Align_Canonical[i]!="<eps>"):
        print("Đọc sai " + Align_Canonical[i] + " thành " + Align_predict[i])
       
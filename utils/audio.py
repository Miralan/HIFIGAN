import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import scipy
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

sample_rate = 22050 
n_fft = 2048
fft_bins = n_fft // 2 + 1
num_mels = 80
#frame_length_ms=50
#frame_shift_ms=12.5
hop_length = 256   #frame_shift_ms * sample_rate / 1000
win_length = 1024  #frame_length_ms * sample_rate / 1000
fmin = 40
min_level_db = -100
ref_level_db = 20


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=40.0,
        mel_fmax=None,
        min_level_db = -100,
        ref_level_db = 16
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db

    def forward(self, audio):
        p = self.n_fft // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = 20 * torch.log10(torch.clamp(mel_output, min=1e-5)) - self.ref_level_db
        log_mel_spec = torch.clamp((log_mel_spec - self.min_level_db) / (-self.min_level_db), min=0.0, max=1.0)
        return log_mel_spec


def convert_audio(wav_path):
    wav = load_wav(wav_path)
    mel = melspectrogram(wav).astype(np.float32)
    return mel.transpose(), wav

def load_wav(filename) :
    x = librosa.load(filename, sr=sample_rate)[0]
    return x

def read_wav(path):
    sr, wav = read(path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)
    return sr, wav

def save_wav(y, filename) :
    scipy.io.wavfile.write(filename, sample_rate, y)

mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x)) - 16

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)



def test():
    cal_mel = Audio2Mel()
    l1_loss = nn.L1Loss()
    samples = torch.randn(10, 1, 10000)
    g_outputs = torch.randn(10, 1, 10000)
    real_score = cal_mel(samples)
    fake_score = cal_mel(g_outputs)
    mel_loss = l1_loss(real_score, fake_score)
    print(mel_loss)


if __name__ == '__main__':
    test()
import gzip
import numpy as np
import librosa
from glob import glob
import os


class MNIST:
    def __init__(self, file_dir='../datasets/MNIST/'):
        self.img_file = file_dir + 'train-images-idx3-ubyte.gz'
        self.lbl_file = file_dir + 'train-labels-idx1-ubyte.gz'

        self.img = self.read_img()
        self.lbl = self.read_lbl()

    def read_img(self):
        with gzip.open(self.img_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28)

    def read_lbl(self):
        with gzip.open(self.lbl_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def data_load(self, idx=None):
        if idx is None:
            return self.img, self.lbl
        else:
            return self.img[idx], self.lbl[idx]

    def load_digits(self, digits: list):
        state = np.zeros_like(self.lbl, dtype=np.bool)
        for digit in digits:
            state += (self.lbl == digit)
        tag = np.where(state)
        return self.img[tag], self.lbl[tag]


def wav2db(wave, n_fft=512, hop_length=256):
    return 20. * np.log10(np.clip(np.abs(librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann', center=False) / n_fft), a_min=1e-6, a_max=None))


def lsd_l1_loss(true_db, pred_db):
    return np.mean(np.abs(true_db-pred_db), axis=0)


class MyData:
    def __init__(self, file_dir='../datasets/nsdtseaCustom/'):
        # init folder name
        self.clean_dir = file_dir + 'test_clean_16k/'
        self.noisy_dir = file_dir + 'test_noisy_16k/'
        self.short_dir = file_dir + 'sw2_512_1st_215_tw_noisy_phase/'
        self.long_dir = file_dir + 'sw2_1024_1st_206_tw_noisy_phase/'

        # init NSDTSEA file name data (ex: p232_001.wav)
        self.file_name = list()
        for temp_file in glob(self.clean_dir + '*.wav'):
            self.file_name.append(os.path.basename(temp_file))
        self.file_name.sort()

        # init data. If STFT all test file with N=512, 50% overlap, we'll get 128268 frame
        self.frames = np.zeros((128268, 257), dtype=np.float32)
        self.lbl = np.zeros(128268, dtype=np.int8)  # 0: short is better, 1: long is better
        self.read_data()

    def read_data(self, sample_rate=16000):
        frame_index = 0
        for idx, file in enumerate(self.file_name):
            clean = wav2db(librosa.load(self.clean_dir + file, sr=sample_rate)[0])
            short = wav2db(librosa.load(self.short_dir + file, sr=sample_rate)[0])
            long = wav2db(librosa.load(self.long_dir + file, sr=sample_rate)[0])
            length = clean.shape[-1]

            lsd_short = lsd_l1_loss(clean, short)
            lsd_long = lsd_l1_loss(clean, long)

            temp_lbl = (lsd_short > lsd_long).astype(np.int8)  # is long better?
            # noisy = wav2db(librosa.load(self.noisy_dir + file, sr=sample_rate)[0])

            self.frames[frame_index: frame_index + length] += np.transpose(long)
            self.lbl[frame_index: frame_index + length] += temp_lbl

            frame_index += length

    def data_load(self, idx=None):
        if idx is None:
            return self.frames, self.lbl[idx]
        else:
            return self.frames[idx], self.lbl[idx]

from glob import glob
import os
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import math
import numpy as np
from data.signal import signal_proc

class Amp2Db(object):

    def __init__(self, min_level_db=-100, stype="power"):
        self.stype = stype
        self.multiplier = 10. if stype == "power" else 20.
        if min_level_db is None:
            self.min_level = None
        else:
            min_level_db = -min_level_db if min_level_db > 0 else min_level_db
            self.min_level = torch.tensor(
                np.exp(min_level_db / self.multiplier * np.log(10))
            )

    def __call__(self, spec):
        if self.min_level is not None:
            spec_ = torch.max(spec, self.min_level)
        else:
            spec_ = spec
        spec_db = self.multiplier * torch.log10(spec_)
        return spec_db


class Normalize(object):

    def __init__(self, min_level_db=-100, ref_level_db=20):
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

    def __call__(self, spec):
        return torch.clamp(
            (spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1
        )


class Interpolate(object):
    def __init__(self, n_freqs=256, sr=44100, f_min=500, f_max=10000):
        self.n_freqs = n_freqs
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spec):
        n_fft = (spec.size(2) - 1) * 2

        if self.sr is not None and n_fft is not None:
            min_bin = int(max(0, math.floor(n_fft * self.f_min / self.sr)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.f_max / self.sr)))
            spec = spec[:, :, min_bin:max_bin]

        spec.unsqueeze_(dim=0)
        spec = F.interpolate(spec, size=(spec.size(2), self.n_freqs), mode="nearest")
        return spec.squeeze(dim=0)


class Spectrogram(object):
    def __init__(self, n_fft, hop_length, center=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        S = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            onesided=True,
            return_complex=False
        ).transpose(1, 2)
        Sp = S/(self.window.pow(2).sum().sqrt())
        Sp = Sp.pow(2).sum(-1)
        return Sp, S


class PaddedSubsequenceSampler(object):

    def __init__(self, sequence_length: int, dim: int = 0, random=True):
        assert isinstance(sequence_length, int)
        assert isinstance(dim, int)
        self.sequence_length = sequence_length
        self.dim = dim
        if random:
            self._sampler = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            self._sampler = lambda x: x // 2

    def _maybe_sample_subsequence(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length > sequence_length:
            start = self._sampler(sample_length - sequence_length)
            end = start + sequence_length
            indices = torch.arange(start, end, dtype=torch.long)
            return torch.index_select(spectrogram, self.dim, indices)
        return spectrogram

    def _maybe_pad(self, spectrogram, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length < sequence_length:
            start = self._sampler(sequence_length - sample_length)
            end = start + sample_length

            shape = list(spectrogram.shape)
            shape[self.dim] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)

            if self.dim == 0:
                padded_spectrogram[start:end] = spectrogram
            elif self.dim == 1:
                padded_spectrogram[:, start:end] = spectrogram
            elif self.dim == 2:
                padded_spectrogram[:, :, start:end] = spectrogram
            elif self.dim == 3:
                padded_spectrogram[:, :, :, start:end] = spectrogram
            return padded_spectrogram
        return spectrogram

    def __call__(self, spectrogram):
        spectrogram = self._maybe_pad(spectrogram)
        spectrogram = self._maybe_sample_subsequence(spectrogram)
        return spectrogram

def get_files(folder, ext, recursive=True):
    path = f"{folder}{os.sep}*{os.sep}{ext}"
    return glob(path, recursive=recursive)

def make_spectrogram_from_audio(file, n_fft=4096, hop=441, fmin=500, fmax=10000, freq_bins=256, sr=44100):
    y, _ = sf.read(file)
    if len(y.shape) == 2:
        y = np.mean(y, axis=1, keepdims=False)
    # elif len(y.shape) == 1:
    #     y = np.expand_dims(y, axis=1)
    return make_spectrogram(y, n_fft, hop, fmin, fmax, freq_bins, sr)


def make_spectrogram(a, n_fft=4096, hop=441, fmin=500, fmax=10000, freq_bins=256, sr=44100, seq_len=128):
    """
    :param a: A numpy array
    :param n_fft: FFT Size
    :param hop: Hop Size
    :return: Spectrogram of the size (n_fft/2 x (audio length / hop))
    """
    sp = signal_proc()

    perc_of_max_signal = 1.0
    min_thresh_detect = 0.05
    max_thresh_detect = 0.4
    t = Spectrogram(n_fft=n_fft, hop_length=hop)
    tensor = torch.from_numpy(a).unsqueeze(0)
    spectrogram = t(tensor)[0]
    spectrogram = Interpolate(n_freqs=freq_bins, f_min=fmin, f_max=fmax, sr=sr)(spectrogram)
    spectrogram = Amp2Db()(spectrogram)
    spectrogram = Normalize()(spectrogram)
    # spectrogram = PaddedSubsequenceSampler(sequence_length=128, dim=1)(spectrogram)
    ss, _ = sp.detect_strong_spectral_region(
        spectrogram=spectrogram,
        spectrogram_to_extract=spectrogram.clone(),
        n_fft=n_fft,
        target_len=seq_len,
        perc_of_max_signal=perc_of_max_signal,
        min_bin_of_interest=int(min_thresh_detect * spectrogram.shape[-1]),
        max_bin_of_inerest=int(max_thresh_detect * spectrogram.shape[-1]))
    spectrogram = ss[0]
    return spectrogram.squeeze().T.numpy()



if __name__ == '__main__':
    folder = "/home/alex/Desktop/general_graphics/audio/CTDC"
    output = "/home/alex/Desktop/general_graphics/img/CTDC"
    files = get_files(folder, ".wav", True)
    for f in files:
        signal, _ = sf.read(f)
        if len(signal.shape) == 2:
            signal = np.mean(signal, axis=1, keepdims=False)
        name = os.path.basename(f).replace(".wav", ".pdf")
        s_name = os.path.basename(f).replace(".wav", "_signal.pdf")
        s_f, s_ax = plt.subplots(dpi=60, figsize=(80 / 2.55, 40 / 2.55))
        s_ax.plot(signal)
        plt.axis('off')
        plt.savefig(os.path.join(output, s_name), bbox_inches='tight', pad_inches=0)
        plt.close(s_f)
        s = make_spectrogram_from_audio(f)
        fig, ax = plt.subplots(dpi=60, figsize=(80 / 2.55, 40 / 2.55))
        ax.imshow(s, origin="lower", interpolation=None)
        plt.axis('off')
        plt.savefig(os.path.join(output, name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    print()
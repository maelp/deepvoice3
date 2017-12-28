# -*- coding: utf-8 -*-
# #/usr/bin/python2

'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/deepvoice3
'''

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import os
import tqdm


def get_spectrograms(sound_file):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # Sequence length
    done = np.ones_like(mel[0, :]).astype(np.int32)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    mag = librosa.amplitude_to_db(mag)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, done, mag

def plot_spectrogram(wav_filename):
    from matplotlib import pyplot as plt

    _, _, mag = get_spectrograms(wav_filename)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    im = axes.imshow(mag.T)
    axes.axis('off')
    fig.subplots_adjust(right=0.8, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig('mag.png', format='png')

def test_reconstruct(wav_filename):
    '''
    Create a spectrogram and reconstruct the wav
    '''
    from utils import spectrogram2wav
    from snips_speech_utils.audio import Audio

    _, _, mag = get_spectrograms(wav_filename)
    audio = spectrogram2wav(mag)
    audio = Audio(data=audio, sample_rate=16000)
    audio.write('out.wav')
    audio.play()


def compute_features():
    # get list of filenames for tqdm
    filenames = [t[0] for t in hp.data.generator()]

    hp.data.create_paths()

    for fname in tqdm.tqdm(filenames):
        mel, dones, mag = get_spectrograms(fname)  # (n_mels, T), (1+n_fft/2, T) float32
        np.save(hp.data.mel_path(fname), mel)
        np.save(hp.data.done_path(fname), dones)
        np.save(hp.data.mag_path(fname), mag)


if __name__ == "__main__":
    compute_features()

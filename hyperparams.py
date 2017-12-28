# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/deepvoice3
'''
import math
import os
import io


FEATURES_PATH = 'features'


def get_Ty(duration, sr, hop_length, r):
    '''Calculates number of paddings for reduction'''
    def _roundup(x):
        return math.ceil(x * .1) * 10
    T = _roundup(duration*sr/hop_length)
    num_paddings = r - (T % r) if T % r != 0 else 0
    T += num_paddings
    return T


class Dataset(object):
    def __init__(self, dataset_name):
        self.features_path = os.path.join(FEATURES_PATH, dataset_name)
        self.mels_path = os.path.join(self.features_path, 'mels')
        self.dones_path = os.path.join(self.features_path, 'dones')
        self.mags_path = os.path.join(self.features_path, 'mags')

    def create_paths(self):
        for path in [self.mels_path, self.dones_path, self.mags_path]:
            if not(os.path.exists(path)):
                os.makedirs(path)

    def mel_path(self, fname):
        fname = os.path.basename(fname)
        return os.path.join(self.mels_path, fname + '.npy')

    def done_path(self, fname):
        fname = os.path.basename(fname)
        return os.path.join(self.dones_path, fname + '.npy')

    def mag_path(self, fname):
        fname = os.path.basename(fname)
        return os.path.join(self.mags_path, fname + '.npy')


class ArcticDataset(Dataset):
    def __init__(self, path):
        features_path = os.path.basename(path).strip()
        assert(len(features_path) > 0)
        super(ArcticDataset, self).__init__(features_path)
        self.path = path

    def generator(self):
        # assumes a `utts.data` file
        # ( arctic_a0001 "Author of the danger trail, Philip Steels, etc." )
        utts_path = os.path.join(self.path, 'utts.data')
        if not(os.path.exists(utts_path)):
            raise Exception('Invalid path for arctic dataset "{}", could not find "utts.data"'.format(path))
        for line in io.open(utts_path, 'r', encoding='utf-8'):
            line = line.strip()
            line = line[1:-1] # remove parentheses
            if len(line) > 0:
                id_, sent = line.split(None, 1)
                sent = sent[1:-1] # remove quotes
                fname = os.path.join(self.path, '{}.wav'.format(id_.strip()))
                yield (fname, sent)


class LJDataset(Dataset):
    def __init__(self, path):
        features_path = os.path.basename(path).strip()
        assert(len(features_path) > 0)
        super(LJDataset, self).__init__(features_path)
        self.path = path

    def generator(self):
        # assumes a `metadata.csv` file
        metadata_path = os.path.join(self.path, 'metadata.csv')
        if not(os.path.exists(metadata_path)):
            raise Exception('Invalid path for LJ speech dataset "{}", could not find "metadata.csv"'.format(path))
        for line in io.open(metadata_path, 'r', encoding='utf-8'):
            fname, _, sent = line.strip().split("|")
            yield (os.path.join(self.path, 'wavs', fname), sent)


class Hyperparams:
    '''Hyper parameters'''
    # signal processing
    sr = 16000 #22050 # Sampling rate.
    n_fft = 1024 #2048 # fft points (samples)
    frame_shift = 0.01 #0.0125 # seconds
    frame_length = 0.025 #0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    sharpening_factor = 1.4 # Exponent for amplifying the predicted magnitude
    n_iter = 200 #50 # Number of inversion iterations, higher is slower but better reconstruction
    preemphasis = 0.97 # or None
    max_db = 100
    ref_db = 50

    # Model
    r = 4 # Reduction factor
    dropout_rate = .2
    ## Enocder
    vocab_size = 32 # [PE a-z'.?]
    embed_size = 256 # == e
    enc_layers = 7
    enc_filter_size = 5
    enc_channels = 64 # == c
    ## Decoder
    dec_layers = 4
    dec_filter_size = 5
    attention_size = 128*2 # == a
    ## Converter
    converter_layers = 5*2
    converter_filter_size = 5
    converter_channels = 256 # == v

    sinusoid = False
    attention_win_size = 3

    # data
    data = ArcticDataset('data/arctic_slt')
    max_duration = 10.0 # seconds
    Tx = 180 # characters. maximum length of text.
    Ty = int(get_Ty(max_duration, sr, hop_length, r)) # Maximum length of sound (frames)

    # training scheme
    lr = 0.001
    logdir = "logdir"
    sampledir = 'samples'
    batch_size = 16
    max_grad_norm = 100.
    max_grad_val = 5.
    num_iterations = 500000

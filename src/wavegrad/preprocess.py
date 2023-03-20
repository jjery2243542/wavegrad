# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from glob import glob
from tqdm import tqdm

from wavegrad.params import params


def transform(filename, target_dir):
    audio, sr = T.load(filename)
    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')
    audio = torch.clamp(audio[0], -1.0, 1.0)

    hop = params.hop_samples
    win = hop * 4
    n_fft = 2**((win-1).bit_length())
    f_max = sr / 2.0
    mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)

    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        spec_filename = "/".join(filename.replace(".wav", ".spec.npy").split("/")[-3:])
        spec_path = os.path.join(target_dir, spec_filename)
        np.save(spec_path, spectrogram.cpu().numpy())


def main(args):
    filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
    target_dirs = [args.target_dir for _ in range(len(filenames))]

    for filename in tqdm(filenames):
        spec_dir = "/".join(filename.split("/")[-3:-1])
        spec_dir = os.path.join(args.target_dir, spec_dir)
        if not os.path.exists(spec_dir):
            os.makedirs(spec_dir)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filenames, target_dirs), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train WaveGrad')
    parser.add_argument('dir',
        help='directory containing .wav files for training')
    parser.add_argument('target_dir',
        help='directory to store spectrograms')
    main(parser.parse_args())

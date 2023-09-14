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

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser
import tqdm

from wavegrad.params import AttrDict, params as base_params
from wavegrad.model import WaveGrad
from wavegrad.inference import sde_predict
from wavegrad.noise_schedule import get_noise_schedule

models = {}

def main(args):
    with open(args.spectrogram_paths) as f_spec, open(args.wav_paths) as f_wav, open(args.outputs) as f_out:
        for line_spec, line_wav, line_out in tqdm.tqdm(zip(f_spec, f_wav, f_out)):
            spec_path = line_spec.strip()
            wav_path = line_wav.strip()
            out_path = line_out.strip()
            spectrogram = torch.from_numpy(np.load(spec_path)).T
            waveform, sr = torchaudio.load(wav_path)
            params = {}
            audio, sr = sde_predict(spectrogram, waveform=waveform, noise_steps=args.noise_steps, cond=args.cond, model_dir=args.model_dir, params=params)
            torchaudio.save(out_path, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by wavegrad.preprocess')
    parser.add_argument('model_dir',
        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('spectrogram_paths',
        help='path to a spectrogram file generated by wavegrad.preprocess')
    parser.add_argument('wav_paths',
        help='path to a wav file generated by wavegrad.preprocess')
    parser.add_argument('--cond', default=None, 
        help='conditioning value.')
    parser.add_argument('--noise_steps', default=500, type=int, 
        help='sde reverse steps.')
    parser.add_argument('--outputs', '-o', default='output.wav',
        help='output file name')
    main(parser.parse_args())

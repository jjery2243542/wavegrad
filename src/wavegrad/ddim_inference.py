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

from wavegrad.params import AttrDict, params as base_params
from wavegrad.model import WaveGrad
from wavegrad.noise_schedule import get_noise_schedule
import tqdm

models = {}

def set_timestep(num_inference_steps, num_training_steps=1000, steps_offset=0):
    step_ratio = num_training_steps // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    timesteps += steps_offset
    return timesteps


def _get_variance(alphas_cumprod, final_alpha_cumprod, timestep, prev_timestep):
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance

def ddim_predict(spectrogram, eta=0.0, num_inference_steps=100, num_training_steps=1000, set_alpha_to_one=True, model_dir=None, params=None, device=torch.device('cuda'), cond=None, clip_every_steps=False):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        params = checkpoint["params"]
        model = WaveGrad(AttrDict(params)).to(device)
        #model = WaveGrad(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)

    timesteps = set_timestep(num_inference_steps)
    timesteps = torch.from_numpy(timesteps).to(device)

    with torch.no_grad():
        #beta = get_noise_schedule(model.params.noise_schedule)
        beta = get_noise_schedule("linear")
 
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        final_alpha_cumprod = torch.tensor(1.0).to(device) if set_alpha_to_one else alpha_cum[0]

        # Expand rank 2 tensors by adding a batch dimension.
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)

        audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
        noise_scale = (alpha_cum**0.5).float().unsqueeze(1).to(device)

        if cond is not None:
            cond = torch.IntTensor([int(cond)]).to(device)

        for timestep in timesteps:
            model_output  = model(audio, spectrogram, noise_scale[timestep], cond=cond).squeeze(1)
            prev_timestep = timestep - num_training_steps // num_inference_steps

            alpha_prod_t = alpha_cum[timestep]
            alpha_prod_t_prev = alpha_cum[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (audio - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
            audio = torch.clamp(audio, -1.0, 1.0)

            variance = _get_variance(alpha_cum, final_alpha_cumprod, timestep, prev_timestep)
            std_dev_t = eta * variance ** (0.5)
            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
            prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            if eta > 0:
                variance_noise = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
                variance = std_dev_t * variance_noise
                prev_sample = prev_sample + variance
            audio = prev_sample

    return audio, model.params.sample_rate


def main(args):
    with open(args.spectrogram_paths) as f_spec, open(args.outputs) as f_out:
        for line_spec, line_out in tqdm.tqdm(zip(f_spec, f_out)):
            spec_path = line_spec.strip()
            out_path = line_out.strip()
            spectrogram = torch.from_numpy(np.load(spec_path)).T
            params = {}
            audio, sr = ddim_predict(spectrogram, num_inference_steps=args.infer_steps, cond=args.cond, model_dir=args.model_dir, params=params, clip_every_steps=args.clip_every_steps)
            torchaudio.save(out_path, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by wavegrad.preprocess')
    parser.add_argument('model_dir',
        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('spectrogram_paths',
        help='path to a spectrogram file generated by wavegrad.preprocess')
    parser.add_argument('--cond', default=None, 
        help='conditioning value.')
    parser.add_argument('--infer_steps', default=25, type=int, 
        help='number of inference steps.')
    parser.add_argument('--clip_every_steps', action="store_true", default=False,
        help='clip every steps.')
    parser.add_argument('--outputs', '-o', default='output.wav',
        help='output file name')
    main(parser.parse_args())

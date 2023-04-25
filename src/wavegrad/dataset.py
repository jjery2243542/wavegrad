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
import random
import torch
import torch.nn.functional as F
import torchaudio
import sys

from glob import glob
from torch.utils.data.distributed import DistributedSampler

# TODO: fix import error
#try:
#    ckpt_path = "/share/data/speech/jjery2243542/checkpoints/avhubert/large_vox_iter5.pt"
#    user_dir = "/home-nfs/jjery2243542/av_hubert/avhubert"
#    extractor = FeatureExtractor(ckpt_path, user_dir)
#    # sample 
#    r = random.random()
#    if r < self.sample_probs[0]:
#        feats = extractor.extract_feature(audio_path=audio_filename, video_path=None)
#    elif r < self.sample_probs[0] + self.sample_probs[1]:
#        feats = extractor.extract_feature(audio_path=None, video_path=video_filename)
#    else:
#        feats = extractor(audio_path=audio_filename, video_path=video_filename)
#            start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
#            end = start + self.params.crop_mel_frames
#            record['spectrogram'] = record['spectrogram'][start:end].T

#            start *= samples_per_frame
#            end *= samples_per_frame
#            record['audio'] = record['audio'][start:end]
#            record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
#
#        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
#        spectrogram = torch.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record], dim=0)
        #spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
#except:
#    print("cannot load feature extrator")
#    exit(0)
#
#    #ckpt_path = "/share/data/speech/jjery2243542/checkpoints/avhubert/large_vox_iter5.pt"
#    #user_dir = "/home-nfs/jjery2243542/av_hubert/avhubert"
#    #extractor = FeatureExtractor(ckpt_path, user_dir)

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, wav_file_lists, mp4_file_lists, root_dir):
        super().__init__()

        self.wav_mp4_pairs = []
        for wav_file_list, mp4_file_list in zip(wav_file_lists, mp4_file_lists):
            with open(wav_file_list) as f_wav, open(mp4_file_list) as f_mp4:
                for wav_line, mp4_line in zip(f_wav, f_mp4):
                    audio_path = os.path.join(root_dir, wav_line.strip())
                    video_path = os.path.join(root_dir, mp4_line.strip())
                    self.wav_mp4_pairs.append((audio_path, video_path))

    def __len__(self):
        return len(self.wav_mp4_pairs)

    def __getitem__(self, idx):
        audio_path = self.wav_mp4_pairs[idx][0]
        video_path = self.wav_mp4_pairs[idx][1]
        signal, _ = torchaudio.load(audio_path)

        return {
            'audio_path': audio_path,
            'video_path': video_path,
            'audio': signal[0],
        }

class NumpyFileDataset(torch.utils.data.Dataset):
    def __init__(self, wav_file_list, npy_file_lists, wav_root_dir, feat_root_dir):
        # npy_files in an order of a, v, av
        super().__init__()

        self.data = []
        with open(wav_file_list) as f_wav, open(npy_file_lists[0]) as f_a, open(npy_file_lists[1]) as f_v, open(npy_file_lists[2]) as f_av:
            for wav_line, a_line, v_line, av_line in zip(f_wav, f_a, f_v, f_av):
                data_dict = {"wav": os.path.join(wav_root_dir, wav_line.strip()), 
                        "a": os.path.join(feat_root_dir, a_line.strip()), 
                        "v": os.path.join(feat_root_dir, v_line.strip()), 
                        "av": os.path.join(feat_root_dir, av_line.strip()),
                    }
                self.data.append(data_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path = self.data[idx]["wav"]
        signal, _ = torchaudio.load(wav_path)

        a_path = self.data["idx"]["a"]
        v_path = self.data["idx"]["v"]
        av_path = self.data["idx"]["av"]

        a_features = np.load(a_path)
        v_features = np.load(v_path)
        av_features = np.load(av_path)

        # cut to the same length
        min_len = min(a_features.shape[0], v_features.shape[0], av_features.shape[0])
        a_features = a_features[:min_len, :]
        v_features = v_features[:min_len, :]
        av_features = av_features[:min_len, :]

        return {
            'filename': wav_path,
            'audio': signal[0],
            'a_features': a_features,
            'v_features': v_features,
            'av_features': av_features,
        }

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, spec_dir):
        super().__init__()
        self.filenames = []
        self.spec_dir = spec_dir
        for path in paths:
            self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = os.path.join(self.spec_dir, "/".join(audio_filename.split("/")[-2:]).replace(".wav", ".npy"))
        #spec_filename = f'{audio_filename}.spec.npy'
        signal, _ = torchaudio.load(audio_filename)
        spectrogram = np.load(spec_filename)
        return {
            'filename': self.filenames[idx],
            'audio': signal[0],
            'spectrogram': spectrogram,
        }

class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        for record in minibatch:
            # Filter out records that aren't long enough.
            if len(record['a_features']) < self.params.crop_mel_frames:
                del record['audio']
                continue

            start = random.randint(0, record['a_features'].shape[0] - self.params.crop_mel_frames)
            end = start + self.params.crop_mel_frames
            record['a_features'] = record['a_features'][start:end].T
            record['v_features'] = record['v_features'][start:end].T
            record['av_features'] = record['av_features'][start:end].T

            start *= samples_per_frame
            end *= samples_per_frame
            record['audio'] = record['audio'][start:end]
            record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        a_features = np.stack([record['a_features'] for record in minibatch if 'audio' in record])
        v_features = np.stack([record['v_features'] for record in minibatch if 'audio' in record])
        av_features = np.stack([record['av_features'] for record in minibatch if 'audio' in record])
        return {
            'audio': torch.from_numpy(audio),
            'a_features': torch.from_numpy(a_features),
            'v_features': torch.from_numpy(v_features),
            'av_features': torch.from_numpy(av_features),
        }

class PathCollator(Collator):
    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        batch = {"audio_paths": [], "video_paths": [], "audios": []}
        for record in minibatch:
            # Filter out records that aren't long enough.
            # add one for safety
            if len(record['audio']) >= (self.params.crop_mel_frames + 1) / self.params.frame_rate * self.params.sample_rate:
                batch["audio_paths"].append(record["audio_path"])
                batch["video_paths"].append(record["video_path"])
                batch["audios"].append(record["audio"])
        return batch

def crop_features(params, features, audios, audio_starts, audio_size):
    samples_per_frame = params.hop_samples
    cropped = {"feature":[], "audio":[]}
    for feature, audio, audio_start in zip(features, audios, audio_starts):
        # Filter out records that aren't long enough.
        start = random.randint(0, feature.shape[0] - params.crop_mel_frames)
        end = start + params.crop_mel_frames
        cropped['feature'].append(feature[start:end])

        # the features are extracted from cropped audio, so have to add audio_start
        start = (start + audio_start) * samples_per_frame
        end = (end + audio_start) * samples_per_frame
        cropped_audio = audio[start:end]
        cropped_audio = F.pad(cropped_audio, (0, (end - start) - len(cropped_audio)))
        cropped["audio"].append(cropped_audio)
    features = torch.stack(cropped["feature"], dim=0).transpose(1, 2)
    audio = torch.stack(cropped["audio"], dim=0)
    return features, audio

def from_file_lists(wav_file_list, npy_file_lists, params, wav_root_dir, feat_root_dir, is_distributed=False, is_valid=False):
    dataset = NumpyFileDataset(wav_file_list, npy_file_lists, wav_root_dir, feat_root_dir)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size if not is_valid else params.valid_batch_size,
        collate_fn=PathCollator(params).collate,
        shuffle=not is_distributed,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
        num_workers=os.cpu_count())

def from_path(data_dirs, spec_dir, params, is_distributed=False):
    dataset = NumpyDataset(data_dirs, spec_dir)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
        num_workers=os.cpu_count())

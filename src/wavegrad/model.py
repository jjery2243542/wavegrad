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
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln
import einops 

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = einops.rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
        """
        Arguments:
          x:
              (shape: [N,C,T], dtype: float32)
          noise_level:
              (shape: [N], dtype: float32)

        Returns:
          noise_level:
              (shape: [N,C,T], dtype: float32)
        """
        N = x.shape[0]
        T = x.shape[2]
        return (x + self._build_encoding(noise_level)[:, :, None])

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, noise_scale):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


class LUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 5

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[4], padding=dilation[4]),
        ])

    def forward(self, x, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[2](block3)

        x = x + block3
        return x


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3
        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual

class LDBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilations):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilations[0], padding=dilations[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilations[1], padding=dilations[1]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilations[2], padding=dilations[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilations[3], padding=dilations[3]),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual

class WaveGrad(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        if self.params.model_size == "base":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 4),
                DBlock(256, 512, 4),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 512),
                FiLM(512, 512),
            ])
            self.upsample = nn.ModuleList([
                UBlock(1024, 512, 5, [1, 2, 1, 2]),
                UBlock(512, 512, 4, [1, 2, 1, 2]),
                UBlock(512, 256, 4, [1, 2, 4, 8]),
                UBlock(256, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
            ])
        elif self.params.model_size == "large":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 2),
                DBlock(256, 256, 2),
                DBlock(256, 512, 2),
                DBlock(512, 512, 2),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 256),
                FiLM(256, 512),
                FiLM(512, 512),
                FiLM(512, 1024),
            ])
            self.upsample = nn.ModuleList([
                UBlock(1024, 1024, 5, [1, 2, 1, 2]),
                UBlock(1024, 512, 2, [1, 2, 1, 2]),
                UBlock(512, 512, 2, [1, 2, 1, 2]),
                UBlock(512, 256, 2, [1, 2, 4, 8]),
                UBlock(256, 256, 2, [1, 2, 4, 8]),
                UBlock(256, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
            ])
        elif self.params.model_size == "xlarge":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                LDBlock(32, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 256, 2, dilations=[1,2,4,8]),
                LDBlock(256, 256, 2, dilations=[1,2,4,8]),
                DBlock(256, 512, 2),
                DBlock(512, 512, 2),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 256),
                FiLM(256, 512),
                FiLM(512, 512),
                FiLM(512, 1024),
            ])
            self.upsample = nn.ModuleList([
                UBlock(1024, 1024, 5, [1, 2, 1, 2]),
                UBlock(1024, 512, 2, [1, 2, 1, 2]),
                UBlock(512, 512, 2, [1, 2, 1, 2]),
                LUBlock(512, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
            ])
        elif self.params.model_size == "xlarge2":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                LDBlock(32, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 256, 2, dilations=[1,2,4,8]),
                LDBlock(256, 256, 2, dilations=[1,2,4,8]),
                DBlock(256, 512, 2),
                DBlock(512, 512, 2),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 256),
                FiLM(256, 512),
                FiLM(512, 512),
                FiLM(512, 1024),
            ])
            self.upsample = nn.ModuleList([
                LUBlock(1024, 1024, 5, [1, 2, 1, 2, 4]),
                LUBlock(1024, 512, 2, [1, 2, 1, 2, 4]),
                LUBlock(512, 512, 2, [1, 2, 1, 2, 4]),
                LUBlock(512, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
            ])
        elif self.params.model_size == "xlarge3":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                LDBlock(32, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 128, 2, dilations=[1,2,4,8]),
                LDBlock(128, 256, 2, dilations=[1,2,4,8]),
                LDBlock(256, 256, 2, dilations=[1,2,4,8]),
                LDBlock(256, 512, 2, dilations=[1,2,4,8]),
                LDBlock(512, 512, 2, dilations=[1,2,4,8]),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 256),
                FiLM(256, 512),
                FiLM(512, 512),
                FiLM(512, 1024),
            ])
            self.upsample = nn.ModuleList([
                LUBlock(1024, 1024, 5, [1, 2, 1, 2, 4]),
                LUBlock(1024, 512, 2, [1, 2, 1, 2, 4]),
                LUBlock(512, 512, 2, [1, 2, 1, 2, 4]),
                LUBlock(512, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 256, 2, [1, 2, 4, 8, 16]),
                LUBlock(256, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
                LUBlock(128, 128, 2, [1, 2, 4, 8, 16]),
            ])
        elif self.params.model_size == "large_long_dilation":
            self.downsample = nn.ModuleList([
                Conv1d(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 2),
                DBlock(256, 256, 2),
                DBlock(256, 512, 2),
                DBlock(512, 512, 2),
            ])
            self.film = nn.ModuleList([
                FiLM(32, 128),
                FiLM(128, 128),
                FiLM(128, 128),
                FiLM(128, 256),
                FiLM(256, 256),
                FiLM(256, 512),
                FiLM(512, 512),
                FiLM(512, 1024),
            ])
            self.upsample = nn.ModuleList([
                UBlock(1024, 1024, 5, [1, 2, 1, 2]),
                UBlock(1024, 512, 2, [1, 2, 1, 2]),
                UBlock(512, 512, 2, [1, 2, 4, 8]),
                UBlock(512, 256, 2, [1, 2, 4, 8]),
                UBlock(256, 256, 2, [1, 2, 4, 8]),
                UBlock(256, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
                UBlock(128, 128, 2, [1, 2, 4, 8]),
            ])
        av_dim = 1024 if self.params.av_model_size == "large" else 768

        if not hasattr(self.params, "cond_norm") or self.params.cond_norm == "ln":
            self.norm = torch.nn.LayerNorm(av_dim)
        elif self.params.cond_norm == "bn":
            self.norm = torch.nn.BatchNorm1d(av_dim)
        else:
            raise ValueError(f"cond_norm has to be one of [ln, bn], got {self.params.cond_norm}")

        if hasattr(self.params, "self_attn") and self.params.self_attn:
            import einops
            embed_dim = 512 
            self.pre_norm = torch.nn.LayerNorm(embed_dim)
            self.attn = Attention(dim=embed_dim, heads=self.params.num_heads, dim_head=embed_dim // self.params.num_heads)

        self.first_conv = Conv1d(av_dim, 1024, 1)
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, spectrogram, noise_scale):
        x = audio.unsqueeze(1)
        downsampled = []
        for film, layer in zip(self.film, self.downsample):
            x = layer(x)
            downsampled.append(film(x, noise_scale))
        if not hasattr(self.params, "cond_norm") or self.params.cond_norm == "ln":
            spectrogram = self.norm(spectrogram.transpose(1, 2)).transpose(1, 2)
        elif self.params.cond_norm == "bn":
            spectrogram = self.norm(spectrogram)

        x = self.first_conv(spectrogram)
        for i, (layer, (film_shift, film_scale)) in enumerate(zip(self.upsample, reversed(downsampled))):
            x = layer(x, film_shift, film_scale)
            if hasattr(self, "attn") and i == 0 and self.params.model_size == "base":
                x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
                x = self.attn(x_norm) + x
            elif hasattr(self, "attn") and i == 1 and (self.params.model_size == "large" or self.params.model_size == "large_long_dilation"):
                x_norm = self.pre_norm(x.transpose(1, 2)).transpose(1, 2)
                x = self.attn(x_norm) + x

        x = self.last_conv(x)
        return x

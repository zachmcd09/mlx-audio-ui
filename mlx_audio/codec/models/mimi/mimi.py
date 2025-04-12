# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from .modules import (
    ConvDownsample1d,
    ConvTrUpsample1d,
    ProjectedTransformer,
    SeanetConfig,
    SeanetDecoder,
    SeanetEncoder,
    SplitResidualVectorQuantizer,
    TransformerConfig,
)


@dataclass
class MimiConfig:
    channels: int
    sample_rate: float
    frame_rate: float
    renormalize: bool
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int


def mimi_202407(num_codebooks: int) -> MimiConfig:
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=0.01,
        positional_embedding="rope",
        use_conv_bias=True,
        gating=False,
        norm="layer_norm",
        context=250,
        max_period=10000,
        max_seq_len=8192,
        kv_repeat=1,
        dim_feedforward=2048,
        conv_layout=True,
        use_conv_block=False,
        cross_attention=False,
        conv_kernel_size=3,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        renormalize=True,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks,
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    def __init__(self, cfg: MimiConfig):
        super().__init__()
        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer,
            input_dim=dim,
            output_dims=[dim],
        )
        self.downsample = ConvDownsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.upsample = ConvTrUpsample1d(
            stride=downsample_stride,
            dim=dim,
            causal=True,
        )
        self.encoder_cache = self.encoder_transformer.make_cache()
        self.decoder_cache = self.decoder_transformer.make_cache()

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        for c in self.encoder_cache:
            c.reset()

    def encode(self, xs: mx.array) -> mx.array:
        self.encoder.reset_state()
        for c in self.encoder_cache:
            c.reset()
        xs = self.encoder(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample(xs)
        return self.quantizer.encode(xs)

    def decode(self, xs: mx.array) -> mx.array:
        self.decoder.reset_state()
        for c in self.decoder_cache:
            c.reset()
        xs = self.quantizer.decode(xs)
        xs = self.upsample(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        return self.decoder(xs)

    def encode_step(self, xs: mx.array) -> mx.array:
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=self.encoder_cache)[0]
        xs = self.downsample.step(xs)
        xs = self.quantizer.encode(xs)
        return xs

    def decode_step(self, xs: mx.array) -> mx.array:
        xs = self.quantizer.decode(xs)
        xs = self.upsample.step(xs)
        xs = self.decoder_transformer(xs, cache=self.decoder_cache)[0]
        xs = self.decoder.step(xs)
        return xs

    def warmup(self):
        pcm = mx.zeros((1, 1, 1920 * 4))
        codes = self.encode(pcm)
        pcm_out = self.decode(codes)
        mx.eval(pcm_out)

    def load_pytorch_weights(
        self,
        file: str,
        strict: bool = True,
    ) -> nn.Module:
        weights = []
        # Rename loop variables to avoid potential conflicts
        for orig_key, orig_value in mx.load(file).items():
            value: mx.array = orig_value
            key: str = ".".join([s.removeprefix("_") for s in orig_key.split(".")]) # Use orig_key here
            if key.startswith("encoder.model."):
                key = key.replace("encoder.model.", "encoder.")
            if key.startswith("decoder.model."):
                key = key.replace("decoder.model.", "decoder.")
            if key.endswith(".in_proj_weight"):
                key = key.replace(".in_proj_weight", ".in_proj.weight")
            if key.endswith(".linear1.weight"):
                key = key.replace(".linear1.weight", ".gating.linear1.weight")
            if key.endswith(".linear2.weight"):
                key = key.replace(".linear2.weight", ".gating.linear2.weight")
            # Awfully hardcoded matching between the pytorch layers and their mlx equivalent :(
            for layerIdx, decoderIdx in enumerate([2, 5, 8, 11]):
                key = key.replace(
                    f"decoder.{decoderIdx}.", f"decoder.layers.{layerIdx}.upsample."
                )
                key = key.replace(
                    f"decoder.{decoderIdx + 1}.",
                    f"decoder.layers.{layerIdx}.residuals.0.",
                )
            for layerIdx, encoderIdx in enumerate([1, 4, 7, 10]):
                key = key.replace(
                    f"encoder.{encoderIdx}.", f"encoder.layers.{layerIdx}.residuals.0."
                )
                key = key.replace(
                    f"encoder.{encoderIdx + 2}.",
                    f"encoder.layers.{layerIdx}.downsample.",
                )

            key = key.replace("decoder.0.", "decoder.init_conv1d.")
            key = key.replace("decoder.14.", "decoder.final_conv1d.")
            key = key.replace("encoder.0.", "encoder.init_conv1d.")
            key = key.replace("encoder.14.", "encoder.final_conv1d.")
            key = key.replace(".block.1.", ".block.0.")
            key = key.replace(".block.3.", ".block.1.")

            # PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
            if (
                key.endswith(".conv.weight")
                or key.endswith(".output_proj.weight")
                or key.endswith(".input_proj.weight")
            ):
                value = value.swapaxes(-1, -2) # Use the hinted 'value'
            # PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
            if key.endswith(".convtr.weight"):
                value = value.transpose(1, 2, 0) # Use the hinted 'value'
            weights.append((key, value)) # Append modified key and potentially modified value
        return self.load_weights(weights, strict=strict)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
    ) -> nn.Module:
        cfg = mimi_202407(32)
        model = cls(cfg)
        model_file = hf_hub_download(repo_id, filename)
        model.load_pytorch_weights(model_file, strict=True)
        return model

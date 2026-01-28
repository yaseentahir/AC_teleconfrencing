# -*- coding: utf-8 -*-

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio


class AcousticEncoder:
    """
    Neutral acoustic latent encoder.
    Backend implementation is abstracted.
    """
    def __init__(self, device="cpu", compression_rate=6.0):
        self.device = device
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(compression_rate)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, audio_np, sr=16000):
        x = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        x = convert_audio(x, sr, 24000, 1)
        x = x.to(self.device)
        return self.model.encode(x)


class AcousticDecoder:
    """
    Neutral acoustic latent decoder.
    Backend implementation is abstracted.
    """
    def __init__(self, device="cpu", compression_rate=6.0):
        self.device = device
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(compression_rate)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, latent_repr):
        return self.model.decode(latent_repr)

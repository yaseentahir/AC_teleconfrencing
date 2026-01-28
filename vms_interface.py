import struct
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------
# Configuration
# ---------------------------
IN_SR = 16000
MODEL_SR = 24000
TARGET_BANDWIDTH = 6.0

# For AMS reliability and to avoid torchaudio GPU/CPU kernel issues, force CPU.
_DEFAULT_DEVICE = "cpu"

_model = None
_device = None


def _get_model(device: str | None = None):
    """Lazy-load EnCodec model once."""
    global _model, _device
    if device is None:
        device = _DEFAULT_DEVICE

    if _model is not None and _device == device:
        return _model, _device

    from encodec import EncodecModel  # lazy import

    _model = EncodecModel.encodec_model_24khz()
    _model.set_target_bandwidth(TARGET_BANDWIDTH)
    _model.to(device)
    _model.eval()
    _device = device
    return _model, _device


# ---------------------------
# Audio helpers
# ---------------------------
def _to_float_mono(x: np.ndarray) -> np.ndarray:
    """Ensure 1D float32 audio in [-1, 1]. Handles int16 PCM safely."""
    x = np.asarray(x)

    # Downmix if needed
    if x.ndim == 2:
        # interpret as (T, C) or (C, T)
        if x.shape[0] >= x.shape[1]:
            x = x.mean(axis=1)
        else:
            x = x.mean(axis=0)
    elif x.ndim != 1:
        x = x.reshape(-1)

    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)

    return np.clip(x, -1.0, 1.0)


def _resample_1d_linear(y: torch.Tensor, in_sr: int, out_sr: int) -> torch.Tensor:
    """Linear resample 1D tensor (T,) -> (T_out,). Works on CPU/GPU."""
    if in_sr == out_sr:
        return y
    if y.dim() != 1:
        y = y.reshape(-1)

    y3 = y.view(1, 1, -1)
    T = int(y3.shape[-1])
    T_out = int(round(T * (out_sr / in_sr)))
    y_out = F.interpolate(y3, size=T_out, mode="linear", align_corners=False)
    return y_out.view(-1)


def _fix_length_1d(y: torch.Tensor, n: int) -> torch.Tensor:
    """Trim/pad to exactly n samples."""
    cur = int(y.numel())
    if cur > n:
        return y[:n]
    if cur < n:
        return F.pad(y, (0, n - cur))
    return y


# ---------------------------
# Packet format (compact, scale-aware)
# ---------------------------
# Header:
#   magic: 4 bytes  b"VMS1"
#   ver: uint8
#   rsv: uint8
#   n_orig: uint16
#   nseg: uint8
#   pad: 3 bytes
#
# Each segment:
#   n_q: uint8
#   has_scale: uint8     (0 or 1)   <-- IMPORTANT FIX
#   T: uint16
#   scale: float32       (present even if has_scale=0; ignored then)
#   codes: int16[n_q*T]  row-major for (n_q, T)

_MAGIC = b"VMS1"
_VERSION = 1
_HDR = struct.Struct("<4sBBH B 3s")
_SEG = struct.Struct("<BBHf")  # n_q, has_scale, T, scale


def _pack_latent(latent: List[Tuple[torch.Tensor, Optional[torch.Tensor]]], n_orig: int) -> bytes:
    out = bytearray()
    nseg = len(latent)
    out += _HDR.pack(_MAGIC, _VERSION, 0, int(n_orig) & 0xFFFF, int(nseg) & 0xFF, b"\x00\x00\x00")

    for (codes, scale) in latent:
        codes = codes.detach().to("cpu")
        if codes.dim() != 3:
            raise ValueError(f"Unexpected codes shape: {tuple(codes.shape)}")
        if codes.shape[0] != 1:
            codes = codes[:1]

        n_q = int(codes.shape[1])
        T = int(codes.shape[2])

        # FIX: scale can be None depending on EnCodec version/bandwidth mode
        if scale is None:
            has_scale = 0
            scale_val = 1.0
        else:
            has_scale = 1
            scale_cpu = scale.detach().to("cpu").float().reshape(-1)
            scale_val = float(scale_cpu[0].item()) if scale_cpu.numel() else 1.0

        out += _SEG.pack(n_q & 0xFF, has_scale & 0xFF, T & 0xFFFF, float(scale_val))

        # codes are integer indices; pack as int16
        codes_i16 = codes.squeeze(0).short().contiguous().numpy()  # (n_q, T)
        out += codes_i16.tobytes(order="C")

    return bytes(out)


def _unpack_latent(buf: bytes) -> Tuple[int, List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]:
    if len(buf) < _HDR.size:
        raise ValueError("Packet too small")

    magic, ver, _rsv, n_orig, nseg, _pad = _HDR.unpack_from(buf, 0)
    if magic != _MAGIC:
        raise ValueError("Bad packet magic")
    if ver != _VERSION:
        raise ValueError(f"Unsupported packet version: {ver}")

    offset = _HDR.size
    latent: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []

    for _ in range(int(nseg)):
        if len(buf) < offset + _SEG.size:
            raise ValueError("Truncated segment header")

        n_q, has_scale, T, scale_val = _SEG.unpack_from(buf, offset)
        offset += _SEG.size

        codes_n = int(n_q) * int(T)
        codes_bytes = codes_n * 2
        if len(buf) < offset + codes_bytes:
            raise ValueError("Truncated codes data")

        codes_np = np.frombuffer(buf, dtype=np.int16, count=codes_n, offset=offset).reshape(1, int(n_q), int(T))
        offset += codes_bytes

        # decode expects integer indices (typically long)
        codes_t = torch.from_numpy(codes_np.astype(np.int64, copy=False))

        # FIX: if original encode had scale=None, pass scale=None back
        if int(has_scale) == 0:
            scale_t = None
        else:
            scale_t = torch.tensor([[[float(scale_val)]]], dtype=torch.float32)

        latent.append((codes_t, scale_t))

    return int(n_orig), latent


# ---------------------------
# AMS required functions
# ---------------------------
def my_encoder_logic(audio_frame: np.ndarray) -> bytes:
    """
    audio_frame: (N,) at 16k. float32 [-1,1] or int16 PCM.
    returns: compressed bytes
    """
    x = _to_float_mono(audio_frame)
    n_orig = int(x.shape[0])

    model, dev = _get_model()

    xt16 = torch.from_numpy(x).to(dev).view(1, 1, -1)

    # Safe resample to 24k (no torchaudio)
    xt24_1d = _resample_1d_linear(xt16.view(-1), IN_SR, MODEL_SR)
    xt24 = xt24_1d.view(1, 1, -1)

    with torch.no_grad():
        latent = model.encode(xt24)  # list of (codes, scale) where scale may be None

    return _pack_latent(latent, n_orig=n_orig)


def my_decoder_logic(compressed_bytes: bytes) -> np.ndarray:
    """
    compressed_bytes: bytes
    returns: (N,) float32 at 16k
    """
    n_orig, latent_cpu = _unpack_latent(compressed_bytes)
    model, dev = _get_model()

    latent = []
    for (codes_t, scale_t) in latent_cpu:
        if scale_t is None:
            latent.append((codes_t.to(dev), None))
        else:
            latent.append((codes_t.to(dev), scale_t.to(dev)))

    with torch.no_grad():
        y24 = model.decode(latent)  # (1,1,T24)

    y24_1d = y24.view(-1)

    # Safe resample back to 16k
    y16_1d = _resample_1d_linear(y24_1d, MODEL_SR, IN_SR)

    # Force exact original frame length
    y16_1d = _fix_length_1d(y16_1d, n_orig)

    return y16_1d.detach().cpu().numpy().astype(np.float32, copy=False)

# ams_interface.py
# =========================================================
# Audio Encoder / Decoder Interface for AMS Deployment
# (Fixed for streamed / frame-based AMS execution)
# =========================================================

import numpy as np
import pickle
import torch

from neural_latent import AcousticEncoder, AcousticDecoder

# =========================================================
# Global initialization (AMS loads module once)
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

_encoder = AcousticEncoder(device=device)
_decoder = AcousticDecoder(device=device)

# =========================================================
# AMS FRAME SETTINGS
# =========================================================
# AMS commonly feeds 20 ms frames at 16 kHz => 320 samples.
# If your AMS config uses a different frame size, change this.
EXPECTED_FRAME_SAMPLES = 320

# If AMS sometimes sends bigger blocks, we’ll chunk them.
ALLOW_CHUNKING = True

# =========================================================
# Helpers
# =========================================================
def _ensure_mono_f32(audio_frame: np.ndarray) -> np.ndarray:
    """Convert AMS audio frame to mono float32 (N,) in [-1, 1]."""
    if not isinstance(audio_frame, np.ndarray):
        audio_frame = np.asarray(audio_frame)

    # (N,1) or (N,C) -> take first channel
    if audio_frame.ndim == 2:
        audio_frame = audio_frame[:, 0]

    audio_frame = audio_frame.astype(np.float32, copy=False)

    # Safety clamp (some pipelines can slightly exceed [-1,1])
    np.clip(audio_frame, -1.0, 1.0, out=audio_frame)
    return audio_frame


def _fit_frame(x: np.ndarray, n: int) -> np.ndarray:
    """Pad/truncate to exactly n samples."""
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    # pad with zeros
    out = np.zeros((n,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _maybe_reset(model) -> None:
    """Reset model state if it supports it."""
    # Common patterns: reset(), reset_state(), clear(), etc.
    for name in ("reset", "reset_state", "clear", "clear_state"):
        fn = getattr(model, name, None)
        if callable(fn):
            fn()
            return


def _latent_to_bytes(latent) -> bytes:
    """
    Make latent AMS-safe:
      - detach from graph
      - move to CPU
      - convert to numpy (picklable, stable)
    """
    if torch.is_tensor(latent):
        latent = latent.detach().cpu()

        # If it's scalar or tensor, convert to numpy for stable pickle
        latent = latent.numpy()
    return pickle.dumps(latent, protocol=pickle.HIGHEST_PROTOCOL)


def _bytes_to_latent(b: bytes):
    """Restore latent from bytes and move to the correct device."""
    latent = pickle.loads(b)

    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)

    if torch.is_tensor(latent):
        latent = latent.to(device)

    return latent


# =========================================================
# REQUIRED BY AMS WRAPPER
# =========================================================
def my_encoder_logic(audio_frame: np.ndarray) -> bytes:
    """
    Parameters
    ----------
    audio_frame : np.ndarray
        Shape: (N,) or (N,1)
        Dtype: float32 (or convertible)
        Range: [-1, 1]

    Returns
    -------
    bytes
        Compressed audio bytes for AMS
    """
    x = _ensure_mono_f32(audio_frame)

    # Ensure deterministic per-call behavior in streamed environment
    _maybe_reset(_encoder)

    if not ALLOW_CHUNKING:
        x = _fit_frame(x, EXPECTED_FRAME_SAMPLES)
        latent = _encoder.forward(x)
        return _latent_to_bytes(latent)

    # If AMS sends N != expected, chunk into expected-size frames
    if x.shape[0] == EXPECTED_FRAME_SAMPLES:
        latent = _encoder.forward(x)
        return _latent_to_bytes(latent)

    # Chunk (or pad final chunk)
    latents = []
    i = 0
    n = x.shape[0]
    while i < n:
        chunk = x[i : i + EXPECTED_FRAME_SAMPLES]
        chunk = _fit_frame(chunk, EXPECTED_FRAME_SAMPLES)
        latents.append(_encoder.forward(chunk))
        i += EXPECTED_FRAME_SAMPLES

    # Store a list of per-frame latents (AMS-safe bytes)
    # This avoids torch.cat dimension mismatch inside your model.
    safe = []
    for z in latents:
        # Keep each latent AMS-safe (numpy / CPU)
        if torch.is_tensor(z):
            safe.append(z.detach().cpu().numpy())
        else:
            safe.append(z)
    return pickle.dumps(safe, protocol=pickle.HIGHEST_PROTOCOL)


def my_decoder_logic(compressed_bytes: bytes) -> np.ndarray:
    """
    Parameters
    ----------
    compressed_bytes : bytes
        Output of my_encoder_logic

    Returns
    -------
    np.ndarray
        Decoded audio frame (float32)
    """
    # Handle empty payloads defensively (some systems allow skipping)
    if not compressed_bytes:
        return np.zeros((EXPECTED_FRAME_SAMPLES,), dtype=np.float32)

    payload = pickle.loads(compressed_bytes)

    # payload can be:
    #  - a single latent (numpy/tensor-like)
    #  - a list of per-frame latents (from chunking mode)
    with torch.no_grad():
        if isinstance(payload, list):
            # Decode each frame latent and concatenate audio
            audios = []
            for item in payload:
                latent = item
                if isinstance(latent, np.ndarray):
                    latent = torch.from_numpy(latent)
                if torch.is_tensor(latent):
                    latent = latent.to(device)

                _maybe_reset(_decoder)  # keep decoder stateless per frame
                a = _decoder.forward(latent)
                a = a.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
                audios.append(a)

            out = np.concatenate(audios, axis=0).astype(np.float32, copy=False)
            return out

        # Single latent case
        latent = payload
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)
        if torch.is_tensor(latent):
            latent = latent.to(device)

        _maybe_reset(_decoder)
        audio = _decoder.forward(latent)

    return audio.squeeze().cpu().numpy().astype(np.float32, copy=False)

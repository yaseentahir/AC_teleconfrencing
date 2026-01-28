# ams_interface.py
# =========================================================
# Audio Encoder / Decoder Interface for AMS Deployment
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
# REQUIRED BY AMS WRAPPER
# =========================================================

def my_encoder_logic(audio_frame: np.ndarray) -> bytes:
    """
    Parameters
    ----------
    audio_frame : np.ndarray
        Shape: (N,) or (N,1)
        Dtype: float32
        Range: [-1, 1]

    Returns
    -------
    bytes
        Compressed audio bytes for AMS
    """

    if audio_frame.ndim == 2:
        audio_frame = audio_frame[:, 0]

    # Encode using EnCodec
    latent = _encoder.forward(audio_frame)

    # Convert to bytes (AMS-safe)
    return pickle.dumps(latent)


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

    latent = pickle.loads(compressed_bytes)

    with torch.no_grad():
        audio = _decoder.forward(latent)

    return audio.squeeze().cpu().numpy().astype(np.float32)

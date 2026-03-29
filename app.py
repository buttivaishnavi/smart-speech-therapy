import io
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Allow the website to call this backend
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load your .pt model once at startup
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
# This checkpoint contains non-tensor objects, so disable weights-only loading.
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

# Adapt these lines to match how your model was saved.
MODEL_LOAD_ERROR = None
if isinstance(checkpoint, dict) and "model" in checkpoint and hasattr(checkpoint["model"], "eval"):
    model = checkpoint["model"]
elif hasattr(checkpoint, "eval"):
    model = checkpoint
else:
    model = None
    MODEL_LOAD_ERROR = (
        "Checkpoint appears to be a state_dict without model architecture. "
        "Re-save with a full model object under key 'model', or reconstruct "
        "the model class and load_state_dict before inference."
    )

if model is not None:
    model.eval()

# ── Analyse endpoint ──
@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": MODEL_LOAD_ERROR})

    data = await audio.read()
    samples, sr = sf.read(io.BytesIO(data), dtype="float32")

    # Convert to mono and channel-first tensor shape [1, num_samples].
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    waveform = torch.from_numpy(samples).unsqueeze(0)

    # Resample to 16 kHz if needed
    if sr != 16000:
        resampled = resample_poly(waveform.numpy(), 16000, sr, axis=1)
        waveform = torch.from_numpy(resampled.astype(np.float32))

    # Run your model — adjust to match your model's API
    with torch.no_grad():
        output = model(waveform)

    # Return results — adapt field names to your model output
    return JSONResponse({
        "transcript": output.get("transcript", ""),
        "phoneme_errors": output.get("errors", []),
        "accuracy": output.get("accuracy", 0),
        "per": output.get("per", 0)
    })
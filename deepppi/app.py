"""REST inference service for DeepPPI using the existing salviachat PyTorch environment."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MAX_LEN = 512
SEEDS = (42, 43, 44)
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: index + 1 for index, aa in enumerate(AA)}


class DeepPPI(nn.Module):
    """PyTorch equivalent of the packaged Keras Siamese CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(21, 64)
        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequence).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        return x.mean(dim=2)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> torch.Tensor:
        feature_a = self.encode(seq_a)
        feature_b = self.encode(seq_b)
        merged = torch.cat((torch.abs(feature_a - feature_b), feature_a * feature_b), dim=1)
        x = F.relu(self.fc1(merged))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output(x)).squeeze(1)


def load_keras_weights(model: DeepPPI, path: Path) -> None:
    """Load Keras 2.15 HDF5 arrays into equivalent PyTorch layers."""

    with h5py.File(path, "r") as weights:
        def array(name: str) -> torch.Tensor:
            return torch.from_numpy(weights[name][()])

        with torch.no_grad():
            model.embedding.weight.copy_(
                array("layers/functional/layers/embedding/vars/0")
            )
            model.conv1.weight.copy_(
                array("layers/functional/layers/conv1d/vars/0").permute(2, 1, 0)
            )
            model.conv1.bias.copy_(
                array("layers/functional/layers/conv1d/vars/1")
            )
            model.conv2.weight.copy_(
                array("layers/functional/layers/conv1d_1/vars/0").permute(2, 1, 0)
            )
            model.conv2.bias.copy_(
                array("layers/functional/layers/conv1d_1/vars/1")
            )
            model.fc1.weight.copy_(array("layers/dense/vars/0").transpose(0, 1))
            model.fc1.bias.copy_(array("layers/dense/vars/1"))
            model.fc2.weight.copy_(array("layers/dense_1/vars/0").transpose(0, 1))
            model.fc2.bias.copy_(array("layers/dense_1/vars/1"))
            model.output.weight.copy_(array("layers/dense_2/vars/0").transpose(0, 1))
            model.output.bias.copy_(array("layers/dense_2/vars/1"))


class PPIRequest(BaseModel):
    seq_a: str = Field(..., min_length=1, description="First amino-acid sequence")
    seq_b: str = Field(..., min_length=1, description="Second amino-acid sequence")
    model_type: str = Field("ensemble", description="ensemble, seed_42, seed_43, or seed_44")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class Runtime:
    def __init__(self) -> None:
        requested = os.getenv("DEEPPPI_DEVICE", "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DEEPPPI_DEVICE requests CUDA, but CUDA is unavailable")
        self.device = torch.device(requested)
        self.models: Dict[int, DeepPPI] = {}
        self.load_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.error: str | None = None

    def load(self) -> None:
        if self.models:
            return
        with self.load_lock:
            if self.models:
                return
            try:
                loaded: Dict[int, DeepPPI] = {}
                for seed in SEEDS:
                    model = DeepPPI()
                    weight_path = ROOT / "weights" / "D" / f"seed_{seed}" / "best.weights.h5"
                    load_keras_weights(model, weight_path)
                    model.to(self.device)
                    model.eval()
                    loaded[seed] = model
                self.models = loaded
                self.error = None
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise

    def predict(self, seq_a: str, seq_b: str, model_type: str) -> Dict[int, float]:
        self.load()
        if model_type in {"ensemble", "custom"}:
            seeds = SEEDS
        else:
            try:
                seed = int(model_type.removeprefix("seed_"))
            except ValueError as exc:
                raise ValueError("model_type must be ensemble or seed_42/seed_43/seed_44") from exc
            if seed not in SEEDS:
                raise ValueError("model_type must be ensemble or seed_42/seed_43/seed_44")
            seeds = (seed,)

        input_a = encode_sequence(seq_a).to(self.device)
        input_b = encode_sequence(seq_b).to(self.device)
        with self.inference_lock, torch.inference_mode():
            return {
                seed: float(self.models[seed](input_a, input_b).item())
                for seed in seeds
            }


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def encode_sequence(sequence: str) -> torch.Tensor:
    clean = normalize_sequence(sequence)
    if not clean:
        raise ValueError("Protein sequences cannot be empty")
    encoded = torch.zeros((1, MAX_LEN), dtype=torch.long)
    values = [AA_TO_ID.get(aa, 0) for aa in clean[:MAX_LEN]]
    encoded[0, : len(values)] = torch.tensor(values, dtype=torch.long)
    return encoded


runtime = Runtime()
app = FastAPI(title="DeepPPI REST API", version="1.1.0")


@app.get("/health")
def health() -> dict:
    try:
        runtime.load()
    except Exception:
        return {"status": "unhealthy", "loaded": False, "error": runtime.error}
    return {
        "status": "ok",
        "loaded": True,
        "backend": "pytorch",
        "models": [f"seed_{seed}" for seed in runtime.models],
        "device": str(runtime.device),
    }


@app.post("/predict_ppi")
def predict_ppi(request: PPIRequest) -> dict:
    try:
        probabilities = runtime.predict(request.seq_a, request.seq_b, request.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DeepPPI model unavailable: {exc}") from exc

    probability = sum(probabilities.values()) / len(probabilities)
    return {
        "model_used": "DeepPPI-D-ensemble" if len(probabilities) > 1 else f"DeepPPI-D-seed_{next(iter(probabilities))}",
        "probability": probability,
        "interaction": probability >= request.threshold,
        "threshold": request.threshold,
        "probabilities_by_seed": {f"seed_{seed}": value for seed, value in probabilities.items()},
        "sequence_lengths": {
            "a": len(normalize_sequence(request.seq_a)),
            "b": len(normalize_sequence(request.seq_b)),
        },
        "truncated_to": MAX_LEN,
        "importance_track_a": [],
        "importance_track_b": [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, workers=1)

"""Vision / page-image embedders (SigLIP) with color-histogram fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from pdf_vlm.utils.logging import get_logger

logger = get_logger("index.vision_embedder")


class VisionEmbedder:
    def __init__(
        self,
        name: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
        normalize: bool = True,
        dim: int = 768,
    ):
        self.name = name
        self.device = device
        self.normalize = normalize
        self.dim = dim
        self._model = None
        self._processor = None
        self._backend = "hist"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.name)
            self._model = AutoModel.from_pretrained(self.name).to(self.device)
            self._model.eval()
            self._torch = torch
            self._backend = "siglip"
            logger.info("Loaded vision embedder %s on %s", self.name, self.device)
        except Exception as exc:
            logger.warning("SigLIP unavailable (%s); using histogram embedder", exc)
            self._backend = "hist"

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        self._load()
        if self._backend == "siglip" and self._model is not None:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                vecs = feats.detach().cpu().numpy().astype(np.float32)
            if self.normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                vecs = vecs / norms
            return vecs

        out = np.zeros((len(image_paths), self.dim), dtype=np.float32)
        for i, path in enumerate(image_paths):
            img = Image.open(path).convert("RGB").resize((64, 64))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            hist = []
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c], bins=min(64, self.dim // 3), range=(0, 1))
                hist.append(h.astype(np.float32))
            vec = np.concatenate(hist)
            if vec.shape[0] < self.dim:
                pad = np.zeros(self.dim - vec.shape[0], dtype=np.float32)
                vec = np.concatenate([vec, pad])
            else:
                vec = vec[: self.dim]
            if self.normalize:
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
            out[i] = vec
        return out

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Text side for SigLIP dual-encoder retrieval; hash fallback otherwise."""
        self._load()
        if self._backend == "siglip" and self._model is not None:
            inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                vecs = feats.detach().cpu().numpy().astype(np.float32)
            if self.normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                vecs = vecs / norms
            return vecs

        # Weak fallback: reuse text hashing into same dim
        from pdf_vlm.index.text_embedder import TextEmbedder

        return TextEmbedder(dim=self.dim, normalize=self.normalize).embed(texts)

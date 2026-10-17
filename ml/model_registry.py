# ml/model_registry.py
"""
Model Registry — tracks trained ML models, their versions, metrics,
and provides loading/saving with automatic versioning.

Supports: scikit-learn models (joblib), metadata in JSON sidecar files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = "data/models"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: int
    model_type: str                # e.g. "isolation_forest", "random_forest", "gradient_boosting"
    created_at: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    training_samples: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: str = "active"         # active, retired, experimental

    @property
    def model_id(self) -> str:
        return f"{self.name}_v{self.version}"


class ModelRegistry:
    """
    Manages ML model persistence, versioning, and lookup.

    Usage:
        registry = ModelRegistry()
        registry.register("anomaly_detector", model, metrics={...})
        model = registry.load("anomaly_detector")       # latest version
        model = registry.load("anomaly_detector", version=2)
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}

    def register(
        self,
        name: str,
        model: Any,
        *,
        model_type: str = "unknown",
        metrics: Dict[str, float] | None = None,
        feature_names: List[str] | None = None,
        training_samples: int = 0,
        hyperparameters: Dict[str, Any] | None = None,
        description: str = "",
    ) -> ModelMetadata:
        """Save a model and register its metadata. Auto-increments version."""
        version = self._next_version(name)

        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            feature_names=feature_names or [],
            training_samples=training_samples,
            hyperparameters=hyperparameters or {},
            description=description,
        )

        # Save model file
        model_path = self._model_path(name, version)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import joblib
            joblib.dump(model, model_path)
        except ImportError:
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        # Save metadata sidecar
        meta_path = self._meta_path(name, version)
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        self._cache[metadata.model_id] = model

        logger.info(
            "Registered model %s (v%d): %s, %d samples, metrics=%s",
            name, version, model_type, training_samples, metrics,
        )
        return metadata

    def load(
        self,
        name: str,
        version: int | None = None,
    ) -> Any | None:
        """Load a model by name. Returns latest version if version is None."""
        if version is None:
            version = self._latest_version(name)
            if version is None:
                logger.warning("No registered model found: %s", name)
                return None

        model_id = f"{name}_v{version}"

        # Check cache
        if model_id in self._cache:
            return self._cache[model_id]

        model_path = self._model_path(name, version)
        if not model_path.exists():
            logger.warning("Model file not found: %s", model_path)
            return None

        try:
            import joblib
            model = joblib.load(model_path)
        except ImportError:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        self._cache[model_id] = model
        logger.info("Loaded model %s v%d", name, version)
        return model

    def get_metadata(
        self,
        name: str,
        version: int | None = None,
    ) -> ModelMetadata | None:
        """Get metadata for a model."""
        if version is None:
            version = self._latest_version(name)
            if version is None:
                return None

        meta_path = self._meta_path(name, version)
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            data = json.load(f)
        return ModelMetadata(**data)

    def list_models(self) -> List[ModelMetadata]:
        """List all registered models (latest version of each)."""
        models = []
        seen_names = set()

        for meta_file in sorted(self._model_dir.rglob("*.meta.json"), reverse=True):
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                name = data.get("name", "")
                if name not in seen_names:
                    seen_names.add(name)
                    models.append(ModelMetadata(**data))
            except (json.JSONDecodeError, TypeError):
                continue

        return models

    def retire(self, name: str, version: int) -> bool:
        """Mark a model version as retired."""
        meta = self.get_metadata(name, version)
        if meta is None:
            return False

        meta.status = "retired"
        meta_path = self._meta_path(name, version)
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)

        # Remove from cache
        self._cache.pop(f"{name}_v{version}", None)
        logger.info("Retired model %s v%d", name, version)
        return True

    # ── internals ──

    def _model_path(self, name: str, version: int) -> Path:
        return self._model_dir / name / f"v{version}.joblib"

    def _meta_path(self, name: str, version: int) -> Path:
        return self._model_dir / name / f"v{version}.meta.json"

    def _next_version(self, name: str) -> int:
        latest = self._latest_version(name)
        return (latest or 0) + 1

    def _latest_version(self, name: str) -> int | None:
        model_dir = self._model_dir / name
        if not model_dir.exists():
            return None

        versions = []
        for p in model_dir.glob("v*.meta.json"):
            try:
                v = int(p.stem.replace("v", "").replace(".meta", ""))
                versions.append(v)
            except ValueError:
                continue

        return max(versions) if versions else None

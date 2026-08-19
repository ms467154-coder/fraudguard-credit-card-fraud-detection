import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import app, required_features  # noqa: E402


def test_required_features_are_deduplicated_and_ordered():
    config = {"a": {"features": ["V1", "V2"]}, "b": {"features": ["V2", "V3"]}}
    assert required_features(config) == ["V1", "V2", "V3"]


def test_app_metadata():
    assert app.title == "FraudGuard API"
    assert app.version == "1.0.0"

@pytest.mark.skipif(os.getenv("RUN_LIVE_MODEL_TESTS") != "1", reason="Set RUN_LIVE_MODEL_TESTS=1 when model dependencies and artifacts are available")
def test_live_health():
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["service"] == "fraudguard-api"

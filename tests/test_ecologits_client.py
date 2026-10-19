import json

import httpx

from ecologits_client import EcoLogitsClient
from models import EstimationRequest


def _mock_transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1beta/providers":
        return httpx.Response(200, json={"providers": ["openai"]})
    if request.url.path == "/v1beta/models/openai":
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "provider": "openai",
                        "name": "gpt-4o-mini",
                        "warnings": [],
                    }
                ]
            },
        )
    if request.url.path == "/v1beta/estimations":
        request_payload = json.loads(request.content.decode("utf-8"))
        assert request_payload["provider"] == "openai"
        assert request_payload["model_name"] == "gpt-4o-mini"
        assert request_payload["output_token_count"] == 42
        return httpx.Response(
            200,
            json={
                "impacts": {
                    "energy": {
                        "type": "energy",
                        "name": "Energy",
                        "unit": "kWh",
                        "value": {"min": 0.1, "max": 0.2},
                    },
                    "gwp": {
                        "type": "GWP",
                        "name": "Global Warming Potential",
                        "unit": "kgCO2eq",
                        "value": {"min": 0.01, "max": 0.02},
                    },
                    "warnings": [
                        {"code": "model-arch-not-released", "message": "Lower precision expected."}
                    ],
                    "errors": None
                }
            },
        )
    raise AssertionError(f"Unexpected request: {request.method} {request.url}")


def test_ecologits_client_fetches_catalog_and_estimation() -> None:
    client = EcoLogitsClient(transport=httpx.MockTransport(_mock_transport))

    providers = client.get_providers()
    models = client.get_models("openai")
    estimation = client.estimate(
        EstimationRequest(provider="openai", model_name="gpt-4o-mini", output_token_count=42)
    )

    assert providers == ["openai"]
    assert models[0].name == "gpt-4o-mini"
    assert estimation.metrics["energy"].value.max == 0.2
    assert estimation.warnings[0].code == "model-arch-not-released"

    client.close()


def test_ecologits_client_ignores_non_metric_impact_entries() -> None:
    client = EcoLogitsClient(transport=httpx.MockTransport(_mock_transport))

    estimation = client.estimate(
        EstimationRequest(provider="openai", model_name="gpt-4o-mini", output_token_count=42)
    )

    assert "warnings" not in estimation.metrics
    assert "errors" not in estimation.metrics
    client.close()
from __future__ import annotations

import httpx

from models import CatalogModelEntry, EcoLogitsEstimationResult, EstimationRequest, ImpactMetric, ImpactWarning


class EcoLogitsClient:
    def __init__(
        self,
        base_url: str = "https://api.ecologits.ai",
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json"},
        )
        self._providers_cache: list[str] | None = None
        self._models_cache: dict[str, list[CatalogModelEntry]] = {}

    def close(self) -> None:
        self._client.close()

    def get_providers(self) -> list[str]:
        if self._providers_cache is None:
            response = self._client.get("/v1beta/providers")
            response.raise_for_status()
            self._providers_cache = list(response.json().get("providers", []))
        return list(self._providers_cache)

    def get_models(self, provider: str) -> list[CatalogModelEntry]:
        if provider not in self._models_cache:
            response = self._client.get(f"/v1beta/models/{provider}")
            response.raise_for_status()
            self._models_cache[provider] = [
                CatalogModelEntry.model_validate(model_payload)
                for model_payload in response.json().get("models", [])
            ]
        return list(self._models_cache[provider])

    def estimate(self, request: EstimationRequest) -> EcoLogitsEstimationResult:
        response = self._client.post(
            "/v1beta/estimations",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        payload = response.json()
        impacts_payload = payload.get("impacts", {})

        metrics: dict[str, ImpactMetric] = {}
        for key, metric_payload in impacts_payload.items():
            if not isinstance(metric_payload, dict):
                continue
            if "value" in metric_payload and "unit" in metric_payload and isinstance(metric_payload["value"], dict):
                metrics[key] = ImpactMetric.model_validate(metric_payload)

        warnings = [
            ImpactWarning.model_validate(warning_payload)
            for warning_payload in impacts_payload.get("warnings") or []
        ]

        return EcoLogitsEstimationResult(
            metrics=metrics,
            warnings=warnings,
            errors=impacts_payload.get("errors"),
            raw_response=payload,
        )
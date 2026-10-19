from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


RuntimeVendor: TypeAlias = Literal["azure", "databricks", "manual", "unknown"]


class MetricInterval(BaseModel):
    min: float
    max: float


class ImpactMetric(BaseModel):
    type: str
    name: str
    unit: str
    value: MetricInterval


class ImpactWarning(BaseModel):
    code: str
    message: str


class CatalogModelEntry(BaseModel):
    provider: str
    name: str
    warnings: list[ImpactWarning] = Field(default_factory=list)


class CatalogIndex(BaseModel):
    providers: list[str]
    models_by_provider: dict[str, list[str]]

    def contains(self, provider: str, model_name: str) -> bool:
        return self.match_model(provider, [model_name]) is not None

    def match_model(self, provider: str, candidates: list[str]) -> str | None:
        provider_models = self.models_by_provider.get(provider, [])
        lowered_lookup = {model.lower(): model for model in provider_models}
        for candidate in candidates:
            if not candidate:
                continue
            matched = lowered_lookup.get(candidate.lower())
            if matched:
                return matched
        return None


class EstimationRequest(BaseModel):
    provider: str
    model_name: str
    output_token_count: int
    request_latency: float | None = None
    electricity_mix_zone: str | None = None


class EcoLogitsEstimationResult(BaseModel):
    metrics: dict[str, ImpactMetric] = Field(default_factory=dict)
    warnings: list[ImpactWarning] = Field(default_factory=list)
    errors: list[dict[str, Any]] | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class ParsedInferencePayload(BaseModel):
    runtime_vendor: RuntimeVendor = "unknown"
    runtime_endpoint: str | None = None
    runtime_deployment: str | None = None
    response_model: str | None = None
    response_provider_hint: str | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    raw_payload_excerpt: str | None = None
    parse_messages: list[str] = Field(default_factory=list)


class ResolutionResult(BaseModel):
    status: Literal["auto", "manual_override", "needs_review", "unresolved"]
    eco_provider: str | None = None
    eco_model_name: str | None = None
    messages: list[str] = Field(default_factory=list)


class ImpactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    project_id: str
    project_name: str
    project_group: str | None = None
    environment: str | None = None
    runtime_vendor: str
    runtime_endpoint: str | None = None
    runtime_deployment: str | None = None
    response_model: str | None = None
    response_provider_hint: str | None = None
    eco_provider: str
    eco_model_name: str
    output_tokens: int
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    electricity_mix_zone: str | None = None
    resolution_status: str
    resolution_notes: list[str] = Field(default_factory=list)
    estimation_warnings: list[ImpactWarning] = Field(default_factory=list)
    estimation_errors: list[dict[str, Any]] | None = None
    source_payload_excerpt: str | None = None
    impact_energy_min: float | None = None
    impact_energy_max: float | None = None
    impact_gwp_min: float | None = None
    impact_gwp_max: float | None = None
    impact_adpe_min: float | None = None
    impact_adpe_max: float | None = None
    impact_pe_min: float | None = None
    impact_pe_max: float | None = None
    impact_wcf_min: float | None = None
    impact_wcf_max: float | None = None

    @classmethod
    def from_estimation(
        cls,
        *,
        project_id: str,
        project_name: str,
        project_group: str | None,
        environment: str | None,
        parsed_payload: ParsedInferencePayload,
        eco_provider: str,
        eco_model_name: str,
        resolution_status: str,
        resolution_notes: list[str],
        electricity_mix_zone: str | None,
        estimation_result: EcoLogitsEstimationResult,
    ) -> "ImpactRecord":
        record_data: dict[str, Any] = {
            "project_id": project_id,
            "project_name": project_name,
            "project_group": project_group,
            "environment": environment,
            "runtime_vendor": parsed_payload.runtime_vendor,
            "runtime_endpoint": parsed_payload.runtime_endpoint,
            "runtime_deployment": parsed_payload.runtime_deployment,
            "response_model": parsed_payload.response_model,
            "response_provider_hint": parsed_payload.response_provider_hint,
            "eco_provider": eco_provider,
            "eco_model_name": eco_model_name,
            "output_tokens": parsed_payload.output_tokens or 0,
            "prompt_tokens": parsed_payload.prompt_tokens,
            "total_tokens": parsed_payload.total_tokens,
            "latency_seconds": parsed_payload.latency_seconds,
            "electricity_mix_zone": electricity_mix_zone,
            "resolution_status": resolution_status,
            "resolution_notes": resolution_notes,
            "estimation_warnings": estimation_result.warnings,
            "estimation_errors": estimation_result.errors,
            "source_payload_excerpt": parsed_payload.raw_payload_excerpt,
        }
        for metric_name in ["energy", "gwp", "adpe", "pe", "wcf"]:
            metric = estimation_result.metrics.get(metric_name)
            if metric:
                record_data[f"impact_{metric_name}_min"] = metric.value.min
                record_data[f"impact_{metric_name}_max"] = metric.value.max
        return cls(**record_data)
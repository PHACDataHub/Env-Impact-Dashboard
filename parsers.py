from __future__ import annotations

import json
from typing import Any

from models import ParsedInferencePayload, RuntimeVendor


def _find_first(payload: dict[str, Any], paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def parse_inference_payload(raw_text: str, runtime_vendor: RuntimeVendor) -> ParsedInferencePayload:
    messages: list[str] = []
    if not raw_text.strip():
        return ParsedInferencePayload(
            runtime_vendor=runtime_vendor if runtime_vendor in {"azure", "databricks", "manual"} else "unknown",
            parse_messages=["No payload provided."],
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ParsedInferencePayload(
            runtime_vendor=runtime_vendor if runtime_vendor in {"azure", "databricks", "manual"} else "unknown",
            raw_payload_excerpt=raw_text[:2000],
            parse_messages=[f"Invalid JSON: {exc.msg}"],
        )

    if not isinstance(payload, dict):
        return ParsedInferencePayload(
            runtime_vendor=runtime_vendor if runtime_vendor in {"azure", "databricks", "manual"} else "unknown",
            raw_payload_excerpt=raw_text[:2000],
            parse_messages=["Payload must be a JSON object."],
        )

    usage = _find_first(payload, [("usage",), ("response", "usage"), ("metadata", "usage")]) or {}
    if not isinstance(usage, dict):
        usage = {}

    response_model = _find_first(
        payload,
        [
            ("model",),
            ("response", "model"),
            ("metadata", "model"),
            ("serving_metadata", "model_name"),
        ],
    )
    runtime_endpoint = _find_first(payload, [("endpoint",), ("serving_endpoint",), ("metadata", "endpoint")])
    runtime_deployment = _find_first(
        payload,
        [("deployment",), ("deployment_id",), ("metadata", "deployment")],
    )
    provider_hint = _find_first(payload, [("provider",), ("metadata", "provider"), ("vendor",)])
    latency_seconds = _find_first(
        payload,
        [("latency",), ("latency_seconds",), ("response_ms",), ("metadata", "latency_seconds")],
    )

    output_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")

    if response_model:
        messages.append("Detected response model from payload.")
    else:
        messages.append("Model was not found in the payload; provide it manually if needed.")

    if output_tokens is not None:
        messages.append("Detected output/completion token count from usage metadata.")
    else:
        messages.append("Output tokens were not found in usage metadata; provide them manually.")

    if prompt_tokens is None:
        messages.append("Prompt tokens not found; optional for EcoLogits estimation.")
    if total_tokens is None:
        messages.append("Total tokens not found; optional for record keeping.")

    if isinstance(latency_seconds, (int, float)) and latency_seconds > 1000:
        latency_seconds = float(latency_seconds) / 1000.0
    elif latency_seconds is not None:
        try:
            latency_seconds = float(latency_seconds)
        except (TypeError, ValueError):
            messages.append("Latency was present but not numeric; ignored.")
            latency_seconds = None

    return ParsedInferencePayload(
        runtime_vendor=runtime_vendor if runtime_vendor in {"azure", "databricks", "manual"} else "unknown",
        runtime_endpoint=str(runtime_endpoint) if runtime_endpoint else None,
        runtime_deployment=str(runtime_deployment) if runtime_deployment else None,
        response_model=str(response_model) if response_model else None,
        response_provider_hint=str(provider_hint) if provider_hint else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        latency_seconds=latency_seconds,
        raw_payload_excerpt=raw_text[:2000],
        parse_messages=messages,
    )
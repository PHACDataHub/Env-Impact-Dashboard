from __future__ import annotations

import re

from models import CatalogIndex, ParsedInferencePayload, ResolutionResult


def _dedupe(items: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        normalized = item.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        result.append(normalized)
    return result


def _infer_provider_candidates(parsed_payload: ParsedInferencePayload) -> list[str]:
    model_name = (parsed_payload.response_model or "").lower()
    provider_hint = (parsed_payload.response_provider_hint or "").lower()
    candidates: list[str | None] = []

    if provider_hint:
        candidates.append(provider_hint)
    if model_name.startswith(("gpt-", "chat-latest", "o1", "o3", "o4")):
        candidates.append("openai")
    if "claude" in model_name:
        candidates.append("anthropic")
    if "gemini" in model_name:
        candidates.append("google_genai")
    if any(keyword in model_name for keyword in ["dbrx", "dolly", "llama", "phi", "/"]):
        candidates.append("huggingface_hub")
    return _dedupe(candidates)


def _model_candidates_for_provider(model_name: str, provider: str) -> list[str]:
    normalized = model_name.strip()
    lowered = normalized.lower().replace("_", "-").replace(" ", "-")
    candidates: list[str | None] = [normalized]

    openai_aliases = {
        "gpt-35-turbo": "gpt-3.5-turbo",
        "gpt-35-turbo-16k": "gpt-3.5-turbo-16k",
        "chatgpt-4o-latest": "chat-latest",
    }

    if provider == "openai":
        candidates.extend(
            [
                lowered,
                lowered.replace("gpt-5.4-", "gpt-5-"),
                lowered.replace("gpt-5.4", "gpt-5"),
            ]
        )
        candidates.append(openai_aliases.get(lowered))
        dated_match = re.match(r"^(gpt-[\w\.-]+?)-\d{4}-\d{2}-\d{2}$", lowered)
        if dated_match:
            candidates.append(dated_match.group(1))

    if provider == "huggingface_hub":
        hf_aliases = {
            "dbrx-instruct": "databricks/dbrx-instruct",
            "dbrx-base": "databricks/dbrx-base",
            "dolly-v2-12b": "databricks/dolly-v2-12b",
            "dolly-v2-7b": "databricks/dolly-v2-7b",
            "dolly-v2-3b": "databricks/dolly-v2-3b",
        }
        candidates.append(hf_aliases.get(lowered))
        if lowered.startswith("databricks-"):
            candidates.append(f"databricks/{lowered.replace('databricks-', '', 1)}")

    return _dedupe(candidates)


def resolve_ecologits_target(
    parsed_payload: ParsedInferencePayload,
    catalog_index: CatalogIndex | None,
) -> ResolutionResult:
    if catalog_index is None:
        return ResolutionResult(
            status="unresolved",
            messages=["EcoLogits catalog is unavailable; choose the provider and model manually."],
        )

    if not parsed_payload.response_model:
        return ResolutionResult(
            status="needs_review",
            messages=["No runtime model detected, so EcoLogits mapping needs manual review."],
        )

    provider_candidates = _infer_provider_candidates(parsed_payload)
    search_order = provider_candidates + [
        provider for provider in catalog_index.providers if provider not in provider_candidates
    ]

    for provider in search_order:
        model_candidates = _model_candidates_for_provider(parsed_payload.response_model, provider)
        matched_model = catalog_index.match_model(provider, model_candidates)
        if matched_model:
            return ResolutionResult(
                status="auto",
                eco_provider=provider,
                eco_model_name=matched_model,
                messages=[
                    f"Resolved provider '{provider}' and model '{matched_model}' from runtime model '{parsed_payload.response_model}'."
                ],
            )

    return ResolutionResult(
        status="needs_review",
        messages=[
            f"Could not automatically map runtime model '{parsed_payload.response_model}' to an EcoLogits catalog model."
        ],
    )
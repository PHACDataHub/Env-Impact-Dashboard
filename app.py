# --- diagnostics bootstrap: install crash/exception logging before app code ---
import os as _diag_os
import sys as _diag_sys

_diag_dir = _diag_os.path.dirname(_diag_os.path.abspath(__file__))
if _diag_dir not in _diag_sys.path:
    _diag_sys.path.insert(0, _diag_dir)
import app_diagnostics  # noqa: E402

app_diagnostics.install_diagnostics()
# --- end diagnostics bootstrap ---

import httpx
import pandas as pd
import streamlit as st

from ecologits_client import EcoLogitsClient
from models import CatalogIndex, EstimationRequest, ImpactRecord, ParsedInferencePayload, RuntimeVendor
from parsers import parse_inference_payload
from resolver import resolve_ecologits_target
from storage import RecordStore


st.set_page_config(page_title="Environment Impact Dashboard", layout="wide")


@st.cache_resource
def get_ecologits_client() -> EcoLogitsClient:
    return EcoLogitsClient()


@st.cache_data(ttl=3600)
def get_catalog_snapshot() -> dict[str, list[str]]:
    client = get_ecologits_client()
    providers = client.get_providers()
    return {
        provider: [model.name for model in client.get_models(provider)]
        for provider in providers
    }


@st.cache_resource
def get_record_store() -> RecordStore:
    return RecordStore()


def _parse_optional_int(raw_value: str) -> int | None:
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    return int(cleaned)


def _parse_optional_float(raw_value: str) -> float | None:
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    return float(cleaned)


def _build_runtime_payload(
    runtime_vendor: RuntimeVendor,
    response_model: str,
    provider_hint: str,
    output_tokens: int,
    prompt_tokens: int | None,
    total_tokens: int | None,
    latency_seconds: float | None,
    runtime_endpoint: str,
    runtime_deployment: str,
    raw_payload_text: str,
    parse_messages: list[str],
) -> ParsedInferencePayload:
    return ParsedInferencePayload(
        runtime_vendor=runtime_vendor,
        runtime_endpoint=runtime_endpoint.strip() or None,
        runtime_deployment=runtime_deployment.strip() or None,
        response_model=response_model.strip() or None,
        response_provider_hint=provider_hint.strip() or None,
        output_tokens=output_tokens if output_tokens > 0 else None,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        raw_payload_excerpt=raw_payload_text[:2000].strip() or None,
        parse_messages=parse_messages,
    )


def _format_range(record: ImpactRecord, prefix: str, unit: str) -> str:
    min_value = getattr(record, f"impact_{prefix}_min")
    max_value = getattr(record, f"impact_{prefix}_max")
    if min_value is None or max_value is None:
        return "Not available"
    return f"{min_value:.3e} – {max_value:.3e} {unit}"


def _summarize_history(records_df: pd.DataFrame) -> None:
    st.subheader("Saved history")
    if records_df.empty:
        st.info("No impact records saved yet.")
        return

    project_options = sorted(records_df["project_name"].dropna().unique().tolist())
    selected_projects = st.multiselect(
        "Filter by project",
        options=project_options,
        default=project_options,
    )
    filtered_df = records_df
    if selected_projects:
        filtered_df = filtered_df[filtered_df["project_name"].isin(selected_projects)]

    st.dataframe(
        filtered_df[
            [
                "captured_at",
                "project_name",
                "runtime_vendor",
                "response_model",
                "eco_provider",
                "eco_model_name",
                "output_tokens",
                "impact_gwp_min",
                "impact_gwp_max",
                "impact_energy_min",
                "impact_energy_max",
            ]
        ].sort_values("captured_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    project_summary = (
        filtered_df.groupby("project_name", dropna=False)
        .agg(
            requests=("record_id", "count"),
            output_tokens=("output_tokens", "sum"),
            gwp_max=("impact_gwp_max", "sum"),
            energy_max=("impact_energy_max", "sum"),
        )
        .reset_index()
        .sort_values("gwp_max", ascending=False)
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.caption("Total GWP max by project")
        st.bar_chart(project_summary.set_index("project_name")[["gwp_max"]], height=300)
    with right_col:
        st.caption("Total energy max by project")
        st.bar_chart(project_summary.set_index("project_name")[["energy_max"]], height=300)


st.title("Environment Impact Dashboard")
st.caption(
    "Normalize Azure and Databricks inference metadata, resolve the underlying provider/model, "
    "and estimate environmental impact with EcoLogits."
)


with st.sidebar:
    st.header("Project context")
    project_id = st.text_input("Project ID", value="demo-project")
    project_name = st.text_input("Project name", value="Demo Project")
    project_group = st.text_input("Project group", value="Sandbox")
    environment = st.selectbox("Environment", options=["dev", "test", "prod", "other"], index=0)
    electricity_mix_zone = st.text_input("Electricity mix zone", value="WOR")


catalog_error: str | None = None
catalog_snapshot: dict[str, list[str]] = {}
catalog_index: CatalogIndex | None = None

try:
    catalog_snapshot = get_catalog_snapshot()
    catalog_index = CatalogIndex(
        providers=list(catalog_snapshot.keys()),
        models_by_provider=catalog_snapshot,
    )
except Exception as exc:  # pragma: no cover
    catalog_error = str(exc)

if catalog_error:
    st.warning(f"EcoLogits catalog unavailable: {catalog_error}")


source_labels: dict[str, RuntimeVendor] = {
    "Azure response JSON": "azure",
    "Databricks response JSON": "databricks",
    "Manual entry": "manual",
}
source_choice = st.radio("Capture mode", options=list(source_labels.keys()), horizontal=True)
runtime_vendor: RuntimeVendor = source_labels[source_choice]

raw_payload_text = ""
parsed_payload = ParsedInferencePayload(runtime_vendor=runtime_vendor, parse_messages=[])

if runtime_vendor != "manual":
    upload = st.file_uploader("Upload response JSON", type=["json"], accept_multiple_files=False)
    if upload is not None:
        raw_payload_text = upload.read().decode("utf-8")
    raw_payload_text = st.text_area(
        "Paste provider response JSON",
        value=raw_payload_text,
        height=220,
        help="Expected to be an OpenAI-compatible response with model and usage metadata.",
    )
    if raw_payload_text.strip():
        parsed_payload = parse_inference_payload(raw_payload_text, runtime_vendor)
    else:
        parsed_payload = ParsedInferencePayload(
            runtime_vendor=runtime_vendor,
            parse_messages=["Paste or upload a JSON response to auto-extract model and token usage."],
        )
else:
    parsed_payload = ParsedInferencePayload(
        runtime_vendor="manual",
        parse_messages=["Manual entry mode selected; provide the model and token counts below."],
    )


with st.expander("Auto-detected metadata", expanded=True):
    detection_cols = st.columns(4)
    detection_cols[0].metric("Model", parsed_payload.response_model or "Unknown")
    detection_cols[1].metric("Output tokens", parsed_payload.output_tokens or 0)
    detection_cols[2].metric("Prompt tokens", parsed_payload.prompt_tokens or 0)
    detection_cols[3].metric("Latency (s)", parsed_payload.latency_seconds or 0.0)
    for message in parsed_payload.parse_messages:
        st.caption(f"• {message}")


st.subheader("Normalized inference record")
input_col_1, input_col_2 = st.columns(2)
with input_col_1:
    runtime_endpoint = st.text_input("Runtime endpoint (optional)", value=parsed_payload.runtime_endpoint or "")
    response_model = st.text_input("Runtime/response model", value=parsed_payload.response_model or "")
    output_tokens = st.number_input(
        "Output tokens",
        min_value=0,
        step=1,
        value=int(parsed_payload.output_tokens or 0),
    )
    prompt_tokens_text = st.text_input(
        "Prompt tokens (optional)",
        value="" if parsed_payload.prompt_tokens is None else str(parsed_payload.prompt_tokens),
    )

with input_col_2:
    runtime_deployment = st.text_input("Runtime deployment (optional)", value=parsed_payload.runtime_deployment or "")
    provider_hint = st.text_input("Provider hint (optional)", value=parsed_payload.response_provider_hint or "")
    latency_seconds_text = st.text_input(
        "Latency in seconds (optional)",
        value="" if parsed_payload.latency_seconds is None else str(parsed_payload.latency_seconds),
    )
    total_tokens_text = st.text_input(
        "Total tokens (optional)",
        value="" if parsed_payload.total_tokens is None else str(parsed_payload.total_tokens),
    )


optional_errors: list[str] = []
prompt_tokens: int | None = None
total_tokens: int | None = None
latency_seconds: float | None = None

for label, raw_value, parser in [
    ("Prompt tokens", prompt_tokens_text, _parse_optional_int),
    ("Total tokens", total_tokens_text, _parse_optional_int),
    ("Latency seconds", latency_seconds_text, _parse_optional_float),
]:
    try:
        parsed_value = parser(raw_value)
        if label == "Prompt tokens":
            prompt_tokens = parsed_value
        elif label == "Total tokens":
            total_tokens = parsed_value
        else:
            latency_seconds = parsed_value
    except ValueError:
        optional_errors.append(f"{label} must be numeric when provided.")


normalized_payload = _build_runtime_payload(
    runtime_vendor=runtime_vendor,
    response_model=response_model,
    provider_hint=provider_hint,
    output_tokens=output_tokens,
    prompt_tokens=prompt_tokens,
    total_tokens=total_tokens,
    latency_seconds=latency_seconds,
    runtime_endpoint=runtime_endpoint,
    runtime_deployment=runtime_deployment,
    raw_payload_text=raw_payload_text,
    parse_messages=parsed_payload.parse_messages,
)

resolution = resolve_ecologits_target(normalized_payload, catalog_index)

st.subheader("EcoLogits mapping")
if resolution.messages:
    for message in resolution.messages:
        st.caption(f"• {message}")

provider_options = catalog_index.providers if catalog_index else []
resolved_provider = resolution.eco_provider if resolution.eco_provider in provider_options else None
selected_provider = None
selected_model = None

if provider_options:
    provider_index = provider_options.index(resolved_provider) if resolved_provider else 0
    selected_provider = st.selectbox("EcoLogits provider", provider_options, index=provider_index)
    model_options = catalog_snapshot.get(selected_provider, [])
    default_model = resolution.eco_model_name if resolution.eco_provider == selected_provider else None
    if default_model in model_options:
        model_index = model_options.index(default_model)
    else:
        model_index = 0 if model_options else None
    if model_options:
        selected_model = st.selectbox("EcoLogits model", model_options, index=model_index)
else:
    selected_provider = st.text_input("EcoLogits provider")
    selected_model = st.text_input("EcoLogits model")


estimate_clicked = st.button("Estimate impact", type="primary", use_container_width=True)

if estimate_clicked:
    validation_errors = list(optional_errors)
    if not project_id.strip():
        validation_errors.append("Project ID is required.")
    if not project_name.strip():
        validation_errors.append("Project name is required.")
    if not response_model.strip():
        validation_errors.append("Runtime/response model is required.")
    if output_tokens <= 0:
        validation_errors.append("Output tokens must be greater than 0.")
    if not selected_provider:
        validation_errors.append("EcoLogits provider is required.")
    if not selected_model:
        validation_errors.append("EcoLogits model is required.")

    if validation_errors:
        for validation_error in validation_errors:
            st.error(validation_error)
    else:
        estimation_request = EstimationRequest(
            provider=str(selected_provider),
            model_name=str(selected_model),
            output_token_count=output_tokens,
            request_latency=latency_seconds,
            electricity_mix_zone=electricity_mix_zone.strip() or None,
        )
        client = get_ecologits_client()
        store = get_record_store()
        try:
            with st.spinner("Requesting EcoLogits estimation..."):
                estimation_result = client.estimate(estimation_request)
            impact_record = ImpactRecord.from_estimation(
                project_id=project_id.strip(),
                project_name=project_name.strip(),
                project_group=project_group.strip() or None,
                environment=environment,
                parsed_payload=normalized_payload,
                eco_provider=str(selected_provider),
                eco_model_name=str(selected_model),
                resolution_status=resolution.status if resolution.eco_model_name == selected_model else "manual_override",
                resolution_notes=resolution.messages,
                electricity_mix_zone=electricity_mix_zone.strip() or None,
                estimation_result=estimation_result,
            )
            store.append(impact_record)
            st.session_state["latest_record"] = impact_record.model_dump(mode="json")
            st.success("Impact record estimated and saved.")
        except httpx.HTTPStatusError as exc:
            st.error(f"EcoLogits request failed: {exc.response.text}")
        except httpx.HTTPError as exc:
            st.error(f"Unable to reach EcoLogits: {exc}")


latest_record_payload = st.session_state.get("latest_record")
if latest_record_payload:
    latest_record = ImpactRecord.model_validate(latest_record_payload)
    st.subheader("Latest estimation")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Energy", _format_range(latest_record, "energy", "kWh"))
    metric_cols[1].metric("GWP", _format_range(latest_record, "gwp", "kgCO2eq"))
    metric_cols[2].metric("Water", _format_range(latest_record, "wcf", "L"))
    more_metric_cols = st.columns(2)
    more_metric_cols[0].metric("ADPe", _format_range(latest_record, "adpe", "kgSbeq"))
    more_metric_cols[1].metric("Primary energy", _format_range(latest_record, "pe", "MJ"))
    if latest_record.estimation_warnings:
        st.warning(
            "Warnings: " + "; ".join(warning.message for warning in latest_record.estimation_warnings)
        )


history_store = get_record_store()
history_df = history_store.to_dataframe()
_summarize_history(history_df)
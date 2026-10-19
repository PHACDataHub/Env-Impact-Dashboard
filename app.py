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
from models import CatalogIndex, EstimationRequest, ImpactRecord, ParsedInferencePayload
from storage import RecordStore


EXCLUDED_PROVIDERS = {"cohere", "mistralai", "mistral"}


st.set_page_config(page_title="Environment Impact Dashboard", layout="wide")


@st.cache_resource
def get_ecologits_client() -> EcoLogitsClient:
    return EcoLogitsClient()


@st.cache_data(ttl=3600)
def get_catalog_snapshot() -> dict[str, list[str]]:
    client = get_ecologits_client()
    providers = [provider for provider in client.get_providers() if provider not in EXCLUDED_PROVIDERS]
    return {
        provider: [model.name for model in client.get_models(provider)]
        for provider in providers
    }


@st.cache_resource
def get_record_store() -> RecordStore:
    return RecordStore()


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "Not available"
    absolute_value = abs(float(value))
    decimals = 6 if 0 < absolute_value < 1 else 3
    formatted = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _format_range(record: ImpactRecord, prefix: str, unit: str) -> str:
    min_value = getattr(record, f"impact_{prefix}_min")
    max_value = getattr(record, f"impact_{prefix}_max")
    if min_value is None or max_value is None:
        return "Not available"
    return f"{_format_number(min_value)} – {_format_number(max_value)} {unit}"


def _display_normalized_payload(parsed_payload: ParsedInferencePayload, heading: str) -> None:
    st.caption(heading)
    payload_cols = st.columns(4)
    payload_cols[0].metric("Source", parsed_payload.runtime_vendor)
    payload_cols[1].metric("Model", parsed_payload.response_model or "Not available")
    payload_cols[2].metric("Output tokens", _format_number(parsed_payload.output_tokens))
    payload_cols[3].metric("Endpoint", parsed_payload.runtime_endpoint or "Not available")
    if parsed_payload.runtime_deployment:
        st.caption(f"Deployment: {parsed_payload.runtime_deployment}")
    for message in parsed_payload.parse_messages:
        st.caption(f"• {message}")


def _summarize_history(records_df: pd.DataFrame) -> None:
    st.subheader("Saved history")
    if records_df.empty:
        st.info("No impact records saved yet.")
        return

    history_columns = [
        "captured_at",
        "runtime_vendor",
        "response_model",
        "eco_provider",
        "eco_model_name",
        "output_tokens",
        "impact_gwp_min",
        "impact_gwp_max",
        "impact_energy_min",
        "impact_energy_max",
        "impact_adpe_min",
        "impact_adpe_max",
        "impact_pe_min",
        "impact_pe_max",
        "impact_wcf_min",
        "impact_wcf_max",
    ]
    history_df = records_df[history_columns].sort_values("captured_at", ascending=False).copy()
    rename_map = {
        "captured_at": "Captured at",
        "runtime_vendor": "Source",
        "response_model": "Runtime model",
        "eco_provider": "Provider",
        "eco_model_name": "EcoLogits model",
        "output_tokens": "Output tokens",
        "impact_gwp_min": "GWP min (kgCO2eq)",
        "impact_gwp_max": "GWP max (kgCO2eq)",
        "impact_energy_min": "Energy min (kWh)",
        "impact_energy_max": "Energy max (kWh)",
        "impact_adpe_min": "ADPe min (kgSbeq)",
        "impact_adpe_max": "ADPe max (kgSbeq)",
        "impact_pe_min": "Primary energy min (MJ)",
        "impact_pe_max": "Primary energy max (MJ)",
        "impact_wcf_min": "Water min (L)",
        "impact_wcf_max": "Water max (L)",
    }
    for column in [
        "impact_gwp_min",
        "impact_gwp_max",
        "impact_energy_min",
        "impact_energy_max",
        "impact_adpe_min",
        "impact_adpe_max",
        "impact_pe_min",
        "impact_pe_max",
        "impact_wcf_min",
        "impact_wcf_max",
    ]:
        history_df[column] = history_df[column].map(_format_number)

    st.dataframe(
        history_df.rename(columns=rename_map),
        width="stretch",
        hide_index=True,
    )


st.title("Environment Impact Dashboard")
st.caption(
    "Estimate environmental impact for a selected provider/model pair with EcoLogits."
)


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


estimation_mode = st.radio(
    "Estimation mode",
    options=["Endpoint estimate", "Manual EcoLogits estimate"],
    horizontal=True,
)

if estimation_mode == "Endpoint estimate":
    st.subheader("Endpoint estimate")
    endpoint_runtime = st.selectbox("Endpoint type", options=["azure", "databricks"], index=0)
    endpoint_url = st.text_input("Endpoint URL")
    endpoint_model = st.text_input("Model / deployment name")
    endpoint_output_tokens = st.number_input(
        "Output tokens",
        min_value=0,
        step=1,
        value=0,
        key="endpoint_output_tokens",
    )
    st.info(
        "Endpoint-based estimation UI is available, but live authenticated Azure/Databricks endpoint calls are not "
        "implemented in this codebase yet. Use this section to stage the endpoint inputs, or use the manual estimator below."
    )
    endpoint_payload = ParsedInferencePayload(
        runtime_vendor=endpoint_runtime,
        runtime_endpoint=endpoint_url.strip() or None,
        runtime_deployment=endpoint_model.strip() or None,
        response_model=endpoint_model.strip() or None,
        output_tokens=int(endpoint_output_tokens) if endpoint_output_tokens > 0 else None,
        parse_messages=[
            "Endpoint estimation inputs are staged here for future authenticated endpoint integration."
        ],
    )
    _display_normalized_payload(endpoint_payload, "Normalized endpoint payload")


st.subheader("Manual EcoLogits estimation")

provider_options = catalog_index.providers if catalog_index else []
selected_provider = None
selected_model = None

if provider_options:
    selected_provider = st.selectbox("EcoLogits provider", provider_options, index=0)
    model_options = catalog_snapshot.get(selected_provider, [])
    if model_options:
        selected_model = st.selectbox("EcoLogits model", model_options, index=0)
else:
    selected_provider = st.text_input("EcoLogits provider")
    selected_model = st.text_input("EcoLogits model")

output_tokens = st.number_input(
    "Output tokens",
    min_value=0,
    step=1,
    value=0,
)

manual_payload = ParsedInferencePayload(
    runtime_vendor="manual",
    response_model=str(selected_model).strip() or None,
    output_tokens=output_tokens if output_tokens > 0 else None,
    parse_messages=["Manual EcoLogits estimation uses the selected provider/model pair directly."],
)
_display_normalized_payload(manual_payload, "Normalized manual payload")


estimate_clicked = st.button("Estimate impact", type="primary", width="stretch")

if estimate_clicked:
    validation_errors: list[str] = []
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
        normalized_payload = manual_payload.model_copy(update={"output_tokens": output_tokens, "parse_messages": []})
        estimation_request = EstimationRequest(
            provider=str(selected_provider),
            model_name=str(selected_model),
            output_token_count=output_tokens,
        )
        client = get_ecologits_client()
        store = get_record_store()
        try:
            with st.spinner("Requesting EcoLogits estimation..."):
                estimation_result = client.estimate(estimation_request)
            impact_record = ImpactRecord.from_estimation(
                project_id="default-project",
                project_name="Default Project",
                project_group=None,
                environment=None,
                parsed_payload=normalized_payload,
                eco_provider=str(selected_provider),
                eco_model_name=str(selected_model),
                resolution_status="manual_override",
                resolution_notes=[],
                electricity_mix_zone=None,
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
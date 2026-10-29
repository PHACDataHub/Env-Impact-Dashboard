# --- diagnostics bootstrap: install crash/exception logging before app code ---
import os as _diag_os
import os
import sys as _diag_sys
from datetime import timedelta

from dotenv import load_dotenv

_diag_dir = _diag_os.path.dirname(_diag_os.path.abspath(__file__))
if _diag_dir not in _diag_sys.path:
    _diag_sys.path.insert(0, _diag_dir)
import app_diagnostics  # noqa: E402

app_diagnostics.install_diagnostics()
# --- end diagnostics bootstrap ---


def _load_local_environment() -> None:
    """Load local environment values without overriding process environment values."""
    env_path = _diag_os.path.join(_diag_dir, ".env")
    load_dotenv(dotenv_path=env_path, override=False)


_load_local_environment()

import httpx
import pandas as pd
import streamlit as st

from azure_metrics import AzureMetricsClient
from ecologits_client import EcoLogitsClient
from models import CatalogIndex, EstimationRequest, ImpactRecord, ParsedInferencePayload
from resolver import resolve_ecologits_target
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


@st.cache_resource
def get_azure_metrics_client() -> AzureMetricsClient:
    return AzureMetricsClient()


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
    payload_cols = st.columns(2)
    payload_cols[0].metric("Source", parsed_payload.runtime_vendor)
    payload_cols[1].metric("Output tokens", _format_number(parsed_payload.output_tokens))
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
        "runtime_endpoint",
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
    history_df = records_df[history_columns].copy()
    history_df["_captured_group"] = history_df["captured_at"].map(
        lambda value: str(value)[:19]
    )
    history_df = history_df.sort_values("captured_at", ascending=False)
    history_ids = {
        captured_group: history_id
        for history_id, captured_group in enumerate(
            history_df["_captured_group"].drop_duplicates(), start=1
        )
    }
    history_df.insert(0, "history_id", history_df["_captured_group"].map(history_ids))
    history_df = history_df.drop(columns=["_captured_group", "captured_at"])
    rename_map = {
        "history_id": "ID",
        "runtime_vendor": "Runtime",
        "runtime_endpoint": "Endpoint",
        "eco_provider": "Provider",
        "eco_model_name": "Model",
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
    endpoint_period_days = st.selectbox(
        "Azure Monitor period",
        options=[1, 7, 14, 30, 60, 90, 120],
        format_func=lambda days: f"{days} day{'s' if days != 1 else ''}",
    )
    endpoint_timespan = timedelta(days=endpoint_period_days)
    endpoint_metrics = {}
    endpoint_model_metrics = {}
    try:
        azure_metrics = get_azure_metrics_client()
        endpoint_metrics = {
            "response_model": azure_metrics.get_response_model(endpoint_timespan),
            "output_tokens": azure_metrics.get_output_tokens(endpoint_timespan),
            "request_latency": azure_metrics.get_request_latency(endpoint_timespan),
        }
        endpoint_model_metrics = azure_metrics.get_model_metrics(endpoint_timespan)
    except Exception as exc:  # pragma: no cover - exercised by deployment configuration
        st.error(f"Unable to read Azure Monitor metrics: {exc}")
    endpoint_payload = ParsedInferencePayload(
        runtime_vendor=endpoint_runtime,
        response_model=endpoint_metrics.get("response_model"),
        output_tokens=endpoint_metrics.get("output_tokens"),
        latency_seconds=endpoint_metrics.get("request_latency"),
    )
    _display_normalized_payload(endpoint_payload, "Normalized endpoint payload")
    preview_requests = []
    for model_name, model_metrics in endpoint_model_metrics.items():
        model_payload = endpoint_payload.model_copy(update={
            "response_model": model_name,
            "output_tokens": model_metrics["output_tokens"],
            "latency_seconds": model_metrics["request_latency"],
            "response_provider_hint": "openai",
        })
        resolution = resolve_ecologits_target(model_payload, catalog_index)
        preview_requests.append({
            "request": EstimationRequest(
                provider=resolution.eco_provider or "openai",
                model_name=resolution.eco_model_name or model_name,
                output_token_count=model_metrics["output_tokens"],
                request_latency=model_metrics["request_latency"],
                electricity_mix_zone=os.getenv("ELECTRICITY_MIX_ZONE"),
            ).model_dump(exclude_none=True),
            "resolution_status": resolution.status,
            "resolution_messages": resolution.messages,
        })
    if not preview_requests:
        preview_requests = [{
            "request": None,
            "resolution_status": "unresolved",
            "resolution_messages": [
                "No per-model Azure Monitor series were returned, so there is no model-specific request body to preview."
            ],
            "aggregate_metrics": {
                "response_model": endpoint_payload.response_model,
                "output_tokens": endpoint_payload.output_tokens,
                "request_latency": endpoint_payload.latency_seconds,
                "provider": "openai",
                "electricity_mix_zone": os.getenv("ELECTRICITY_MIX_ZONE"),
            },
        }]
    st.caption("EcoLogits request body diagnostic")
    st.json(preview_requests)
    estimate_clicked = st.button(
        "Estimate impact",
        type="primary",
        width="stretch",
        key="endpoint_estimate_button",
    )

if estimation_mode == "Endpoint estimate" and estimate_clicked:
    validation_errors: list[str] = []
    if not endpoint_model_metrics and endpoint_payload.output_tokens is None:
        validation_errors.append("No Azure Monitor model metrics were available.")

    if validation_errors:
        for validation_error in validation_errors:
            st.error(validation_error)
    elif not endpoint_model_metrics:
        st.warning(
            "Azure Monitor returned aggregate output tokens but no per-model series. "
            "No model-specific EcoLogits estimate was sent."
        )
    elif not endpoint_payload.output_tokens:
        st.warning(
            "Azure Monitor returned zero output tokens. The normalized payload "
            "is shown above, but no EcoLogits request was sent."
        )
    else:
        client = get_ecologits_client()
        store = get_record_store()
        try:
            with st.spinner("Requesting EcoLogits estimations..."):
                impact_records = []
                for model_name, model_metrics in endpoint_model_metrics.items():
                    model_payload = endpoint_payload.model_copy(update={
                        "response_model": model_name,
                        "output_tokens": model_metrics["output_tokens"],
                        "latency_seconds": model_metrics["request_latency"],
                    })
                    # Azure endpoint models are Azure OpenAI models.
                    model_payload = model_payload.model_copy(
                        update={"response_provider_hint": "openai"}
                    )
                    resolution = resolve_ecologits_target(model_payload, catalog_index)
                    if resolution.eco_provider is None or resolution.eco_model_name is None:
                        st.warning(
                            f"Could not map Azure model '{model_name}' to an EcoLogits model: "
                            + "; ".join(resolution.messages)
                        )
                        continue
                    model_request = EstimationRequest(
                        provider=resolution.eco_provider,
                        model_name=resolution.eco_model_name,
                        output_token_count=model_metrics["output_tokens"],
                        request_latency=model_metrics["request_latency"],
                        electricity_mix_zone=os.getenv("ELECTRICITY_MIX_ZONE"),
                    )
                    estimation_result = client.estimate(model_request)
                    impact_records.append(ImpactRecord.from_estimation(
                project_id="default-project",
                project_name="Default Project",
                project_group=None,
                environment=None,
                parsed_payload=model_payload,
                eco_provider=model_request.provider,
                eco_model_name=model_request.model_name,
                resolution_status=resolution.status,
                resolution_notes=resolution.messages,
                electricity_mix_zone=model_request.electricity_mix_zone,
                estimation_result=estimation_result,
                    ))
                for impact_record in impact_records:
                    store.append(impact_record)
            if impact_records:
                st.session_state["latest_record"] = impact_records[-1].model_dump(mode="json")
            st.success(f"Azure endpoint impact estimated for {len(impact_records)} model(s) and saved.")
        except httpx.HTTPStatusError as exc:
            st.error(f"EcoLogits request failed: {exc.response.text}")
        except httpx.HTTPError as exc:
            st.error(f"Unable to reach EcoLogits: {exc}")
if estimation_mode == "Manual EcoLogits estimate":
    st.subheader("Manual estimation")

    provider_options = catalog_index.providers if catalog_index else []
    selected_provider = None
    selected_model = None

    if provider_options:
        selected_provider = st.selectbox("Provider", provider_options, index=0)
        model_options = catalog_snapshot.get(selected_provider, [])
        if model_options:
            selected_model = st.selectbox("Model", model_options, index=0)
    else:
        selected_provider = st.text_input("Provider")
        selected_model = st.text_input("Model")

    output_tokens = st.number_input(
        "Output tokens",
        min_value=0,
        step=1,
        value=0,
    )
    request_latency = st.number_input(
        "Request latency (seconds)",
        min_value=0.0,
        step=0.1,
        value=0.0,
    )
    electricity_mix_zone = st.selectbox(
        "Electricity mix zone",
        options=["WOR"],
        index=0,
        help="EcoLogits accepts ISO 3166-1 alpha-3 zone codes. WOR is the world average.",
    )

    manual_payload = ParsedInferencePayload(
        runtime_vendor="manual",
        response_model=str(selected_model).strip() or None,
        output_tokens=output_tokens if output_tokens > 0 else None,
        latency_seconds=request_latency if request_latency > 0 else None,
        parse_messages=["Manual estimation uses the selected provider/model pair directly."],
    )
    _display_normalized_payload(manual_payload, "Normalized manual payload")
    estimate_clicked = st.button(
        "Estimate impact",
        type="primary",
        width="stretch",
        key="manual_estimate_button",
    )

if estimation_mode == "Manual EcoLogits estimate" and estimate_clicked:
    validation_errors: list[str] = []
    if output_tokens <= 0:
        validation_errors.append("Output tokens must be greater than 0.")
    if not selected_provider:
        validation_errors.append("Provider is required.")
    if not selected_model:
        validation_errors.append("Model is required.")
    if request_latency <= 0:
        validation_errors.append("Request latency must be greater than 0 seconds.")

    if validation_errors:
        for validation_error in validation_errors:
            st.error(validation_error)
    else:
        normalized_payload = manual_payload.model_copy(update={"output_tokens": output_tokens, "parse_messages": []})
        estimation_request = EstimationRequest(
            provider=str(selected_provider),
            model_name=str(selected_model),
            output_token_count=output_tokens,
            request_latency=float(request_latency),
            electricity_mix_zone=electricity_mix_zone,
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
                electricity_mix_zone=electricity_mix_zone,
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
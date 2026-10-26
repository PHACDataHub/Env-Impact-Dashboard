from __future__ import annotations

import os
from collections import defaultdict
from datetime import timedelta

import certifi
from azure.identity import DefaultAzureCredential
from azure.monitor.query import MetricAggregationType, MetricsQueryClient


class AzureMetricsClient:
    """Read Azure Monitor metrics using the application's Azure identity."""

    OUTPUT_TOKENS_METRIC = "OutputTokens"
    REQUEST_LATENCY_METRIC = "AzureOpenAITTLTInMS"

    def __init__(
        self,
        resource_id: str | None = None,
        credential=None,
    ) -> None:
        self.resource_id = resource_id or os.getenv("AZURE_RESOURCE_ID")
        if not self.resource_id:
            raise RuntimeError(
                "AZURE_RESOURCE_ID must be set to the full Azure resource ID."
            )
        self._credential = credential or DefaultAzureCredential(
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_visual_studio_code_credential=True,
            exclude_shared_token_cache_credential=True,
        )
        self._client = MetricsQueryClient(
            self._credential,
            connection_verify=os.getenv("SSL_CERT_FILE", certifi.where()),
        )

    def _get_metric_values(
        self,
        metric_name: str,
        aggregation: MetricAggregationType,
        timespan: timedelta,
    ) -> list[float]:
        result = self._client.query_resource(
            self.resource_id,
            metric_names=[metric_name],
            timespan=timespan,
            aggregations=[aggregation],
        )
        values: list[float] = []
        for metric in result.metrics:
            for series in metric.timeseries:
                for value in series.data:
                    selected_value = value.total if aggregation == MetricAggregationType.TOTAL else value.average
                    if selected_value is not None:
                        values.append(float(selected_value))
        return values

    @staticmethod
    def _model_name(series) -> str | None:
        for metadata in getattr(series, "metadata_values", []):
            if getattr(metadata, "name", None) == "ModelName":
                value = getattr(metadata, "value", None)
                if value and value != "*":
                    return str(value)
        return None

    def get_model_metrics(
        self, timespan: timedelta
    ) -> dict[str, dict[str, float | int | None]]:
        results: dict[str, dict[str, float | int | None]] = defaultdict(
            lambda: {"output_tokens": 0, "request_latency": 0.0, "latency_samples": 0}
        )
        output_result = self._client.query_resource(
            self.resource_id,
            metric_names=[self.OUTPUT_TOKENS_METRIC],
            timespan=timespan,
            aggregations=[MetricAggregationType.TOTAL],
        )
        for metric in output_result.metrics:
            for series in metric.timeseries:
                model_name = self._model_name(series)
                if not model_name:
                    continue
                results[model_name]["output_tokens"] += sum(
                    float(point.total) for point in series.data if point.total is not None
                )

        latency_result = self._client.query_resource(
            self.resource_id,
            metric_names=[self.REQUEST_LATENCY_METRIC],
            timespan=timespan,
            aggregations=[MetricAggregationType.AVERAGE],
        )
        for metric in latency_result.metrics:
            for series in metric.timeseries:
                model_name = self._model_name(series)
                if not model_name:
                    continue
                samples = [float(point.average) for point in series.data if point.average is not None]
                if samples:
                    results[model_name]["request_latency"] = sum(samples) / len(samples)
                    results[model_name]["latency_samples"] = len(samples)

        return {
            model_name: {
                "output_tokens": int(values["output_tokens"]),
                "request_latency": float(values["request_latency"]) if values["latency_samples"] else None,
            }
            for model_name, values in results.items()
            if values["output_tokens"]
        }

    def get_output_tokens(self, timespan: timedelta) -> int | None:
        values = self._get_metric_values(self.OUTPUT_TOKENS_METRIC, MetricAggregationType.TOTAL, timespan)
        return int(sum(values)) if values else None

    def get_request_latency(self, timespan: timedelta) -> float | None:
        values = self._get_metric_values(self.REQUEST_LATENCY_METRIC, MetricAggregationType.AVERAGE, timespan)
        return values[-1] if values else None

    def get_response_model(self, timespan: timedelta) -> str | None:
        result = self._client.query_resource(
            self.resource_id,
            metric_names=[self.OUTPUT_TOKENS_METRIC],
            timespan=timespan,
        )
        for metric in result.metrics:
            for series in metric.timeseries:
                for metadata in getattr(series, "metadata_values", []):
                    if getattr(metadata, "name", None) == "ModelName":
                        value = getattr(metadata, "value", None)
                        if value:
                            return str(value)
        return None

    def close(self) -> None:
        self._client.close()
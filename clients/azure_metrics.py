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
    PER_MODEL_FILTER = (
        "ModelName eq '*' and ModelDeploymentName eq '*' and ModelVersion eq '*'"
    )

    def __init__(self, resource_id: str | None = None, credential=None) -> None:
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

    MODEL_DIMENSIONS = ("ModelName", "ModelDeploymentName", "ModelVersion")

    @classmethod
    def _model_name(cls, series) -> str | None:
        metadata_by_name = cls._series_dimensions(series)
        for dimension in cls.MODEL_DIMENSIONS:
            value = metadata_by_name.get(dimension.lower())
            if value and str(value) != "*":
                return str(value)
        return None

    @staticmethod
    def _series_dimensions(series) -> dict[str, object]:
        dimensions: dict[str, object] = {}
        for collection_name in ("metadata_values", "dimension_values", "dimensions"):
            collection = getattr(series, collection_name, None)
            if isinstance(collection, dict):
                dimensions.update({str(key).lower(): value for key, value in collection.items()})
            else:
                for item in collection or []:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("key")
                        value = item.get("value")
                    else:
                        name = getattr(item, "name", None) or getattr(item, "key", None)
                        value = getattr(item, "value", None)
                    if name is not None:
                        dimensions[str(name).lower()] = value
        return dimensions

    def get_model_metrics(self, timespan: timedelta) -> dict[str, dict[str, float | int | None]]:
        result = self._client.query_resource(
            self.resource_id,
            metric_names=[self.OUTPUT_TOKENS_METRIC],
            timespan=timespan,
            aggregations=[MetricAggregationType.TOTAL],
            filter=self.PER_MODEL_FILTER,
            max_results=1000,
        )
        output_by_model: defaultdict[str, float] = defaultdict(float)
        for metric in result.metrics:
            for series in metric.timeseries:
                model_name = self._model_name(series)
                if not model_name:
                    continue
                for point in series.data:
                    if point.total is not None:
                        output_by_model[model_name] += float(point.total)
        latency_result = self._client.query_resource(
            self.resource_id,
            metric_names=[self.REQUEST_LATENCY_METRIC],
            timespan=timespan,
            aggregations=[MetricAggregationType.AVERAGE],
            filter=self.PER_MODEL_FILTER,
            max_results=1000,
        )
        latency_by_model: dict[str, list[float]] = defaultdict(list)
        for metric in latency_result.metrics:
            for series in metric.timeseries:
                model_name = self._model_name(series)
                if model_name:
                    latency_by_model[model_name].extend(
                        float(point.average) for point in series.data if point.average is not None
                    )
        return {
            model_name: {
                "output_tokens": int(tokens),
                "request_latency": (
                    sum(latency_by_model[model_name]) / len(latency_by_model[model_name])
                    if latency_by_model[model_name]
                    else None
                ),
            }
            for model_name, tokens in output_by_model.items()
            if tokens
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
            aggregations=[MetricAggregationType.TOTAL],
            filter=self.PER_MODEL_FILTER,
            max_results=1000,
        )
        for metric in result.metrics:
            for series in metric.timeseries:
                model_name = self._model_name(series)
                if model_name:
                    return model_name
        return None

    def close(self) -> None:
        self._client.close()
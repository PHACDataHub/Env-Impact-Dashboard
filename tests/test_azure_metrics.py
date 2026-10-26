from types import SimpleNamespace
from datetime import timedelta

from azure_metrics import AzureMetricsClient


def test_init_reads_azure_resource_id_from_environment(monkeypatch) -> None:
    resource_id = "/subscriptions/test/resourceGroups/test/providers/test/resource"
    monkeypatch.setenv("AZURE_RESOURCE_ID", resource_id)

    class _FakeClient:
        def __init__(self, credential, **kwargs) -> None:
            self.credential = credential

        def close(self) -> None:
            pass

    monkeypatch.setattr("azure_metrics.MetricsQueryClient", _FakeClient)
    client = AzureMetricsClient(credential=object())

    assert client.resource_id == resource_id


def test_init_allows_azure_cli_credential(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_RESOURCE_ID", "/subscriptions/test/resource")
    captured = {}

    class _FakeCredential:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, credential, **kwargs) -> None:
            pass

    monkeypatch.setattr("azure_metrics.DefaultAzureCredential", _FakeCredential)
    monkeypatch.setattr("azure_metrics.MetricsQueryClient", _FakeClient)
    AzureMetricsClient()

    assert "exclude_cli_credential" not in captured


def test_init_configures_certifi_ca_bundle(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_RESOURCE_ID", "/subscriptions/test/resource")
    captured = {}

    class _FakeClient:
        def __init__(self, credential, **kwargs) -> None:
            captured["connection_verify"] = kwargs["connection_verify"]

    monkeypatch.setattr("azure_metrics.MetricsQueryClient", _FakeClient)
    AzureMetricsClient(credential=object())

    assert captured["connection_verify"]


def test_init_requires_azure_resource_id(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_RESOURCE_ID", raising=False)

    try:
        AzureMetricsClient(credential=object())
    except RuntimeError as exc:
        assert str(exc) == "AZURE_RESOURCE_ID must be set to the full Azure resource ID."
    else:
        raise AssertionError("Expected missing AZURE_RESOURCE_ID to raise RuntimeError")


class _FakeMetricsClient:
    def __init__(self, result) -> None:
        self.result = result
        self.arguments = None

    def query_resource(self, resource_id, **kwargs):
        self.arguments = (resource_id, kwargs)
        return self.result

    def close(self) -> None:
        pass


def test_get_output_tokens_reads_single_metrics_query_result(monkeypatch) -> None:
    result = SimpleNamespace(
        metrics=[
            SimpleNamespace(
                timeseries=[
                    SimpleNamespace(
                        data=[
                            SimpleNamespace(total=125, average=125),
                            SimpleNamespace(total=150, average=150),
                        ]
                    )
                ]
            )
        ]
    )
    client = object.__new__(AzureMetricsClient)
    client.resource_id = "/subscriptions/test/resourceGroups/test/providers/test/resource"
    fake_client = _FakeMetricsClient(result)
    client._client = fake_client

    assert client.get_output_tokens(timedelta(hours=1)) == 275
    assert fake_client.arguments[0].startswith("/subscriptions/test/")


def test_get_response_model_reads_model_name_dimension() -> None:
    result = SimpleNamespace(
        metrics=[
            SimpleNamespace(
                timeseries=[
                    SimpleNamespace(
                        metadata_values=[SimpleNamespace(name="ModelName", value="gpt-4o")],
                        data=[],
                    )
                ]
            )
        ]
    )
    client = object.__new__(AzureMetricsClient)
    client.resource_id = "/subscriptions/test/resourceGroups/test/providers/test/resource"
    client._client = _FakeMetricsClient(result)

    assert client.get_response_model(timedelta(hours=1)) == "gpt-4o"


def test_get_request_latency_uses_azure_openai_time_to_last_byte_metric(monkeypatch) -> None:
    result = SimpleNamespace(
        metrics=[
            SimpleNamespace(
                timeseries=[
                    SimpleNamespace(data=[SimpleNamespace(average=1.5)])
                ]
            )
        ]
    )
    client = object.__new__(AzureMetricsClient)
    client.resource_id = "/subscriptions/test/resourceGroups/test/providers/test/resource"
    fake_client = _FakeMetricsClient(result)
    client._client = fake_client

    assert client.get_request_latency(timedelta(hours=1)) == 1.5
    assert fake_client.arguments[1]["metric_names"] == ["AzureOpenAITTLTInMS"]


def test_get_model_metrics_groups_output_tokens_and_latency_by_model() -> None:
    output_result = SimpleNamespace(metrics=[SimpleNamespace(timeseries=[
        SimpleNamespace(
            metadata_values=[SimpleNamespace(name="ModelName", value="gpt-4o")],
            data=[SimpleNamespace(total=100)],
        ),
        SimpleNamespace(
            metadata_values=[SimpleNamespace(name="ModelName", value="gpt-4o-mini")],
            data=[SimpleNamespace(total=50)],
        ),
    ])])
    latency_result = SimpleNamespace(metrics=[SimpleNamespace(timeseries=[
        SimpleNamespace(
            metadata_values=[SimpleNamespace(name="ModelName", value="gpt-4o")],
            data=[SimpleNamespace(average=1.0), SimpleNamespace(average=2.0)],
        ),
        SimpleNamespace(
            metadata_values=[SimpleNamespace(name="ModelName", value="gpt-4o-mini")],
            data=[SimpleNamespace(average=0.5)],
        ),
    ])])

    class _MultiResultClient(_FakeMetricsClient):
        def __init__(self) -> None:
            self.arguments = []

        def query_resource(self, resource_id, **kwargs):
            self.arguments.append((resource_id, kwargs))
            return output_result if kwargs["metric_names"] == ["OutputTokens"] else latency_result

    client = object.__new__(AzureMetricsClient)
    client.resource_id = "/subscriptions/test/resourceGroups/test/providers/test/resource"
    client._client = _MultiResultClient()

    assert client.get_model_metrics(timedelta(days=14)) == {
        "gpt-4o": {"output_tokens": 100, "request_latency": 1.5},
        "gpt-4o-mini": {"output_tokens": 50, "request_latency": 0.5},
    }
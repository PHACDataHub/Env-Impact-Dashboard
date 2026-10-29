import pandas as pd

import app


def test_saved_history_uses_unique_display_columns(monkeypatch) -> None:
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(app.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "dataframe", lambda frame, **_kwargs: displayed.append(frame))

    app._summarize_history(
        pd.DataFrame(
            [
                {
                    "captured_at": "2026-10-19T00:00:00Z",
                    "runtime_vendor": "manual",
                    "runtime_endpoint": "https://example.azure.com",
                    "eco_provider": "openai",
                    "eco_model_name": "catalog-model",
                    "output_tokens": 42,
                    "impact_gwp_min": 0.1,
                    "impact_gwp_max": 0.2,
                    "impact_energy_min": 0.1,
                    "impact_energy_max": 0.2,
                    "impact_adpe_min": 0.1,
                    "impact_adpe_max": 0.2,
                    "impact_pe_min": 0.1,
                    "impact_pe_max": 0.2,
                    "impact_wcf_min": 0.1,
                    "impact_wcf_max": 0.2,
                }
            ]
        )
    )

    assert len(displayed) == 1
    assert displayed[0].columns.is_unique
    assert displayed[0].columns.tolist()[:5] == [
        "ID",
        "Runtime",
        "Endpoint",
        "Provider",
        "Model",
    ]


def test_saved_history_groups_models_by_estimation_second_and_numbers_groups(monkeypatch) -> None:
    displayed: list[pd.DataFrame] = []
    monkeypatch.setattr(app.st, "subheader", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app.st, "dataframe", lambda frame, **_kwargs: displayed.append(frame))

    base_record = {
        "runtime_vendor": "azure",
        "runtime_endpoint": "https://example.azure.com",
        "eco_provider": "openai",
        "output_tokens": 42,
        "impact_gwp_min": 0.1,
        "impact_gwp_max": 0.2,
        "impact_energy_min": 0.1,
        "impact_energy_max": 0.2,
        "impact_adpe_min": 0.1,
        "impact_adpe_max": 0.2,
        "impact_pe_min": 0.1,
        "impact_pe_max": 0.2,
        "impact_wcf_min": 0.1,
        "impact_wcf_max": 0.2,
    }
    app._summarize_history(
        pd.DataFrame(
            [
                {**base_record, "captured_at": "2026-10-19T00:00:00.100Z", "eco_model_name": "model-a"},
                {**base_record, "captured_at": "2026-10-19T00:00:00.900Z", "eco_model_name": "model-b"},
                {**base_record, "captured_at": "2026-10-18T00:00:00.000Z", "eco_model_name": "model-c"},
            ]
        )
    )

    history = displayed[0]
    assert history["ID"].tolist() == [1, 1, 2]
    assert dict(zip(history["Model"], history["ID"])) == {
        "model-a": 1,
        "model-b": 1,
        "model-c": 2,
    }
    assert "Captured at" not in history.columns
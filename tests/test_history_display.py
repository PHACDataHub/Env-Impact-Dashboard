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
                    "response_model": "runtime-model",
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
    assert displayed[0].columns.tolist()[2:5] == [
        "Runtime model",
        "Provider",
        "Model",
    ]
from models import CatalogIndex, ParsedInferencePayload
from resolver import resolve_ecologits_target


def test_gpt_model_resolves_to_openai_and_normalizes_spacing() -> None:
    catalog = CatalogIndex(
        providers=["openai"],
        models_by_provider={"openai": ["gpt-5-mini"]},
    )

    result = resolve_ecologits_target(
        ParsedInferencePayload(response_model="gpt 5 mini"),
        catalog,
    )

    assert result.status == "auto"
    assert result.eco_provider == "openai"
    assert result.eco_model_name == "gpt-5-mini"


def test_gpt_5_4_deployment_can_map_to_gpt_5_catalog_model() -> None:
    catalog = CatalogIndex(
        providers=["openai"],
        models_by_provider={"openai": ["gpt-5-mini"]},
    )

    result = resolve_ecologits_target(
        ParsedInferencePayload(response_model="gpt-5.4-mini"),
        catalog,
    )

    assert result.eco_provider == "openai"
    assert result.eco_model_name == "gpt-5-mini"
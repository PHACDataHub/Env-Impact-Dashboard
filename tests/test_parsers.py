from parsers import parse_inference_payload


def test_parse_invalid_json_returns_message() -> None:
    parsed = parse_inference_payload("not-json", "databricks")

    assert parsed.runtime_vendor == "databricks"
    assert parsed.response_model is None
    assert any("Invalid JSON" in message for message in parsed.parse_messages)
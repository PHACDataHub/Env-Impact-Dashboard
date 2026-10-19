from parsers import parse_inference_payload


def test_parse_invalid_json_returns_message() -> None:
    parsed = parse_inference_payload("not-json", "databricks")

    assert parsed.runtime_vendor == "databricks"
    assert parsed.response_model is None
    assert any("Invalid JSON" in message for message in parsed.parse_messages)


def test_parse_invalid_token_values_returns_message() -> None:
    parsed = parse_inference_payload(
        '{"model": "gpt-4o", "usage": {"completion_tokens": "unknown"}}',
        "databricks",
    )

    assert parsed.output_tokens is None
    assert any("Output tokens was present but not an integer" in message for message in parsed.parse_messages)
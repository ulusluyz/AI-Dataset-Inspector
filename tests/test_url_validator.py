import pytest
from backend.security.url_validator import validate_and_parse_dataset_url, URLValidationError

def test_valid_dataset_urls():
    valid_urls = [
        "https://huggingface.co/datasets/squad/squad",
        "https://huggingface.co/datasets/tatsu-lab/alpaca",
        "https://huggingface.co/datasets/allenai/c4/viewer/default/train",
        "https://www.huggingface.co/datasets/user123/my-dataset_v2.0",
        "http://huggingface.co/datasets/foo/bar?view=true#summary"
    ]
    for url in valid_urls:
        parsed = validate_and_parse_dataset_url(url)
        assert parsed.owner is not None
        assert parsed.dataset_name is not None
        assert parsed.dataset_id == f"{parsed.owner}/{parsed.dataset_name}"
        assert parsed.normalized_url.startswith("https://huggingface.co/datasets/")

def test_invalid_urls():
    invalid_cases = [
        ("https://huggingface.co/gpt2", "model"),
        ("https://huggingface.co/models/bert-base-uncased", "model"),
        ("https://huggingface.co/spaces/gradio/hello_world", "spaces"),
        ("https://google.co/datasets/squad/squad", "non-hf domain"),
        ("https://huggingface.co.fake.com/datasets/squad/squad", "fake domain"),
        ("https://127.0.0.1/datasets/squad/squad", "IP address"),
        ("file:///etc/passwd", "file scheme"),
        ("ftp://huggingface.co/datasets/foo/bar", "ftp scheme"),
        ("https://huggingface.co/datasets/", "missing owner and dataset"),
        ("https://huggingface.co/datasets/owner_only", "missing dataset name"),
    ]
    for url, desc in invalid_cases:
        with pytest.raises(URLValidationError):
            validate_and_parse_dataset_url(url)

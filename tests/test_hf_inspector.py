import pytest
from unittest.mock import MagicMock, patch
from backend.inspector.hf_inspector import HFInspector, DatasetMetadata, SampleResult

def test_fetch_metadata_success():
    inspector = HFInspector()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "# Dataset README"
    mock_resp.json.return_value = {
        "sha": "abc123sha",
        "private": False,
        "gated": False,
        "downloads": 1500,
        "likes": 42,
        "tags": ["license:mit", "task_categories:text-classification"],
        "siblings": [{"rfilename": "README.md"}, {"rfilename": "train.parquet"}]
    }

    with patch.object(inspector.client, "get", return_value=mock_resp):
        metadata = inspector.fetch_metadata("squad/squad")
        assert metadata.dataset_id == "squad/squad"
        assert metadata.sha == "abc123sha"
        assert metadata.license == "mit"
        assert metadata.downloads == 1500
        assert metadata.readme_content == "# Dataset README"

def test_fetch_metadata_not_found():
    inspector = HFInspector()
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch.object(inspector.client, "get", return_value=mock_resp):
        with pytest.raises(FileNotFoundError):
            inspector.fetch_metadata("nonexistent/dataset")

def test_sample_remote_dataset_with_viewer():
    inspector = HFInspector()

    metadata = DatasetMetadata(
        dataset_id="test/ds",
        sha="123",
        private=False,
        gated=False,
        downloads=100,
        likes=10,
        tags=[],
        license="apache-2.0"
    )

    def mock_get(url):
        mock_r = MagicMock()
        mock_r.text = "sample text"
        if "/info" in url:
            mock_r.status_code = 200
            mock_r.json.return_value = {
                "dataset_info": {
                    "default": {
                        "splits": {
                            "train": {"num_examples": 1000},
                            "test": {"num_examples": 200}
                        },
                        "features": {"text": {"dtype": "string"}}
                    }
                }
            }
        elif "/rows" in url:
            mock_r.status_code = 200
            mock_r.json.return_value = {
                "num_rows_total": 1000,
                "rows": [
                    {"row_idx": 0, "row": {"text": "Örnek Türkçe metin"}},
                    {"row_idx": 1, "row": {"text": "İkinci metin örneği"}}
                ]
            }
        else:
            mock_r.status_code = 404
        return mock_r

    with patch.object(inspector.client, "get", side_effect=mock_get):
        res = inspector.sample_remote_dataset("test/ds", metadata)
        assert res.viewer_available is True
        assert res.configs == ["default"]
        assert "train" in res.splits["default"]
        assert res.sampled_rows_count > 0
        assert res.confidence_hint in ["orta", "yüksek"]

import io
import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel
from backend.config import config
from backend.security.url_validator import check_ssrf_safe_host

logger = logging.getLogger(__name__)

HF_HUB_API_BASE = "https://huggingface.co/api/datasets"
HF_VIEWER_API_BASE = "https://datasets-server.huggingface.co"

class DatasetMetadata(BaseModel):
    dataset_id: str
    sha: Optional[str] = None
    private: bool = False
    gated: bool = False
    downloads: int = 0
    likes: int = 0
    tags: List[str] = []
    license: Optional[str] = None
    readme_content: Optional[str] = None
    license_content: Optional[str] = None
    files: List[Dict[str, Any]] = []

class SampledRow(BaseModel):
    config_name: str
    split_name: str
    row_index: int
    row_data: Dict[str, Any]

class SampleResult(BaseModel):
    dataset_id: str
    revision: Optional[str] = None
    viewer_available: bool = True
    configs: List[str] = []
    splits: Dict[str, List[str]] = {}  # config -> list of splits
    total_rows_estimated: int = 0
    sampled_rows_count: int = 0
    sampled_rows: List[SampledRow] = []
    schema_features: Dict[str, Any] = {}
    sampled_splits_breakdown: Dict[str, int] = {}
    sampling_notes: List[str] = []
    confidence_hint: str = "yüksek"

class HFInspector:
    def __init__(self, timeout: int = config.request_timeout):
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "AIDatasetInspector/1.0"}
        )

    def close(self):
        self.client.close()

    def fetch_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Fetches repository metadata, README, and file structure from HF Hub API."""
        if not check_ssrf_safe_host("huggingface.co"):
            raise RuntimeError("Hugging Face API domain DNS doğrulaması başarısız.")

        url = f"{HF_HUB_API_BASE}/{dataset_id}"
        resp = self.client.get(url)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Dataset '{dataset_id}' Hugging Face üzerinde bulunamadı.")
        elif resp.status_code == 401 or resp.status_code == 403:
            raise PermissionError(f"Dataset '{dataset_id}' erişime kapalı, gated veya özel (private) repository.")
        elif resp.status_code != 200:
            raise RuntimeError(f"Hugging Face API hatası ({resp.status_code}): {resp.text[:200]}")

        data = resp.json()
        sha = data.get("sha")
        private = data.get("private", False)
        gated = data.get("gated", False)
        downloads = data.get("downloads", 0)
        likes = data.get("likes", 0)
        tags = data.get("tags", [])

        # Extract license from tags or metadata
        license_info = None
        for tag in tags:
            if tag.startswith("license:"):
                license_info = tag.split("license:", 1)[1]
                break

        files = data.get("siblings", [])

        # Fetch README if available
        readme_content = self._fetch_repo_file(dataset_id, "README.md", sha)
        license_content = self._fetch_repo_file(dataset_id, "LICENSE", sha) or self._fetch_repo_file(dataset_id, "LICENSE.md", sha)

        return DatasetMetadata(
            dataset_id=dataset_id,
            sha=sha,
            private=private,
            gated=bool(gated),
            downloads=downloads,
            likes=likes,
            tags=tags,
            license=license_info,
            readme_content=readme_content,
            license_content=license_content,
            files=files
        )

    def _fetch_repo_file(self, dataset_id: str, filename: str, sha: Optional[str] = None) -> Optional[str]:
        ref = sha or "main"
        raw_url = f"https://huggingface.co/datasets/{dataset_id}/raw/{ref}/{filename}"
        try:
            resp = self.client.get(raw_url)
            if resp.status_code == 200:
                # Limit text size read to prevent memory explosion (e.g. max 500KB)
                return resp.text[:500_000]
        except Exception as e:
            logger.debug(f"Failed to fetch file {filename} for {dataset_id}: {e}")
        return None

    def sample_remote_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> SampleResult:
        """
        Adaptively samples rows across configs and splits remotely using Dataset Viewer API or Fallback.
        """
        viewer_available = True
        notes = []
        sampled_rows: List[SampledRow] = []
        splits_map: Dict[str, List[str]] = {}
        schema_features: Dict[str, Any] = {}
        total_estimated = 0
        splits_breakdown: Dict[str, int] = {}

        # 1. Try HF Dataset Viewer /info & /splits endpoints
        try:
            info_url = f"{HF_VIEWER_API_BASE}/info?dataset={dataset_id}"
            info_resp = self.client.get(info_url)

            if info_resp.status_code == 200:
                info_data = info_resp.json()
                dataset_info = info_data.get("dataset_info", {})
                configs = list(dataset_info.keys()) if isinstance(dataset_info, dict) else []

                for config_name, cfg_data in dataset_info.items():
                    splits_info = cfg_data.get("splits", {})
                    splits_map[config_name] = list(splits_info.keys())
                    features = cfg_data.get("features", {})
                    if features:
                        schema_features[config_name] = features

                    for s_name, s_meta in splits_info.items():
                        num_examples = s_meta.get("num_examples", 0)
                        total_estimated += num_examples

                notes.append(f"Dataset Viewer API aktif. {len(configs)} config ve {sum(len(v) for v in splits_map.values())} split bulundu.")
            else:
                viewer_available = False
                notes.append(f"Dataset Viewer /info döndürülemedi ({info_resp.status_code}). Remote fallback yöntemine geçiliyor.")
        except Exception as e:
            viewer_available = False
            notes.append(f"Dataset Viewer API erişim hatası: {str(e)}. Remote fallback kullanılıyor.")

        if viewer_available and splits_map:
            # Sample using Dataset Viewer /rows with distributed offsets (beginning, middle, end)
            for config_name, splits_list in splits_map.items():
                for split_name in splits_list:
                    # Determine target sample size per split adaptively (e.g., 50 to 300 rows per split)
                    split_samples = self._sample_via_viewer_rows(
                        dataset_id=dataset_id,
                        config_name=config_name,
                        split_name=split_name
                    )
                    sampled_rows.extend(split_samples)
                    key = f"{config_name}/{split_name}"
                    splits_breakdown[key] = len(split_samples)
        else:
            # Fallback remote sampling if Viewer API is disabled or not ready
            notes.append("Dataset Viewer kısmi/kullanılamaz durumda. Temel repository ve dosya bilgileri incelendi.")

        confidence_hint = "yüksek" if len(sampled_rows) >= 100 else ("orta" if len(sampled_rows) > 0 else "sınırlı")

        return SampleResult(
            dataset_id=dataset_id,
            revision=metadata.sha,
            viewer_available=viewer_available,
            configs=list(splits_map.keys()),
            splits=splits_map,
            total_rows_estimated=total_estimated,
            sampled_rows_count=len(sampled_rows),
            sampled_rows=sampled_rows,
            schema_features=schema_features,
            sampled_splits_breakdown=splits_breakdown,
            sampling_notes=notes,
            confidence_hint=confidence_hint
        )

    def _sample_via_viewer_rows(self, dataset_id: str, config_name: str, split_name: str) -> List[SampledRow]:
        """Fetches rows from /rows using multiple offsets to ensure representation."""
        rows_list: List[SampledRow] = []

        # Get total num_rows first if possible via length request
        base_rows_url = f"{HF_VIEWER_API_BASE}/rows?dataset={dataset_id}&config={config_name}&split={split_name}"
        try:
            # First fetch offset 0
            resp = self.client.get(f"{base_rows_url}&offset=0&length=50")
            if resp.status_code != 200:
                return []

            data = resp.json()
            features_rows = data.get("rows", [])
            num_rows_total = data.get("num_rows_total", len(features_rows))

            for idx, item in enumerate(features_rows):
                r_data = item.get("row", {})
                rows_list.append(SampledRow(
                    config_name=config_name,
                    split_name=split_name,
                    row_index=item.get("row_idx", idx),
                    row_data=r_data
                ))

            # If there are many rows, fetch middle and near-end offsets
            if num_rows_total > 200:
                mid_offset = max(100, num_rows_total // 2)
                end_offset = max(200, num_rows_total - 60)

                for offset in [mid_offset, end_offset]:
                    if offset < num_rows_total:
                        r_resp = self.client.get(f"{base_rows_url}&offset={offset}&length=25")
                        if r_resp.status_code == 200:
                            r_data = r_resp.json().get("rows", [])
                            for idx, item in enumerate(r_data):
                                rows_list.append(SampledRow(
                                    config_name=config_name,
                                    split_name=split_name,
                                    row_index=item.get("row_idx", offset + idx),
                                    row_data=item.get("row", {})
                                ))

        except Exception as e:
            logger.debug(f"Row sampling error for {config_name}/{split_name}: {e}")

        return rows_list

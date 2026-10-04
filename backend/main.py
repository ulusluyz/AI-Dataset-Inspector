import json
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import config
from backend.security.secret_store import secret_store
from backend.security.url_validator import validate_and_parse_dataset_url, URLValidationError
from backend.inspector.hf_inspector import HFInspector
from backend.analyzer.deterministic_analyzer import DeterministicAnalyzer
from backend.analyzer.gpt_analyzer import GPTAnalyzer, GPTAnalysisReport

logger = logging.getLogger("ai_dataset_inspector")

app = FastAPI(
    title="AI Dataset Inspector",
    description="Hugging Face Datasetlerini Uzaktan İnceleyen Güvenli Denetim Uygulaması",
    version="1.0.0"
)

# Request / Response Schemas
class SettingsRequest(BaseModel):
    openai_api_key: str
    openai_model: Optional[str] = "gpt-4o-mini"

class SettingsResponse(BaseModel):
    configured: bool
    masked_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

# In-memory storage for generated reports (report_id -> report_dict)
analysis_reports_cache: Dict[str, Dict[str, Any]] = {}

@app.get("/api/settings", response_model=SettingsResponse)
def get_settings():
    settings = secret_store.get_settings()
    key = settings.get("openai_api_key")
    model = settings.get("openai_model") or "gpt-4o-mini"
    return SettingsResponse(
        configured=bool(key),
        masked_key=secret_store.get_masked_key(),
        openai_model=model
    )

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    if not req.openai_api_key or not req.openai_api_key.strip().startswith("sk-"):
        raise HTTPException(status_code=400, detail="Geçersiz OpenAI API anahtarı formatı. 'sk-' ile başlamalıdır.")

    # Verify API key with OpenAI
    try:
        gpt_analyzer = GPTAnalyzer(api_key=req.openai_api_key.strip(), model=req.openai_model)
        models = gpt_analyzer.list_available_models()

        selected_model = req.openai_model if req.openai_model in models else (models[0] if models else "gpt-4o-mini")
        secret_store.save_settings(api_key=req.openai_api_key.strip(), model=selected_model)

        return {
            "success": True,
            "message": "OpenAI API anahtarı başarıyla doğrulandı ve kaydedildi.",
            "available_models": models,
            "selected_model": selected_model
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"API anahtarı doğrulanamadı: {str(e)}")

@app.delete("/api/settings")
def delete_settings():
    success = secret_store.delete_settings()
    if success:
        return {"success": True, "message": "API anahtarı ve ayarlar silindi."}
    raise HTTPException(status_code=500, detail="Ayarlar silinirken hata oluştu.")

@app.get("/api/models")
def get_available_models():
    settings = secret_store.get_settings()
    key = settings.get("openai_api_key")
    if not key:
        return {"models": ["gpt-4o-mini", "gpt-4o"]}
    try:
        analyzer = GPTAnalyzer(api_key=key)
        models = analyzer.list_available_models()
        return {"models": models}
    except Exception:
        return {"models": ["gpt-4o-mini", "gpt-4o"]}

@app.get("/api/analyze/stream")
async def analyze_dataset_stream(dataset_url: str = Query(...)):
    """Runs end-to-end dataset inspector agent with real-time SSE progress updates."""
    # 1. Validate settings
    settings = secret_store.get_settings()
    if not settings.get("openai_api_key"):
        err_msg = "OpenAI API anahtarı yapılandırılmamış. Lütfen önce Ayarlar bölümünden API anahtarınızı girin."
        async def err_gen_key():
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen_key(), media_type="text/event-stream")

    # 2. Validate URL
    try:
        parsed_url = validate_and_parse_dataset_url(dataset_url)
    except URLValidationError as err:
        err_msg = str(err)
        async def err_gen_url():
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen_url(), media_type="text/event-stream")

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'step': 1, 'message': '✓ URL doğrulandı. Hugging Face dataset bağlantısı onaylandı.'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)

            inspector = HFInspector()

            # Step 2: Fetch metadata
            yield f"data: {json.dumps({'type': 'progress', 'step': 2, 'message': '✓ Hugging Face bağlantısı kuruluyor. Repository metadata ve Dataset Card çekiliyor...'}, ensure_ascii=False)}\n\n"
            metadata = await asyncio.to_thread(inspector.fetch_metadata, parsed_url.dataset_id)
            await asyncio.sleep(0.2)

            # Step 3: Remote sampling
            yield f"data: {json.dumps({'type': 'progress', 'step': 3, 'message': f'✓ Splitler ve schema inceleniyor. Uzaktan temsil edici örnekleme başlatıldı...'}, ensure_ascii=False)}\n\n"
            sample_result = await asyncio.to_thread(inspector.sample_remote_dataset, parsed_url.dataset_id, metadata)
            inspector.close()
            await asyncio.sleep(0.2)

            # Step 4: Deterministic Analysis
            yield f"data: {json.dumps({'type': 'progress', 'step': 4, 'message': f'✓ {sample_result.sampled_rows_count} örnek uzaktan incelendi. Deterministik kod tabanlı kalite analizi yapılıyor...'}, ensure_ascii=False)}\n\n"
            det_analyzer = DeterministicAnalyzer()
            metrics = det_analyzer.analyze(sample_result.sampled_rows)
            await asyncio.sleep(0.2)

            # Step 5: GPT Semantic Analysis
            yield f"data: {json.dumps({'type': 'progress', 'step': 5, 'message': '✓ GPT semantik analizi ve LLM eğitim uygunluğu değerlendiriliyor...'}, ensure_ascii=False)}\n\n"
            gpt_analyzer = GPTAnalyzer()
            report: GPTAnalysisReport = await asyncio.to_thread(gpt_analyzer.analyze_dataset, metadata, sample_result, metrics)
            await asyncio.sleep(0.2)

            # Cache report for download endpoints
            report_dict = report.model_dump()
            report_id = parsed_url.dataset_id.replace("/", "_")
            analysis_reports_cache[report_id] = {
                "report": report_dict,
                "metadata": metadata.model_dump(),
                "sample_result": sample_result.model_dump(),
                "metrics": metrics.model_dump()
            }

            yield f"data: {json.dumps({'type': 'progress', 'step': 6, 'message': '✓ Analiz tamamlandı. Rapor oluşturuldu.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'report_id': report_id, 'report': report_dict}, ensure_ascii=False)}\n\n"

        except FileNotFoundError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset Bulunamadı: {str(e)}'}, ensure_ascii=False)}\n\n"
        except PermissionError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Erişim Engellendi: {str(e)}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("Analysis streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Analiz sırasında hata oluştu: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/reports/{report_id}/json")
def download_report_json(report_id: str):
    if report_id not in analysis_reports_cache:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    data = analysis_reports_cache[report_id]
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.json"}
    )

@app.get("/api/reports/{report_id}/markdown")
def download_report_markdown(report_id: str):
    if report_id not in analysis_reports_cache:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    data = analysis_reports_cache[report_id]
    rep = data["report"]
    metr = data["metrics"]

    md_content = f"""# DATASET ANALİZ RAPORU

**Dataset:** {rep['dataset_id']}
**Revision:** `{rep.get('revision') or 'main'}`
**Analiz Kararı:** {rep['download_recommendation']}
**Veri Kalitesi Skor:** {rep['data_quality']}
**Analiz Güven Seviyesi:** %{rep['analysis_confidence_score']} ({rep['analysis_confidence_reason']})

---

## 1. TEMEL VERİ BİLGİLERİ

- **Ana Dil:** {rep['primary_language']}
- **Dil Dağılımı:** {json.dumps(rep['language_distribution'], ensure_ascii=False)}
- **Veri Temizliği:** {rep['data_cleanliness']}
- **Veri Türü:** {', '.join(rep['dataset_types'])}
- **Veri Kaynağı / Köken:** {rep['data_origin_type']}
- **Sohbet Verisi:** {rep['is_chat_data']}
- **Instruction Verisi:** {rep['is_instruction_data']}
- **Soru-Cevap Verisi:** {rep['is_qa_data']}
- **Tekrar Riski:** {rep['repetition_risk']}
- **PII / Gizlilik Riski:** {rep['pii_risk']}
- **Lisans Durumu:** {rep['license_status']}
- **Provenance / Kaynak Açıklaması:** {rep['provenance_status']}

---

## 2. DETERMINİSTİK ÖLÇÜMLER

- **Toplam İncelenen Kayıt:** {metr['total_sampled']}
- **Exact Duplicate Sayısı / Oranı:** {metr['exact_duplicates_count']} (%{metr['exact_duplicate_ratio']*100:.1f})
- **Kısa Kayıt Sayısı:** {metr['short_rows_count']}
- **HTML / Markup Artığı:** {metr['html_junk_count']}
- **PII Eşleşmesi:** {metr['pii_matches_count']} ({', '.join(metr['pii_types_found']) if metr['pii_types_found'] else 'Yok'})

---

## 3. LLM EĞİTİM UYGUNLUĞU

| Eğitim Türü | Sonuç |
|---|---|
| Pretraining | {rep['llm_suitability']['pretraining']} |
| Continued Pretraining | {rep['llm_suitability']['continued_pretraining']} |
| Instruction/SFT | {rep['llm_suitability']['instruction_sft']} |
| Chat Training | {rep['llm_suitability']['chat_training']} |
| QA Training | {rep['llm_suitability']['qa_training']} |

**Açıklama:** {rep['llm_suitability']['explanation']}

---

## 4. DETAYLI AÇIKLAMA VE ÖNERİ

{rep['summary_explanation']}

### Nihai Karar: **{rep['download_recommendation']}**
"""
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.md"}
    )

# Serve Frontend static assets
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

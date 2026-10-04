import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import config
from backend.security.secret_store import secret_store
from backend.security.sanitizer import sanitize_json_payload, sanitize_error_message
from backend.security.url_validator import validate_and_parse_dataset_url, URLValidationError
from backend.inspector.hf_inspector import HFInspector
from backend.analyzer.deterministic_analyzer import DeterministicAnalyzer
from backend.providers.factory import LLMProviderFactory
from backend.providers.base_provider import SemanticAnalysisReport

logger = logging.getLogger("ai_dataset_inspector")

app = FastAPI(
    title="AI Dataset Inspector",
    description="Hugging Face Datasetlerini Uzaktan İnceleyen Güvenli Denetim Uygulaması",
    version="2.1.0"
)

# Request / Response Schemas
class SettingsRequest(BaseModel):
    provider: str = "gemini"  # gemini / openai
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = "gemini-2.5-flash"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = "gpt-4o-mini"
    max_estimated_cost_usd: Optional[float] = None
    max_input_tokens: Optional[int] = None

class SettingsResponse(BaseModel):
    provider: str = "gemini"
    gemini_configured: bool = False
    gemini_masked_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    openai_configured: bool = False
    openai_masked_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    max_estimated_cost_usd: Optional[float] = None
    max_input_tokens: Optional[int] = None

# In-memory storage for generated reports
analysis_reports_cache: Dict[str, Dict[str, Any]] = {}

@app.get("/api/settings", response_model=SettingsResponse)
def get_settings():
    settings = secret_store.get_settings()
    p = settings.get("provider", "gemini").lower()
    g_key = settings.get("gemini_api_key")
    o_key = settings.get("openai_api_key")

    return SettingsResponse(
        provider=p,
        gemini_configured=bool(g_key),
        gemini_masked_key=secret_store.get_masked_key("gemini"),
        gemini_model=settings.get("gemini_model") or "gemini-2.5-flash",
        openai_configured=bool(o_key),
        openai_masked_key=secret_store.get_masked_key("openai"),
        openai_model=settings.get("openai_model") or "gpt-4o-mini",
        max_estimated_cost_usd=settings.get("max_estimated_cost_usd"),
        max_input_tokens=settings.get("max_input_tokens")
    )

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    provider_name = req.provider.strip().lower() if req.provider else "gemini"
    if provider_name not in ("gemini", "openai"):
        raise HTTPException(status_code=400, detail="Geçersiz AI Sağlayıcısı. 'gemini' veya 'openai' seçiniz.")

    if provider_name == "gemini":
        key = req.gemini_api_key.strip() if req.gemini_api_key else secret_store.get_settings().get("gemini_api_key")
        if not key:
            raise HTTPException(status_code=400, detail="Lütfen geçerli bir Google Gemini API anahtarı girin.")
        model = req.gemini_model.strip() if req.gemini_model else "gemini-2.5-flash"

        try:
            llm_provider = LLMProviderFactory.get_provider("gemini", api_key=key, model_name=model)
            avail_models = llm_provider.list_models()
            selected_model = model if model in avail_models else (avail_models[0] if avail_models else "gemini-2.5-flash")
            secret_store.save_settings(
                provider="gemini",
                gemini_api_key=key,
                gemini_model=selected_model,
                max_estimated_cost_usd=req.max_estimated_cost_usd,
                max_input_tokens=req.max_input_tokens
            )

            return {
                "success": True,
                "message": "Google Gemini API anahtarı ve bütçe ayarları başarıyla kaydedildi.",
                "provider": "gemini",
                "available_models": avail_models,
                "selected_model": selected_model
            }
        except Exception as e:
            safe_err = sanitize_error_message(str(e))
            raise HTTPException(status_code=400, detail=f"Gemini API anahtarı doğrulanamadı: {safe_err}")

    else: # openai
        key = req.openai_api_key.strip() if req.openai_api_key else secret_store.get_settings().get("openai_api_key")
        if not key or not key.startswith("sk-"):
            raise HTTPException(status_code=400, detail="Geçersiz OpenAI API anahtarı formatı. 'sk-' ile başlamalıdır.")
        model = req.openai_model.strip() if req.openai_model else "gpt-4o-mini"

        try:
            llm_provider = LLMProviderFactory.get_provider("openai", api_key=key, model_name=model)
            avail_models = llm_provider.list_models()
            selected_model = model if model in avail_models else (avail_models[0] if avail_models else "gpt-4o-mini")
            secret_store.save_settings(
                provider="openai",
                openai_api_key=key,
                openai_model=selected_model,
                max_estimated_cost_usd=req.max_estimated_cost_usd,
                max_input_tokens=req.max_input_tokens
            )

            return {
                "success": True,
                "message": "OpenAI API anahtarı ve bütçe ayarları başarıyla kaydedildi.",
                "provider": "openai",
                "available_models": avail_models,
                "selected_model": selected_model
            }
        except Exception as e:
            safe_err = sanitize_error_message(str(e))
            raise HTTPException(status_code=400, detail=f"OpenAI API anahtarı doğrulanamadı: {safe_err}")

@app.delete("/api/settings")
def delete_settings():
    success = secret_store.delete_settings()
    if success:
        return {"success": True, "message": "API anahtarları ve sağlayıcı ayarları silindi."}
    raise HTTPException(status_code=500, detail="Ayarlar silinirken hata oluştu.")

@app.get("/api/models")
def get_available_models(provider: str = Query("gemini")):
    try:
        p_name = provider.strip().lower()
        llm_provider = LLMProviderFactory.get_provider(provider_name=p_name)
        models = llm_provider.list_models()
        return {"models": models}
    except Exception:
        if provider.lower() == "openai":
            return {"models": ["gpt-4o-mini", "gpt-4o"]}
        return {"models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]}

@app.get("/api/analyze/stream")
async def analyze_dataset_stream(dataset_url: str = Query(...)):
    """Runs end-to-end dataset inspector agent with token usage telemetry and SSE progress updates."""
    settings = secret_store.get_settings()
    active_p = settings.get("provider", "gemini").lower()
    active_key = settings.get(f"{active_p}_api_key")

    if not active_key:
        p_label = "Google Gemini" if active_p == "gemini" else "OpenAI"
        err_msg = f"{p_label} API anahtarı yapılandırılmamış. Lütfen önce Ayarlar bölümünden API anahtarınızı girin."
        async def err_gen_key():
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"
        return StreamingResponse(err_gen_key(), media_type="text/event-stream")

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
            yield f"data: {json.dumps({'type': 'progress', 'step': 3, 'message': '✓ Splitler ve schema inceleniyor. Uzaktan temsil edici örnekleme başlatıldı...'}, ensure_ascii=False)}\n\n"
            sample_result = await asyncio.to_thread(inspector.sample_remote_dataset, parsed_url.dataset_id, metadata)
            inspector.close()
            await asyncio.sleep(0.2)

            # Step 4: Deterministic Analysis
            yield f"data: {json.dumps({'type': 'progress', 'step': 4, 'message': f'✓ {sample_result.sampled_rows_count} örnek uzaktan incelendi. Deterministik kod tabanlı kalite analizi yapılıyor...'}, ensure_ascii=False)}\n\n"
            det_analyzer = DeterministicAnalyzer()
            metrics = det_analyzer.analyze(sample_result.sampled_rows)
            await asyncio.sleep(0.2)

            # Step 5: Active Provider Semantic Analysis & Telemetry
            p_label = "Google Gemini" if active_p == "gemini" else "OpenAI"
            yield f"data: {json.dumps({'type': 'progress', 'step': 5, 'message': f'✓ {p_label} semantik analizi ve LLM token telemetrisi değerlendiriliyor...'}, ensure_ascii=False)}\n\n"
            llm_provider = LLMProviderFactory.get_provider(active_p)
            report: SemanticAnalysisReport = await asyncio.to_thread(llm_provider.analyze_dataset, metadata, sample_result, metrics)
            await asyncio.sleep(0.2)

            # Defense-in-depth sanitization on report payload
            report_dict = sanitize_json_payload(report.model_dump())
            report_id = parsed_url.dataset_id.replace("/", "_")

            analysis_reports_cache[report_id] = {
                "report": report_dict,
                "metadata": metadata.model_dump(),
                "sample_result": sample_result.model_dump(),
                "metrics": metrics.model_dump()
            }

            yield f"data: {json.dumps({'type': 'progress', 'step': 6, 'message': '✓ Analiz tamamlandı. Rapor ve LLM kullanım telemetrisi oluşturuldu.'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'report_id': report_id, 'report': report_dict}, ensure_ascii=False)}\n\n"

        except FileNotFoundError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset Bulunamadı: {str(e)}'}, ensure_ascii=False)}\n\n"
        except PermissionError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Erişim Engellendi: {str(e)}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            safe_err = sanitize_error_message(str(e))
            logger.exception("Analysis streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Analiz sırasında hata oluştu: {safe_err}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/reports/{report_id}/json")
def download_report_json(report_id: str):
    if report_id not in analysis_reports_cache:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    data = analysis_reports_cache[report_id]
    sanitized_data = sanitize_json_payload(data)
    return JSONResponse(
        content=sanitized_data,
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.json"}
    )

@app.get("/api/reports/{report_id}/markdown")
def download_report_markdown(report_id: str):
    if report_id not in analysis_reports_cache:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    data = sanitize_json_payload(analysis_reports_cache[report_id])
    rep = data["report"]
    metr = data["metrics"]
    usage = rep.get("llm_usage") or {}

    md_content = f"""# DATASET ANALİZ RAPORU

**Dataset:** {rep['dataset_id']}
**Revision:** `{rep.get('revision') or 'main'}`
**AI Sağlayıcısı:** `{rep.get('llm_provider', 'gemini').upper()}` ({rep.get('llm_model', '')})
**Semantik Analiz Durumu:** {rep.get('semantic_analysis_status', 'Başarılı')}
**Analiz Kararı:** {rep['download_recommendation']}
**Veri Kalitesi Skor:** {rep['data_quality']}
**Analiz Güven Seviyesi:** %{rep['analysis_confidence_score']} ({rep['analysis_confidence_reason']})

---

## 1. LLM TOKEN KULLANIMI VE MALİYET ÖZETİ

- **Sağlayıcı / Model:** `{usage.get('provider', 'gemini').upper()}` / `{usage.get('model', '')}`
- **API Çağrı Sayısı:** {usage.get('api_call_count', 0)}
- **Kullanım Durumu:** {usage.get('usage_status', 'TAHMİNİ')}
- **Tahmini Input / Output / Toplam Token:** {usage.get('estimated_input_tokens', 0):,} / {usage.get('estimated_output_tokens', 0):,} / {usage.get('estimated_total_tokens', 0):,}
- **Gerçek Input / Output / Toplam Token:** {usage.get('actual_input_tokens') or 'Bildirilmedi'} / {usage.get('actual_output_tokens') or 'Bildirilmedi'} / {usage.get('actual_total_tokens') or 'Bildirilmedi'}
- **Tahmini Maliyet:** ${usage.get('estimated_cost_usd') or 0.0:.6f} USD
- **Gerçek Maliyet:** ${usage.get('actual_cost_usd') or 0.0:.6f} USD
- **Free Tier Durumu:** {usage.get('free_tier_status', 'Bilinmiyor')}

---

## 2. TEMEL VERİ BİLGİLERİ

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

## 3. DETERMINİSTİK ÖLÇÜMLER

- **Toplam İncelenen Kayıt:** {metr['total_sampled']}
- **Exact Duplicate Sayısı / Oranı:** {metr['exact_duplicates_count']} (%{metr['exact_duplicate_ratio']*100:.1f})
- **Kısa Kayıt Sayısı:** {metr['short_rows_count']}
- **HTML / Markup Artığı:** {metr['html_junk_count']}
- **PII Eşleşmesi:** {metr['pii_matches_count']} ({', '.join(metr['pii_types_found']) if metr['pii_types_found'] else 'Yok'})

---

## 4. LLM EĞİTİM UYGUNLUĞU

| Eğitim Türü | Sonuç |
|---|---|
| Pretraining | {rep['llm_suitability']['pretraining']} |
| Continued Pretraining | {rep['llm_suitability']['continued_pretraining']} |
| Instruction/SFT | {rep['llm_suitability']['instruction_sft']} |
| Chat Training | {rep['llm_suitability']['chat_training']} |
| QA Training | {rep['llm_suitability']['qa_training']} |

**Açıklama:** {rep['llm_suitability']['explanation']}

---

## 5. DETAYLI AÇIKLAMA VE ÖNERİ

{rep['summary_explanation']}

### Nihai Karar: **{rep['download_recommendation']}**
"""
    clean_md = sanitize_error_message(md_content)
    return Response(
        content=clean_md,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=report_{report_id}.md"}
    )

# Serve Frontend static assets
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

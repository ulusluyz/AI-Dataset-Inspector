# AI Dataset Inspector

Hugging Face üzerinde bulunan datasetleri bilgisayara indirmeden uzaktan inceleyen, verisetinin yapısını ve gerçek veri örneklerini analiz eden, LLM eğitimi açısından kalitesini sınıflandıran ve **Google Gemini API** veya **OpenAI API** destekli bir ajan aracılığıyla açıklamalı **İNDİR / DİKKAT / İNDİRME** kararı veren güvenli dataset denetim uygulaması.

---

## 🎯 Özellikler

- **Önce İncele, Sonra İndir:** Dataset'in tamamı bilgisayara indirilmeden Hugging Face Hub / Dataset Viewer API ve uzaktan adaptif örnekleme mekanizmaları ile analiz edilir.
- **Çoklu AI Sağlayıcısı Desteği (Google Gemini + OpenAI):**
  - **Google Gemini API (Varsayılan):** `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash` vb. modeller.
  - **OpenAI API:** `gpt-4o-mini`, `gpt-4o` modelleri.
- **LLM Token Telemetrisi & Bütçe Kontrolü:**
  - API çağrısı öncesi tahmini token kullanımı ve USD maliyet hesabı.
  - API yanıtından alınan gerçek token telemetrisi (`actual_input_tokens`, `actual_output_tokens`, `cached_input_tokens`).
  - İsteğe bağlı **Maksimum Analiz Maliyeti ($ USD)** ve **Maksimum Token Sınırı** bütçe koruma kontrolleri.
- **Güvenlik & Secret Sanitization:**
  - API keyler `.data/key.json` dosyasında kısıtlı OS dosya izinleriyle (`0600`) saklanır.
  - Merkezi **error sanitization** katmanı sayesinde ham exceptionlar, 401/429 hataları veya credential parçaları loglara, SSE mesajlarına, JSON veya Markdown raporlarına sızamaz.
- **URL Doğrulama & SSRF Koruması:** Yalnızca geçerli Hugging Face dataset URL'leri (`https://huggingface.co/datasets/<owner>/<dataset>`) kabul edilir. Model, Space, yerel IP ve sahte domainler reddedilir.
- **İki Katmanlı Analiz (Deterministik + LLM Semantik):**
  - **Katman A (Deterministik):** Exact duplicate oranı, kısa/uzun metinler, HTML artıkları, PII (e-posta, telefon, TC/kimlik) taraması, istatistiksel dil dağılımı.
  - **Katman B (LLM Semantik):** Doğal/sentetik veri ayrımı, sohbet/instruction yapısı, veri kalitesi skoru, LLM eğitim türlerine uygunluk (Pretraining, SFT, Chat, QA) ve Türkçe gerekçeli karar üretimi.
- **Untrusted Data Koruması & Prompt Injection Savunması:** Repository veya veri içerisindeki sistem talimatı benzeri metinler yalnızca incelenen veri olarak değerlendirilir.
- **Canlı Ajan İlerlemesi (SSE):** Analiz adımları ve LLM kullanım durumu gerçek zamanlı olarak arayüzde takip edilebilir.
- **Rapor Dışa Aktarma:** Analiz sonuçları ve LLM kullanım telemetrisi **JSON** ve **Markdown** formatlarında indirilebilir.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimler

- Python 3.10+
- `pip`

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatma

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Tarayıcınızda `http://localhost:8000` adresini açarak uygulamayı kullanabilirsiniz.

---

## 🔒 Güvenlik Modeli

1. **API Key Güvenliği:** API keyler `.data/key.json` dosyasında yalnızca işletim sistemi kullanıcısına özel okuma/yazma izniyle (`chmod 0600`) saklanır. `.gitignore` ile Git repository dışında tutulur.
2. **Secret Leakage Scrubbing:** `sanitize_error_message` ve `sanitize_json_payload` fonksiyonları `sk-`, `AIzaSy` ve Bearer token desenlerini maskeler.
3. **SSRF Savunması:** DNS resolution seviyesinde private IP blokları (127.0.0.1, 10.0.0.0/8, 169.254.0.0/16 vb.) engellenir.
4. **No Code Execution:** Repository içindeki Python, shell veya executable dosyalar çalıştırılmaz.

---

## 🧪 Testlerin Çalıştırılması

Unit, güvenlik, telemetri ve entegrasyon testlerini çalıştırmak için:

```bash
PYTHONPATH=. pytest tests/
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const openSettingsBtn = document.getElementById("openSettingsBtn");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const settingsForm = document.getElementById("settingsForm");

  const providerSelect = document.getElementById("providerSelect");
  const geminiKeyGroup = document.getElementById("geminiKeyGroup");
  const geminiKeyInput = document.getElementById("geminiKeyInput");
  const openaiKeyGroup = document.getElementById("openaiKeyGroup");
  const openaiKeyInput = document.getElementById("openaiKeyInput");

  const modelSelect = document.getElementById("modelSelect");
  const maxCostInput = document.getElementById("maxCostInput");
  const maxTokenInput = document.getElementById("maxTokenInput");
  const deleteKeyBtn = document.getElementById("deleteKeyBtn");
  const settingsMsg = document.getElementById("settingsMsg");
  const apiKeyStatusText = document.getElementById("apiKeyStatusText");

  const analyzeForm = document.getElementById("analyzeForm");
  const datasetUrlInput = document.getElementById("datasetUrlInput");
  const submitBtn = document.getElementById("submitBtn");

  const progressCard = document.getElementById("progressCard");
  const progressLogs = document.getElementById("progressLogs");
  const progressPercentText = document.getElementById("progressPercentText");

  const errorCard = document.getElementById("errorCard");
  const errorMessageText = document.getElementById("errorMessageText");

  const resultsCard = document.getElementById("resultsCard");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");
  const downloadMdBtn = document.getElementById("downloadMdBtn");

  let activeReportId = null;

  // Initialize Settings
  checkSettingsStatus();

  // Provider toggle visibility listener
  providerSelect.addEventListener("change", () => {
    updateProviderKeyVisibility();
    fetchAvailableModels(providerSelect.value);
  });

  function updateProviderKeyVisibility() {
    const selected = providerSelect.value;
    if (selected === "openai") {
      openaiKeyGroup.classList.remove("hidden");
      geminiKeyGroup.classList.add("hidden");
    } else {
      geminiKeyGroup.classList.remove("hidden");
      openaiKeyGroup.classList.add("hidden");
    }
  }

  // Modal Listeners
  openSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
    updateProviderKeyVisibility();
    fetchAvailableModels(providerSelect.value);
  });

  closeSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
  });

  // Save Settings
  settingsForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const provider = providerSelect.value;
    const gKey = geminiKeyInput.value.trim();
    const oKey = openaiKeyInput.value.trim();
    const selectedModel = modelSelect.value;
    const maxCost = maxCostInput.value ? parseFloat(maxCostInput.value) : null;
    const maxTokens = maxTokenInput.value ? parseInt(maxTokenInput.value) : null;

    showSettingsMsg("Doğrulanıyor...", "info");

    const payload = {
      provider: provider,
      gemini_api_key: gKey || null,
      gemini_model: provider === "gemini" ? selectedModel : null,
      openai_api_key: oKey || null,
      openai_model: provider === "openai" ? selectedModel : null,
      max_estimated_cost_usd: maxCost,
      max_input_tokens: maxTokens
    };

    try {
      const resp = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (resp.ok && data.success) {
        showSettingsMsg(`✓ ${data.message}`, "success");
        geminiKeyInput.value = "";
        openaiKeyInput.value = "";
        checkSettingsStatus();
        setTimeout(() => settingsModal.classList.add("hidden"), 1200);
      } else {
        showSettingsMsg(`Hata: ${data.detail || "API Key doğrulanamadı."}`, "error");
      }
    } catch (err) {
      showSettingsMsg(`Bağlantı hatası: ${err.message}`, "error");
    }
  });

  // Delete Settings
  deleteKeyBtn.addEventListener("click", async () => {
    if (!confirm("Tüm AI Sağlayıcısı ve API anahtarı ayarlarını silmek istediğinize emin misiniz?")) return;
    try {
      await fetch("/api/settings", { method: "DELETE" });
      checkSettingsStatus();
      showSettingsMsg("Tüm ayarlar ve API anahtarları silindi.", "info");
      geminiKeyInput.value = "";
      openaiKeyInput.value = "";
      maxCostInput.value = "";
      maxTokenInput.value = "";
    } catch (err) {
      showSettingsMsg("Silme hatası oluştu.", "error");
    }
  });

  async function checkSettingsStatus() {
    try {
      const resp = await fetch("/api/settings");
      const data = await resp.json();
      providerSelect.value = data.provider || "gemini";
      updateProviderKeyVisibility();

      if (data.max_estimated_cost_usd) maxCostInput.value = data.max_estimated_cost_usd;
      if (data.max_input_tokens) maxTokenInput.value = data.max_input_tokens;

      if (data.provider === "gemini" && data.gemini_configured) {
        apiKeyStatusText.textContent = `Gemini: ${data.gemini_masked_key} (${data.gemini_model})`;
        openSettingsBtn.className = "flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg border border-emerald-500/50 text-emerald-400 text-sm font-medium transition";
      } else if (data.provider === "openai" && data.openai_configured) {
        apiKeyStatusText.textContent = `OpenAI: ${data.openai_masked_key} (${data.openai_model})`;
        openSettingsBtn.className = "flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg border border-emerald-500/50 text-emerald-400 text-sm font-medium transition";
      } else {
        apiKeyStatusText.textContent = "AI Sağlayıcısı Yapılandır";
        openSettingsBtn.className = "flex items-center space-x-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg border border-amber-500/50 text-amber-400 text-sm font-medium transition";
      }
    } catch (err) {
      apiKeyStatusText.textContent = "Ayarlar Yüklenemedi";
    }
  }

  async function fetchAvailableModels(providerName) {
    try {
      const resp = await fetch(`/api/models?provider=${providerName}`);
      const data = await resp.json();
      if (data.models && data.models.length > 0) {
        modelSelect.innerHTML = "";
        data.models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m;
          opt.textContent = m;
          modelSelect.appendChild(opt);
        });
      }
    } catch (e) {
      console.log("Model list fetch error:", e);
    }
  }

  function showSettingsMsg(msg, type) {
    settingsMsg.textContent = msg;
    settingsMsg.classList.remove("hidden", "bg-indigo-950", "text-indigo-300", "bg-emerald-950", "text-emerald-300", "bg-red-950", "text-red-300");
    if (type === "success") {
      settingsMsg.classList.add("bg-emerald-950", "text-emerald-300");
    } else if (type === "error") {
      settingsMsg.classList.add("bg-red-950", "text-red-300");
    } else {
      settingsMsg.classList.add("bg-indigo-950", "text-indigo-300");
    }
  }

  // Handle Analysis Stream
  analyzeForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const url = datasetUrlInput.value.trim();
    if (!url) return;

    handleStartAnalysis(url);
  });

  function handleStartAnalysis(datasetUrl) {
    errorCard.classList.add("hidden");
    resultsCard.classList.add("hidden");
    progressCard.classList.remove("hidden");
    progressLogs.innerHTML = "";
    submitBtn.disabled = true;

    const eventSource = new EventSource(`/api/analyze/stream?dataset_url=${encodeURIComponent(datasetUrl)}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "progress") {
        progressPercentText.textContent = `Adım ${data.step}/6`;
        const logLine = document.createElement("div");
        logLine.className = "flex items-center space-x-2 text-slate-300 py-1 border-b border-slate-700/40";
        logLine.innerHTML = `<span class="text-emerald-400 font-bold">${data.message}</span>`;
        progressLogs.appendChild(logLine);
        progressLogs.scrollTop = progressLogs.scrollHeight;
      } else if (data.type === "complete") {
        eventSource.close();
        submitBtn.disabled = false;
        activeReportId = data.report_id;
        renderAnalysisResults(data.report);
      } else if (data.type === "error") {
        eventSource.close();
        submitBtn.disabled = false;
        progressCard.classList.add("hidden");
        errorCard.classList.remove("hidden");
        errorMessageText.textContent = data.message;
      }
    };

    eventSource.onerror = (err) => {
      eventSource.close();
      submitBtn.disabled = false;
      progressCard.classList.add("hidden");
      errorCard.classList.remove("hidden");
      errorMessageText.textContent = "Ajan sunucu bağlantısında beklenmeyen bir kopma oluştu.";
    };
  }

  function renderAnalysisResults(report) {
    progressCard.classList.add("hidden");
    resultsCard.classList.remove("hidden");

    // Decision Badge
    const decisionBadge = document.getElementById("decisionBadge");
    decisionBadge.textContent = report.download_recommendation;
    decisionBadge.className = "px-6 py-2 rounded-xl text-xl font-black tracking-wide shadow-xl uppercase ";
    if (report.download_recommendation === "İNDİR") {
      decisionBadge.classList.add("badge-indir");
    } else if (report.download_recommendation === "DİKKAT") {
      decisionBadge.classList.add("badge-dikkat");
    } else {
      decisionBadge.classList.add("badge-indirme");
    }

    // Quality, Provider & Confidence
    document.getElementById("qualityBadge").textContent = `Data Quality: ${report.data_quality}`;
    document.getElementById("providerBadge").textContent = `Provider: ${(report.llm_provider || 'gemini').toUpperCase()} (${report.llm_model || ''})`;
    document.getElementById("confidenceBadge").textContent = `Analiz Güveni: %${report.analysis_confidence_score}`;
    document.getElementById("datasetTitleText").textContent = report.dataset_id;
    document.getElementById("summaryText").textContent = report.summary_explanation;

    // Telemetry Summary Card
    const usage = report.llm_usage || {};
    const statusBadge = document.getElementById("telemetryStatusBadge");
    const uStatus = usage.usage_status || "TAHMİNİ";
    statusBadge.textContent = uStatus;
    if (uStatus === "GERÇEK") {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase bg-emerald-950 text-emerald-300 border border-emerald-800";
    } else if (uStatus === "DOĞRULANAMADI" || uStatus.includes("LİMİTİ")) {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase bg-red-950 text-red-300 border border-red-800";
    } else {
      statusBadge.className = "px-3 py-1 rounded-full text-xs font-bold uppercase bg-amber-950 text-amber-300 border border-amber-800";
    }

    document.getElementById("telemetryModelText").textContent = `${(usage.provider || 'gemini').toUpperCase()} (${usage.model || ''})`;
    document.getElementById("telemetryCallsText").textContent = usage.api_call_count || 0;

    const inTokens = usage.actual_input_tokens !== undefined && usage.actual_input_tokens !== null ? usage.actual_input_tokens : usage.estimated_input_tokens;
    const outTokens = usage.actual_output_tokens !== undefined && usage.actual_output_tokens !== null ? usage.actual_output_tokens : usage.estimated_output_tokens;
    const totTokens = usage.actual_total_tokens !== undefined && usage.actual_total_tokens !== null ? usage.actual_total_tokens : usage.estimated_total_tokens;
    const cost = usage.actual_cost_usd !== undefined && usage.actual_cost_usd !== null ? usage.actual_cost_usd : usage.estimated_cost_usd;

    document.getElementById("telemetryInTokenText").textContent = inTokens ? inTokens.toLocaleString() : "0";
    document.getElementById("telemetryOutTokenText").textContent = outTokens ? outTokens.toLocaleString() : "0";
    document.getElementById("telemetryTotTokenText").textContent = totTokens ? totTokens.toLocaleString() : "0";
    document.getElementById("telemetryCostText").textContent = cost !== null && cost !== undefined ? `$${cost.toFixed(6)} USD` : "Bilinmiyor";
    document.getElementById("telemetryFreeTierText").textContent = usage.free_tier_status ? `Free Tier Durumu: ${usage.free_tier_status}` : "";

    // Grid stats
    document.getElementById("primaryLangText").textContent = report.primary_language;
    document.getElementById("datasetTypesText").textContent = report.dataset_types.join(", ");
    document.getElementById("dataOriginText").textContent = report.data_origin_type;
    document.getElementById("isChatText").textContent = report.is_chat_data;
    document.getElementById("isInstructionText").textContent = report.is_instruction_data;
    document.getElementById("isQaText").textContent = report.is_qa_data;

    document.getElementById("cleanlinessText").textContent = report.data_cleanliness;
    document.getElementById("repetitionRiskText").textContent = report.repetition_risk;
    document.getElementById("piiRiskText").textContent = report.pii_risk;
    document.getElementById("licenseStatusText").textContent = report.license_status;
    document.getElementById("provenanceText").textContent = report.provenance_status;

    // LLM Suitability Matrix
    const suit = report.llm_suitability;
    const matrixContainer = document.getElementById("llmSuitabilityMatrix");
    matrixContainer.innerHTML = `
      <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
        <div class="text-xs text-slate-400 mb-1">Pretraining</div>
        <div class="text-sm font-bold ${getSuitabilityColor(suit.pretraining)}">${suit.pretraining}</div>
      </div>
      <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
        <div class="text-xs text-slate-400 mb-1">Continued Pretrain</div>
        <div class="text-sm font-bold ${getSuitabilityColor(suit.continued_pretraining)}">${suit.continued_pretraining}</div>
      </div>
      <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
        <div class="text-xs text-slate-400 mb-1">Instruction / SFT</div>
        <div class="text-sm font-bold ${getSuitabilityColor(suit.instruction_sft)}">${suit.instruction_sft}</div>
      </div>
      <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
        <div class="text-xs text-slate-400 mb-1">Chat Training</div>
        <div class="text-sm font-bold ${getSuitabilityColor(suit.chat_training)}">${suit.chat_training}</div>
      </div>
      <div class="bg-slate-900 p-3 rounded-xl border border-slate-700">
        <div class="text-xs text-slate-400 mb-1">QA Training</div>
        <div class="text-sm font-bold ${getSuitabilityColor(suit.qa_training)}">${suit.qa_training}</div>
      </div>
    `;
    document.getElementById("llmSuitabilityExplanation").textContent = suit.explanation;

    // Evidence List
    const evidenceList = document.getElementById("evidenceList");
    evidenceList.innerHTML = "";
    if (report.findings_evidence && report.findings_evidence.length > 0) {
      report.findings_evidence.forEach((ev) => {
        const item = document.createElement("div");
        item.className = "bg-slate-900 p-3 rounded-xl border border-slate-700/80 space-y-1";
        item.innerHTML = `
          <div class="flex items-center justify-between text-xs font-semibold text-purple-300">
            <span><i class="fa-solid fa-bug mr-1"></i> ${ev.bulgu || "Bulgu"}</span>
            <span class="text-slate-400">${ev.split || ""}</span>
          </div>
          <p class="text-xs font-mono text-slate-300 bg-slate-950 p-2 rounded border border-slate-800 break-all">${ev.ornek || ""}</p>
        `;
        evidenceList.appendChild(item);
      });
    } else {
      evidenceList.innerHTML = `<div class="text-xs text-slate-400 italic">Ciddi veya kritik olumsuz kanıt tespit edilmedi.</div>`;
    }
  }

  function getSuitabilityColor(val) {
    if (val.includes("Çok uygun")) return "text-emerald-400";
    if (val.includes("Uygun") && !val.includes("değil")) return "text-emerald-300";
    if (val.includes("Kısmen")) return "text-amber-400";
    return "text-red-400";
  }

  // Export handlers
  downloadJsonBtn.addEventListener("click", () => {
    if (activeReportId) {
      window.location.href = `/api/reports/${activeReportId}/json`;
    }
  });

  downloadMdBtn.addEventListener("click", () => {
    if (activeReportId) {
      window.location.href = `/api/reports/${activeReportId}/markdown`;
    }
  });
});

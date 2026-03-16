const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(text) {
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}

function appendLog(line) {
  const logWindow = qs('#logWindow') || qs('#historyLogs');
  if (!logWindow) return;
  const entry = document.createElement('div');
  entry.innerHTML = escapeHtml(line);
  logWindow.appendChild(entry);
  logWindow.scrollTop = logWindow.scrollHeight;
  const limitAttr = logWindow.dataset.limit;
  const limit = limitAttr ? Number(limitAttr) : 2000;
  if (Number.isFinite(limit) && limit > 0 && logWindow.childNodes.length > limit) {
    logWindow.removeChild(logWindow.firstChild);
  }
}

async function fetchTasks() {
  const res = await fetch('/api/tasks');
  const data = await res.json();
  return data.tasks || [];
}

function renderTaskList(tasks) {
  const taskList = qs('#taskList');
  if (!taskList) return;
  taskList.innerHTML = '';
  tasks.forEach((task) => {
    const card = document.createElement('div');
    card.className = 'task-card';
    const status = task.status || 'Unknown';
    const progress = task.progress || 0;
    const output = task.output_path;
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <div>
          <div class="task-status text-amber-300">${status}</div>
          <div class="text-sm text-slate-200 mt-1">${task.id}</div>
        </div>
        ${output ? `<a class="btn-ghost" href="/api/download/${task.id}">Download</a>` : ''}
      </div>
      <div class="text-xs text-slate-400 mt-2">${escapeHtml(task.prompt || '')}</div>
      <div class="progress-bar mt-3"><div class="progress-fill" style="width:${progress}%"></div></div>
    `;
    taskList.appendChild(card);
  });
}

function renderHistory(tasks) {
  const historyList = qs('#historyList');
  if (!historyList) return;
  historyList.innerHTML = '';
  tasks.forEach((task) => {
    const card = document.createElement('div');
    card.className = 'task-card';
    const output = task.output_path;
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <div>
          <div class="task-status text-amber-300">${task.status}</div>
          <div class="text-sm text-slate-200 mt-1">${task.id}</div>
        </div>
        <div class="flex gap-2">
          <button class="btn-ghost" data-log="${task.id}">View Logs</button>
          <button class="btn-ghost" data-full-log="${task.id}">View Full Logs</button>
          <a class="btn-ghost" href="/api/logs/${task.id}">Download Logs</a>
          ${output ? `<a class="btn-ghost" href="/api/download/${task.id}">Download</a>` : ''}
        </div>
      </div>
      <div class="text-xs text-slate-400 mt-2">${escapeHtml(task.prompt || '')}</div>
      <div class="progress-bar mt-3"><div class="progress-fill" style="width:${task.progress || 0}%"></div></div>
    `;
    historyList.appendChild(card);
  });

  qsa('[data-log]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const taskId = btn.getAttribute('data-log');
      const res = await fetch(`/api/tasks/${taskId}/logs?limit=20000`);
      const data = await res.json();
      const logWindow = qs('#historyLogs');
      if (logWindow) {
        logWindow.innerHTML = '';
        logWindow.dataset.limit = '2000';
        (data.logs || []).forEach((log) => {
          appendLog(`[${log.timestamp}] ${log.level.toUpperCase()}: ${log.message}`);
        });
      }
    });
  });

  qsa('[data-full-log]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const taskId = btn.getAttribute('data-full-log');
      const res = await fetch(`/api/logs/${taskId}`);
      const logWindow = qs('#historyLogs');
      if (!logWindow) return;
      logWindow.innerHTML = '';
      logWindow.dataset.limit = '0';
      if (!res.ok) {
        appendLog('Failed to load full log file.');
        return;
      }
      const text = await res.text();
      const pre = document.createElement('pre');
      pre.className = 'whitespace-pre-wrap text-xs text-slate-200';
      pre.textContent = text;
      logWindow.appendChild(pre);
    });
  });
}

function connectWebSocket(activeTaskIdRef) {
  const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProtocol}://${location.host}/ws/logs?task_id=all`);
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    const { task_id, level, message } = payload;
    if (level === 'progress') {
      try {
        const data = JSON.parse(message);
        if (task_id === activeTaskIdRef.value) {
          const progressFill = qs('#progressFill');
          if (progressFill) {
            progressFill.style.width = `${data.progress || 0}%`;
          }
        }
      } catch (_) {
        return;
      }
    } else if (task_id === activeTaskIdRef.value) {
      appendLog(`[${level.toUpperCase()}] ${message}`);
    }
  };
}

async function initDashboard() {
  const generateBtn = qs('#generateBtn');
  if (!generateBtn) return;

  const activeTaskIdRef = { value: null };
  connectWebSocket(activeTaskIdRef);

  try {
    const settingsRes = await fetch('/api/settings');
    if (settingsRes.ok) {
      const settingsData = await settingsRes.json();
      const defaults = settingsData.settings?.pipeline_defaults || {};
      const addMusic = qs('#add_music');
      const addGreeting = qs('#add_greeting');
      const addClosing = qs('#add_closing');
      const useStock = qs('#use_stock_video');
      const useImages = qs('#use_images');
      const burnSubs = qs('#burn_subtitles');
      if (addMusic) addMusic.checked = defaults.add_music ?? true;
      if (addGreeting) addGreeting.checked = defaults.add_greeting ?? false;
      if (addClosing) addClosing.checked = defaults.add_closing ?? false;
      if (useStock) useStock.checked = defaults.use_stock_video ?? true;
      if (useImages) useImages.checked = defaults.use_images ?? true;
      if (burnSubs) burnSubs.checked = defaults.burn_subtitles ?? true;
    }
  } catch (_) {
    // ignore settings load errors on dashboard
  }

  const refresh = async () => {
    const tasks = await fetchTasks();
    renderTaskList(tasks);
  };

  qs('#refreshTasks')?.addEventListener('click', refresh);
  await refresh();

  generateBtn.addEventListener('click', async () => {
    const payload = {
      prompt: qs('#prompt').value,
      format: qs('#format').value,
      language: qs('#language').value,
      duration: parseInt(qs('#duration').value, 10),
      add_music: qs('#add_music').checked,
      add_greeting: qs('#add_greeting')?.checked ?? false,
      add_closing: qs('#add_closing')?.checked ?? false,
      use_stock_video: qs('#use_stock_video').checked,
      use_images: qs('#use_images').checked,
      burn_subtitles: qs('#burn_subtitles').checked,
    };

    if (!payload.prompt) {
      appendLog('Prompt is required.');
      return;
    }

    generateBtn.disabled = true;
    qs('#taskStatus').textContent = 'Queued...';
    qs('#logWindow').innerHTML = '';

    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      appendLog(data.error || 'Failed to start generation.');
      generateBtn.disabled = false;
      return;
    }

    activeTaskIdRef.value = data.task_id;
    qs('#taskStatus').textContent = `Running: ${data.task_id}`;

    const poll = setInterval(async () => {
      const tasks = await fetchTasks();
      renderTaskList(tasks);
      const active = tasks.find((t) => t.id === data.task_id);
      if (!active) return;
      qs('#taskStatus').textContent = `${active.status} (${active.progress || 0}%)`;
      if (active.status === 'Completed' || active.status === 'Failed') {
        clearInterval(poll);
        generateBtn.disabled = false;
      }
    }, 3000);
  });
}

async function initSettings() {
  const saveBtn = qs('#saveSettings');
  if (!saveBtn) return;

  const numOr = (value, fallback) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  };

  const populateSelect = (selectEl, models, current) => {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    const list = models && models.length ? models : [current || 'default'];
    list.forEach((name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      if (name === current) option.selected = true;
      selectEl.appendChild(option);
    });
  };

  const fetchModels = async (apiUrl) => {
    if (!apiUrl) return [];
    const res = await fetch(`/api/ollama/tags?url=${encodeURIComponent(apiUrl)}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.models || [];
  };

  const res = await fetch('/api/settings');
  const data = await res.json();
  const settings = data.settings || {};

  populateSelect(qs('#ollama_model'), data.models || [], settings.ollama_model);
  populateSelect(qs('#planner_model'), data.planner_models || [], settings.ollama_planner_model);

  qs('#ollama_api_url').value = settings.ollama_api_url || '';
  qs('#ollama_timeout').value = settings.ollama_timeout ?? 180;
  qs('#ollama_request_delay').value = settings.ollama_request_delay ?? 0.8;
  const thinkToggle = qs('#ollama_think');
  if (thinkToggle) thinkToggle.checked = settings.ollama_think ?? false;
  const params = settings.ollama_params || {};
  qs('#num_ctx').value = params.num_ctx ?? 4096;
  qs('#num_thread').value = params.num_thread ?? 20;
  qs('#temperature').value = params.temperature ?? 0.7;
  qs('#top_k').value = params.top_k ?? 40;
  qs('#top_p').value = params.top_p ?? 0.9;
  qs('#repeat_penalty').value = params.repeat_penalty ?? 1.1;
  qs('#num_predict').value = params.num_predict ?? 1024;

  qs('#planner_api_url').value = settings.ollama_planner_api_url || '';
  qs('#planner_timeout').value = settings.ollama_planner_timeout ?? 120;
  qs('#planner_think').checked = settings.ollama_planner_think ?? false;
  const plannerParams = settings.ollama_planner_params || {};
  qs('#planner_num_ctx').value = plannerParams.num_ctx ?? 4096;
  qs('#planner_num_thread').value = plannerParams.num_thread ?? 20;
  qs('#planner_temperature').value = plannerParams.temperature ?? 0.3;
  qs('#planner_top_k').value = plannerParams.top_k ?? 40;
  qs('#planner_top_p').value = plannerParams.top_p ?? 0.9;
  qs('#planner_repeat_penalty').value = plannerParams.repeat_penalty ?? 1.05;
  qs('#planner_num_predict').value = plannerParams.num_predict ?? 512;

  const video = settings.video_settings || {};
  qs('#resolution_landscape').value = video.resolution_landscape || '1280x720';
  qs('#resolution_portrait').value = video.resolution_portrait || '720x1280';
  qs('#fps').value = video.fps ?? 30;
  qs('#font_name').value = video.font_name || 'Inter';
  qs('#font_size').value = video.font_size ?? 52;
  qs('#font_color').value = video.font_color || '&H00FFFFFF';
  qs('#outline_color').value = video.outline_color || '&H00000000';
  qs('#outline').value = video.outline ?? 2;
  qs('#shadow').value = video.shadow ?? 1;
  qs('#subtitle_position').value = video.subtitle_position || 'bottom';
  qs('#subtitle_margin_x').value = video.subtitle_margin_x ?? 60;
  qs('#subtitle_margin_y').value = video.subtitle_margin_y ?? 60;

  qs('#tts_engine').value = settings.tts_engine || 'piper';
  qs('#coqui_model').value = settings.coqui_model || 'tts_models/en/vctk/vits';
  qs('#coqui_speaker').value = settings.coqui_speaker || '';
  qs('#tts_voice_path').value = settings.tts_voice_path || '/models/piper/en_US-lessac-medium.onnx';
  qs('#tts_voice_config').value = settings.tts_voice_config || '/models/piper/en_US-lessac-medium.onnx.json';
  qs('#voiceover_wps').value = settings.voiceover_words_per_sec ?? 2.2;

  qs('#helper_api_url').value = settings.ollama_helper_api_url || '';
  qs('#helper_timeout').value = settings.ollama_helper_timeout ?? 120;
  qs('#helper_think').checked = settings.ollama_helper_think ?? false;
  const helperParams = settings.ollama_helper_params || {};
  qs('#helper_num_ctx').value = helperParams.num_ctx ?? 4096;
  qs('#helper_num_thread').value = helperParams.num_thread ?? 20;
  qs('#helper_temperature').value = helperParams.temperature ?? 0.2;
  qs('#helper_top_k').value = helperParams.top_k ?? 40;
  qs('#helper_top_p').value = helperParams.top_p ?? 0.9;
  qs('#helper_repeat_penalty').value = helperParams.repeat_penalty ?? 1.05;
  qs('#helper_num_predict').value = helperParams.num_predict ?? 512;

  qs('#vision_api_url').value = settings.ollama_vision_api_url || '';
  qs('#vision_timeout').value = settings.ollama_vision_timeout ?? 120;
  qs('#vision_think').checked = settings.ollama_vision_think ?? false;
  qs('#vision_enabled').checked = settings.ollama_vision_enabled ?? false;
  const visionParams = settings.ollama_vision_params || {};
  qs('#vision_num_ctx').value = visionParams.num_ctx ?? 4096;
  qs('#vision_num_thread').value = visionParams.num_thread ?? 20;
  qs('#vision_temperature').value = visionParams.temperature ?? 0.2;
  qs('#vision_top_k').value = visionParams.top_k ?? 40;
  qs('#vision_top_p').value = visionParams.top_p ?? 0.9;
  qs('#vision_repeat_penalty').value = visionParams.repeat_penalty ?? 1.05;
  qs('#vision_num_predict').value = visionParams.num_predict ?? 256;

  populateSelect(qs('#helper_model'), data.helper_models || [], settings.ollama_helper_model);
  populateSelect(qs('#vision_model'), data.vision_models || [], settings.ollama_vision_model);

  const wireRefresh = (apiInputId, refreshBtnId, selectId, currentValue) => {
    const apiInput = qs(apiInputId);
    const refreshBtn = qs(refreshBtnId);
    const selectEl = qs(selectId);
    let timer = null;
    const refresh = async () => {
      const models = await fetchModels(apiInput?.value || '');
      populateSelect(selectEl, models, selectEl?.value || currentValue);
    };
    apiInput?.addEventListener('change', refresh);
    apiInput?.addEventListener('input', () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(refresh, 400);
    });
    refreshBtn?.addEventListener('click', refresh);
  };

  wireRefresh('#ollama_api_url', '#refreshOllamaModels', '#ollama_model', settings.ollama_model);
  wireRefresh('#planner_api_url', '#refreshPlannerModels', '#planner_model', settings.ollama_planner_model);
  wireRefresh('#helper_api_url', '#refreshHelperModels', '#helper_model', settings.ollama_helper_model);
  wireRefresh('#vision_api_url', '#refreshVisionModels', '#vision_model', settings.ollama_vision_model);

  const pipelineDefaults = settings.pipeline_defaults || {};
  const defaultAddMusic = qs('#default_add_music');
  const defaultAddGreeting = qs('#default_add_greeting');
  const defaultAddClosing = qs('#default_add_closing');
  const defaultUseStock = qs('#default_use_stock_video');
  const defaultUseImages = qs('#default_use_images');
  const defaultBurnSubs = qs('#default_burn_subtitles');
  if (defaultAddMusic) defaultAddMusic.checked = pipelineDefaults.add_music ?? true;
  if (defaultAddGreeting) defaultAddGreeting.checked = pipelineDefaults.add_greeting ?? false;
  if (defaultAddClosing) defaultAddClosing.checked = pipelineDefaults.add_closing ?? false;
  if (defaultUseStock) defaultUseStock.checked = pipelineDefaults.use_stock_video ?? true;
  if (defaultUseImages) defaultUseImages.checked = pipelineDefaults.use_images ?? true;
  if (defaultBurnSubs) defaultBurnSubs.checked = pipelineDefaults.burn_subtitles ?? true;

  const scraper = settings.scraper_settings || {};
  qs('#scraper_request_delay').value = scraper.request_delay_sec ?? 1.2;
  qs('#scraper_sleep_min').value = scraper.yt_dlp_sleep_min ?? 1.0;
  qs('#scraper_sleep_max').value = scraper.yt_dlp_sleep_max ?? 3.0;
  qs('#scraper_search_count').value = scraper.yt_dlp_search_count ?? 8;
  qs('#scraper_image_delay').value = scraper.image_delay_sec ?? 0.6;

  saveBtn.addEventListener('click', async () => {
    const payload = {
      ollama_api_url: qs('#ollama_api_url').value,
      ollama_model: qs('#ollama_model').value,
      ollama_timeout: numOr(qs('#ollama_timeout').value, settings.ollama_timeout ?? 180),
      ollama_request_delay: numOr(qs('#ollama_request_delay').value, settings.ollama_request_delay ?? 0.8),
      ollama_think: qs('#ollama_think')?.checked ?? false,
      ollama_params: {
        num_ctx: numOr(qs('#num_ctx').value, params.num_ctx ?? 4096),
        num_thread: numOr(qs('#num_thread').value, params.num_thread ?? 20),
        temperature: numOr(qs('#temperature').value, params.temperature ?? 0.7),
        top_k: numOr(qs('#top_k').value, params.top_k ?? 40),
        top_p: numOr(qs('#top_p').value, params.top_p ?? 0.9),
        repeat_penalty: numOr(qs('#repeat_penalty').value, params.repeat_penalty ?? 1.1),
        num_predict: numOr(qs('#num_predict').value, params.num_predict ?? 1024),
      },
      ollama_planner_api_url: qs('#planner_api_url').value,
      ollama_planner_model: qs('#planner_model').value,
      ollama_planner_timeout: numOr(qs('#planner_timeout').value, settings.ollama_planner_timeout ?? 120),
      ollama_planner_think: qs('#planner_think')?.checked ?? false,
      ollama_planner_params: {
        num_ctx: numOr(qs('#planner_num_ctx').value, plannerParams.num_ctx ?? 4096),
        num_thread: numOr(qs('#planner_num_thread').value, plannerParams.num_thread ?? 20),
        temperature: numOr(qs('#planner_temperature').value, plannerParams.temperature ?? 0.3),
        top_k: numOr(qs('#planner_top_k').value, plannerParams.top_k ?? 40),
        top_p: numOr(qs('#planner_top_p').value, plannerParams.top_p ?? 0.9),
        repeat_penalty: numOr(qs('#planner_repeat_penalty').value, plannerParams.repeat_penalty ?? 1.05),
        num_predict: numOr(qs('#planner_num_predict').value, plannerParams.num_predict ?? 512),
      },
      video_settings: {
        resolution_landscape: qs('#resolution_landscape').value,
        resolution_portrait: qs('#resolution_portrait').value,
        fps: numOr(qs('#fps').value, video.fps ?? 30),
        font_name: qs('#font_name').value,
        font_size: numOr(qs('#font_size').value, video.font_size ?? 52),
        font_color: qs('#font_color').value,
        outline_color: qs('#outline_color').value,
        outline: numOr(qs('#outline').value, video.outline ?? 2),
        shadow: numOr(qs('#shadow').value, video.shadow ?? 1),
        subtitle_position: qs('#subtitle_position').value,
        subtitle_margin_x: numOr(qs('#subtitle_margin_x').value, video.subtitle_margin_x ?? 60),
        subtitle_margin_y: numOr(qs('#subtitle_margin_y').value, video.subtitle_margin_y ?? 60),
      },
      tts_engine: qs('#tts_engine').value,
      coqui_model: qs('#coqui_model').value,
      coqui_speaker: qs('#coqui_speaker').value,
      tts_voice_path: qs('#tts_voice_path').value,
      tts_voice_config: qs('#tts_voice_config').value,
      voiceover_words_per_sec: numOr(qs('#voiceover_wps').value, settings.voiceover_words_per_sec ?? 2.2),
      ollama_helper_api_url: qs('#helper_api_url').value,
      ollama_helper_model: qs('#helper_model').value,
      ollama_helper_timeout: numOr(qs('#helper_timeout').value, settings.ollama_helper_timeout ?? 120),
      ollama_helper_think: qs('#helper_think').checked,
      ollama_helper_params: {
        num_ctx: numOr(qs('#helper_num_ctx').value, helperParams.num_ctx ?? 4096),
        num_thread: numOr(qs('#helper_num_thread').value, helperParams.num_thread ?? 20),
        temperature: numOr(qs('#helper_temperature').value, helperParams.temperature ?? 0.2),
        top_k: numOr(qs('#helper_top_k').value, helperParams.top_k ?? 40),
        top_p: numOr(qs('#helper_top_p').value, helperParams.top_p ?? 0.9),
        repeat_penalty: numOr(qs('#helper_repeat_penalty').value, helperParams.repeat_penalty ?? 1.05),
        num_predict: numOr(qs('#helper_num_predict').value, helperParams.num_predict ?? 512),
      },
      ollama_vision_api_url: qs('#vision_api_url').value,
      ollama_vision_model: qs('#vision_model').value,
      ollama_vision_timeout: numOr(qs('#vision_timeout').value, settings.ollama_vision_timeout ?? 120),
      ollama_vision_think: qs('#vision_think').checked,
      ollama_vision_enabled: qs('#vision_enabled').checked,
      ollama_vision_params: {
        num_ctx: numOr(qs('#vision_num_ctx').value, visionParams.num_ctx ?? 4096),
        num_thread: numOr(qs('#vision_num_thread').value, visionParams.num_thread ?? 20),
        temperature: numOr(qs('#vision_temperature').value, visionParams.temperature ?? 0.2),
        top_k: numOr(qs('#vision_top_k').value, visionParams.top_k ?? 40),
        top_p: numOr(qs('#vision_top_p').value, visionParams.top_p ?? 0.9),
        repeat_penalty: numOr(qs('#vision_repeat_penalty').value, visionParams.repeat_penalty ?? 1.05),
        num_predict: numOr(qs('#vision_num_predict').value, visionParams.num_predict ?? 256),
      },
      pipeline_defaults: {
        add_music: qs('#default_add_music')?.checked ?? true,
        add_greeting: qs('#default_add_greeting')?.checked ?? false,
        add_closing: qs('#default_add_closing')?.checked ?? false,
        use_stock_video: qs('#default_use_stock_video')?.checked ?? true,
        use_images: qs('#default_use_images')?.checked ?? true,
        burn_subtitles: qs('#default_burn_subtitles')?.checked ?? true,
      },
      scraper_settings: {
        request_delay_sec: numOr(qs('#scraper_request_delay').value, scraper.request_delay_sec ?? 1.2),
        yt_dlp_sleep_min: numOr(qs('#scraper_sleep_min').value, scraper.yt_dlp_sleep_min ?? 1.0),
        yt_dlp_sleep_max: numOr(qs('#scraper_sleep_max').value, scraper.yt_dlp_sleep_max ?? 3.0),
        yt_dlp_search_count: numOr(qs('#scraper_search_count').value, scraper.yt_dlp_search_count ?? 8),
        image_delay_sec: numOr(qs('#scraper_image_delay').value, scraper.image_delay_sec ?? 0.6),
      },
    };

    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const status = qs('#settingsStatus');
    if (res.ok) {
      status.textContent = 'Settings saved.';
    } else {
      status.textContent = 'Failed to save settings.';
    }
  });
}

async function initHistory() {
  const refreshBtn = qs('#refreshHistory');
  if (!refreshBtn) return;

  const refresh = async () => {
    const tasks = await fetchTasks();
    renderHistory(tasks);
  };

  refreshBtn.addEventListener('click', refresh);
  await refresh();
}

window.addEventListener('DOMContentLoaded', () => {
  initDashboard();
  initSettings();
  initHistory();
});

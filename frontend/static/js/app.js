/**
 * app.js — Drone Security Analyst Agent frontend
 * Vanilla JS, no dependencies, communicates with FastAPI backend.
 */

// ── State ──────────────────────────────────────────────────────────
const state = {
  activeSession: null,
  framesPage: 1,
  framesTotal: 0,
  alertPollingInterval: null,
  sessionPollingInterval: null,
};

// ── DOM refs ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Clock ──────────────────────────────────────────────────────────
function startClock() {
  const tick = () => {
    $('clock').textContent = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  };
  tick();
  setInterval(tick, 1000);
}

// ── Connection check ───────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (r.ok) {
      $('connDot').className = 'status-dot ok';
      $('connLabel').textContent = 'ONLINE';
    } else throw new Error();
  } catch {
    $('connDot').className = 'status-dot err';
    $('connLabel').textContent = 'OFFLINE';
  }
}

// ── API helpers ────────────────────────────────────────────────────
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
}

async function apiPatch(url) {
  const r = await fetch(url, { method: 'PATCH' });
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ── Sessions ───────────────────────────────────────────────────────
async function loadSessions() {
  const list = $('sessionList');
  list.innerHTML = '<div class="loader"></div>';
  try {
    const data = await apiGet('/api/sessions?page=1&page_size=50');
    renderSessionList(data.sessions || []);
  } catch (e) {
    list.innerHTML = `<div class="empty-state">Failed to load sessions<br><span class="mono dim">${e.message}</span></div>`;
  }
}

function renderSessionList(sessions) {
  const list = $('sessionList');
  if (!sessions.length) {
    list.innerHTML = '<div class="empty-state">No sessions yet.<br>Start one with + NEW</div>';
    return;
  }
  list.innerHTML = sessions.map(s => `
    <div class="session-item ${state.activeSession === s._id ? 'active' : ''}"
         onclick="selectSession('${s._id}')">
      <div class="session-item-id">
        ${s._id}
        <span class="session-badge ${s.status}">${s.status.toUpperCase()}</span>
      </div>
      <div class="session-item-meta">
        ${s.drone_id || '—'} · ${s.sampled_frame_count || 0} frames
      </div>
    </div>
  `).join('');
}

async function selectSession(videoId) {
  state.activeSession = videoId;
  $('activeSessionLabel').textContent = videoId;
  clearPolling();

  // Reload session list to update active highlight
  const data = await apiGet('/api/sessions?page=1&page_size=50').catch(() => ({ sessions: [] }));
  renderSessionList(data.sessions || []);

  // Load data for active tab
  const activeTab = document.querySelector('.tab.active')?.dataset?.tab || 'frames';
  loadTabData(activeTab);

  // Poll for status if processing
  const session = await apiGet(`/api/sessions/${videoId}`).catch(() => null);
  if (session && ['pending', 'processing'].includes(session.status)) {
    startSessionPolling(videoId);
  }
}

function startSessionPolling(videoId) {
  state.sessionPollingInterval = setInterval(async () => {
    try {
      const s = await apiGet(`/api/sessions/${videoId}`);
      if (['processed', 'failed'].includes(s.status)) {
        clearPolling();
        loadSessions();
        loadTabData('frames');
        loadTabData('alerts');
      } else {
        // Refresh frames during processing
        if (state.activeSession === videoId) loadFrames(1);
        loadAlerts();
      }
    } catch { /* silent */ }
  }, 5000);
}

function clearPolling() {
  if (state.sessionPollingInterval) { clearInterval(state.sessionPollingInterval); state.sessionPollingInterval = null; }
  if (state.alertPollingInterval) { clearInterval(state.alertPollingInterval); state.alertPollingInterval = null; }
}

// ── New session form ───────────────────────────────────────────────
$('btnNewSession').addEventListener('click', () => {
  $('newSessionForm').classList.toggle('hidden');
});
$('btnCancelForm').addEventListener('click', () => {
  $('newSessionForm').classList.add('hidden');
});
$('btnIngest').addEventListener('click', async () => {
  const videoId = $('fVideoId').value.trim();
  const droneId = $('fDroneId').value.trim();
  const videoPath = $('fVideoPath').value.trim();
  const telPath = $('fTelPath').value.trim();

  if (!videoId || !droneId || !videoPath || !telPath) {
    showMsg('ingestMsg', 'All fields are required.', true); return;
  }

  $('btnIngest').disabled = true;
  $('btnIngest').textContent = 'STARTING…';
  try {
    const r = await apiPost('/api/ingest/session', {
      video_id: videoId, drone_id: droneId,
      video_path: videoPath, telemetry_path: telPath,
    });
    showMsg('ingestMsg', `✓ ${r.message}`);
    $('newSessionForm').classList.add('hidden');
    await loadSessions();
    selectSession(videoId);
  } catch (e) {
    showMsg('ingestMsg', e.message, true);
  } finally {
    $('btnIngest').disabled = false;
    $('btnIngest').textContent = '▶ START INGESTION';
  }
});

function showMsg(id, text, isErr = false) {
  const el = $(id);
  el.textContent = text;
  el.className = `msg${isErr ? ' err' : ''}`;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 8000);
}

// ── Tabs ───────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    $(`tab-${tab.dataset.tab}`).classList.add('active');
    loadTabData(tab.dataset.tab);
  });
});

function loadTabData(tab) {
  if (!state.activeSession) return;
  if (tab === 'frames') loadFrames(state.framesPage);
  else if (tab === 'alerts') loadAlerts();
  else if (tab === 'summary') { /* manual trigger */ }
  else if (tab === 'ask') { /* user triggered */ }
  else if (tab === 'search') { /* user triggered */ }
}

// ── Frames ─────────────────────────────────────────────────────────
async function loadFrames(page = 1) {
  if (!state.activeSession) return;
  state.framesPage = page;
  const grid = $('framesGrid');
  grid.innerHTML = '<div class="loader" style="grid-column:1/-1"></div>';

  try {
    const data = await apiGet(`/api/sessions/${state.activeSession}/frames?page=${page}&page_size=24`);
    state.framesTotal = data.total;
    $('framesTotal').textContent = `${data.total} frames indexed`;
    renderFrameGrid(grid, data.frames);
    renderPagination('framesPager', page, Math.ceil(data.total / 24), loadFrames);
  } catch (e) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">${e.message}</div>`;
  }
}

function renderFrameGrid(container, frames) {
  if (!frames.length) {
    container.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No frames yet. Ingestion may still be running.</div>';
    return;
  }
  container.innerHTML = frames.map(f => buildFrameCard(f)).join('');
  container.querySelectorAll('.frame-card').forEach((card, i) => {
    card.addEventListener('click', () => showFrameDetail(frames[i]));
  });
}

function buildFrameCard(f) {
  const ctx = f.llm_context || {};
  const tel = f.telemetry || {};
  const risks = ctx.risk_signals || [];
  const hasRisk = risks.length > 0;
  const imgSrc = f.frame_path ? `/storage/${f.frame_path.replace(/^storage\//, '')}` : null;

  return `
    <div class="frame-card ${hasRisk ? 'has-risk' : ''}">
      ${hasRisk ? '<div class="risk-tag">⚠ RISK</div>' : ''}
      ${imgSrc
        ? `<img class="frame-thumb" src="${imgSrc}" alt="frame ${f.frame_no}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">`
        : ''}
      <div class="frame-thumb-placeholder" ${imgSrc ? 'style="display:none"' : ''}>🎞</div>
      <div class="frame-info">
        <div class="frame-no">F${String(f.frame_no || '?').padStart(4, '0')}</div>
        <div class="frame-zone">${tel.zone_name || '—'}</div>
        <div class="frame-summary">${ctx.scene_summary || '…'}</div>
      </div>
    </div>
  `;
}

// ── Alerts ─────────────────────────────────────────────────────────
async function loadAlerts() {
  if (!state.activeSession) return;
  const list = $('alertsList');
  try {
    const data = await apiGet(`/api/sessions/${state.activeSession}/alerts`);
    const alerts = data.alerts || [];
    updateAlertBadge(alerts.filter(a => a.status === 'new').length);
    if (!alerts.length) {
      list.innerHTML = '<div class="empty-state">No alerts raised for this session.</div>';
      return;
    }
    list.innerHTML = alerts.map(a => buildAlertCard(a)).join('');
    list.querySelectorAll('[data-ack]').forEach(btn => {
      btn.addEventListener('click', () => ackAlert(btn.dataset.ack, btn.dataset.vid, 'acknowledged'));
    });
    list.querySelectorAll('[data-resolve]').forEach(btn => {
      btn.addEventListener('click', () => ackAlert(btn.dataset.resolve, btn.dataset.vid, 'resolved'));
    });
  } catch (e) {
    list.innerHTML = `<div class="empty-state">${e.message}</div>`;
  }
}

function buildAlertCard(a) {
  const ts = a.trigger_timestamp ? a.trigger_timestamp.slice(0, 19).replace('T', ' ') : '—';
  return `
    <div class="alert-card ${a.severity}">
      <div class="alert-header">
        <span class="alert-type">${(a.alert_type || 'UNKNOWN').replace(/_/g, ' ').toUpperCase()}</span>
        <span class="alert-sev ${a.severity}">${(a.severity || '').toUpperCase()}</span>
        <span class="tag">${a.status || 'new'}</span>
      </div>
      <div class="alert-title">${a.title || 'Security Alert'}</div>
      <div class="alert-desc">${a.description || ''}</div>
      <div class="alert-meta">
        <span>🕐 ${ts}</span>
        <span>📍 ${a.zone_name || '—'}</span>
        <span>🎞 ${(a.frame_ids || []).length} frames</span>
      </div>
      ${a.status === 'new' ? `
        <div class="alert-actions">
          <button class="btn btn--sm" data-ack="${a._id}" data-vid="${a.video_id}">ACKNOWLEDGE</button>
          <button class="btn btn--sm btn--ghost" data-resolve="${a._id}" data-vid="${a.video_id}">RESOLVE</button>
        </div>` : ''}
    </div>
  `;
}

async function ackAlert(alertId, videoId, newStatus) {
  try {
    await apiPatch(`/api/sessions/${videoId}/alerts/${alertId}/status?status=${newStatus}`);
    loadAlerts();
  } catch (e) {
    console.error('Alert update failed:', e);
  }
}

function updateAlertBadge(count) {
  const badge = $('alertBadge');
  badge.textContent = count;
  badge.className = `badge${count === 0 ? ' zero' : ''}`;
}

// ── Search ─────────────────────────────────────────────────────────
$('btnSearch').addEventListener('click', runSearch);
$('searchZone').addEventListener('keydown', e => e.key === 'Enter' && runSearch());
$('searchObject').addEventListener('keydown', e => e.key === 'Enter' && runSearch());

async function runSearch() {
  const zone = $('searchZone').value.trim();
  const obj = $('searchObject').value.trim();
  const act = $('searchActivity').value.trim();
  const grid = $('searchResults');
  const vid = state.activeSession;

  const params = new URLSearchParams();
  if (vid) params.set('video_id', vid);
  if (zone) params.set('zone', zone);
  if (obj) params.set('object_kw', obj);
  if (act) params.set('activity_kw', act);

  grid.innerHTML = '<div class="loader" style="grid-column:1/-1"></div>';
  try {
    const data = await apiGet(`/api/frames/search?${params}`);
    $('searchTotal').textContent = `${data.total} results`;
    renderFrameGrid(grid, data.frames);
  } catch (e) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">${e.message}</div>`;
  }
}

// ── Summary ────────────────────────────────────────────────────────
$('btnGenSummary').addEventListener('click', async () => {
  if (!state.activeSession) return alert('Select a session first.');
  $('btnGenSummary').disabled = true;
  $('btnGenSummary').textContent = '⟳ GENERATING…';
  $('summaryText').textContent = '';
  $('summaryStats').classList.add('hidden');
  try {
    const data = await apiPost(`/api/sessions/${state.activeSession}/summary`, {});
    $('summaryStats').innerHTML = `
      <div class="stat-box"><div class="stat-value">${data.frame_count}</div><div class="stat-label">FRAMES</div></div>
      <div class="stat-box"><div class="stat-value">${data.alert_count}</div><div class="stat-label">ALERTS</div></div>
      <div class="stat-box"><div class="stat-value">${(data.zones_visited || []).length}</div><div class="stat-label">ZONES</div></div>
    `;
    $('summaryStats').classList.remove('hidden');
    $('summaryText').textContent = data.summary_text || 'No summary available.';
  } catch (e) {
    $('summaryText').textContent = `Error: ${e.message}`;
  } finally {
    $('btnGenSummary').disabled = false;
    $('btnGenSummary').textContent = '⟳ GENERATE SUMMARY';
  }
});

// ── QA ─────────────────────────────────────────────────────────────
$('btnAsk').addEventListener('click', runQA);
$('qaInput').addEventListener('keydown', e => e.key === 'Enter' && runQA());

async function runQA() {
  const q = $('qaInput').value.trim();
  if (!q) return;
  if (!state.activeSession) return alert('Select a session first.');

  $('btnAsk').disabled = true;
  $('btnAsk').textContent = '…';
  $('qaAnswer').classList.add('hidden');
  $('qaFrames').innerHTML = '<div class="loader" style="grid-column:1/-1"></div>';

  try {
    const data = await apiPost(`/api/sessions/${state.activeSession}/ask`, { question: q });
    $('qaAnswer').innerHTML = `
      <div class="qa-answer-label">AI ANSWER · ${data.frame_count} supporting frames</div>
      <div>${data.answer}</div>
    `;
    $('qaAnswer').classList.remove('hidden');
    renderFrameGrid($('qaFrames'), data.supporting_frames || []);
  } catch (e) {
    $('qaFrames').innerHTML = `<div class="empty-state" style="grid-column:1/-1">${e.message}</div>`;
  } finally {
    $('btnAsk').disabled = false;
    $('btnAsk').textContent = 'ASK';
  }
}

// ── Frame detail panel ─────────────────────────────────────────────
function showFrameDetail(f) {
  const panel = $('detailContent');
  const ctx = f.llm_context || {};
  const tel = f.telemetry || {};
  const entities = f.normalized_entities || [];
  const imgSrc = f.frame_path ? `/storage/${f.frame_path.replace(/^storage\//, '')}` : null;
  const ts = (f.video_timestamp || '').slice(0, 19).replace('T', ' ');

  const tagsHtml = arr => arr.map(t => `<span class="tag">${t}</span>`).join('');
  const riskHtml = arr => arr.map(t => `<span class="tag risk">⚠ ${t}</span>`).join('');
  const entityHtml = entities.map(e => `<span class="tag entity">${e.label} (${e.entity_type})</span>`).join('');

  panel.innerHTML = `
    <div class="detail-content">
      ${imgSrc
        ? `<img class="detail-img" src="${imgSrc}" alt="frame ${f.frame_no}"
              onerror="this.style.display='none';this.nextSibling.style.display='flex'">`
        : ''}
      <div class="detail-img-placeholder" ${imgSrc ? 'style="display:none"' : ''}>🎞</div>

      <div class="detail-section">
        <div class="detail-section-title">FRAME INFO</div>
        <div class="detail-kv"><span class="detail-k">frame_no</span><span class="detail-v mono">${f.frame_no || '—'}</span></div>
        <div class="detail-kv"><span class="detail-k">timestamp</span><span class="detail-v mono">${ts}</span></div>
        <div class="detail-kv"><span class="detail-k">zone</span><span class="detail-v">${tel.zone_name || '—'}</span></div>
        <div class="detail-kv"><span class="detail-k">gps</span><span class="detail-v mono">${tel.latitude || 0}, ${tel.longitude || 0}</span></div>
        <div class="detail-kv"><span class="detail-k">altitude</span><span class="detail-v mono">${tel.altitude || 0}m</span></div>
      </div>

      <div class="detail-section">
        <div class="detail-section-title">SCENE ANALYSIS</div>
        <div class="detail-kv"><span class="detail-k">summary</span><span class="detail-v">${ctx.scene_summary || '—'}</span></div>
        <div class="detail-kv"><span class="detail-k">people</span><span class="detail-v mono">${ctx.people_count || 0}</span></div>
        <div class="detail-kv"><span class="detail-k">vehicles</span><span class="detail-v mono">${ctx.vehicles_count || 0}</span></div>
      </div>

      ${(ctx.objects_detected || []).length ? `
        <div class="detail-section">
          <div class="detail-section-title">OBJECTS DETECTED</div>
          ${tagsHtml(ctx.objects_detected)}
        </div>` : ''}

      ${(ctx.activity_detected || []).length ? `
        <div class="detail-section">
          <div class="detail-section-title">ACTIVITIES</div>
          ${tagsHtml(ctx.activity_detected)}
        </div>` : ''}

      ${(ctx.risk_signals || []).length ? `
        <div class="detail-section">
          <div class="detail-section-title">⚠ RISK SIGNALS</div>
          ${riskHtml(ctx.risk_signals)}
        </div>` : ''}

      ${entities.length ? `
        <div class="detail-section">
          <div class="detail-section-title">ENTITIES</div>
          ${entityHtml}
        </div>` : ''}

      ${ctx.confidence_notes ? `
        <div class="detail-section">
          <div class="detail-section-title">CONFIDENCE NOTES</div>
          <div class="detail-v dim" style="font-size:11px">${ctx.confidence_notes}</div>
        </div>` : ''}
    </div>
  `;
}

$('btnCloseDetail').addEventListener('click', () => {
  $('detailContent').innerHTML = '<div class="detail-empty">Select a frame to inspect</div>';
});

// ── Pagination helper ──────────────────────────────────────────────
function renderPagination(containerId, currentPage, totalPages, onPageChange) {
  const pager = $(containerId);
  if (totalPages <= 1) { pager.innerHTML = ''; return; }

  const pages = [];
  for (let p = Math.max(1, currentPage - 2); p <= Math.min(totalPages, currentPage + 2); p++) {
    pages.push(p);
  }

  pager.innerHTML = [
    currentPage > 1 ? `<button class="page-btn" data-p="${currentPage - 1}">‹</button>` : '',
    ...pages.map(p => `<button class="page-btn ${p === currentPage ? 'active' : ''}" data-p="${p}">${p}</button>`),
    currentPage < totalPages ? `<button class="page-btn" data-p="${currentPage + 1}">›</button>` : '',
  ].join('');

  pager.querySelectorAll('.page-btn').forEach(btn => {
    btn.addEventListener('click', () => onPageChange(parseInt(btn.dataset.p)));
  });
}

// ── Init ───────────────────────────────────────────────────────────
async function init() {
  startClock();
  await checkHealth();
  setInterval(checkHealth, 30000);
  await loadSessions();
}

init();

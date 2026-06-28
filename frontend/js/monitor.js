const API_BASE = '/api/v1';
const EVAL_REPORT_PATH = '/eval-data/reports/retrieval_eval_20260620_234140.json';
const TEST_SET_PATH = '/eval-data/test_sets/fpt_2025_qa_100.jsonl';

let activeView = 'monitoring';
let testCases = [];

const formatMs = (value) => {
  const number = Number(value || 0);
  if (number >= 1000) return `${(number / 1000).toFixed(2)}s`;
  return `${number.toFixed(0)}ms`;
};

const formatUsd = (value) => {
  const number = Number(value || 0);
  if (!number) return '$0';
  return `$${number.toFixed(6)}`;
};

const formatPercent = (value) => `${Number(value || 0).toFixed(1)}%`;

const shortText = (value, max = 150) => {
  const text = String(value || '');
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
};

const escapeHtml = (value) => String(value ?? '')
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#039;');

const statusClass = (status) => {
  if (status === 'success') return 'status-success';
  if (status === 'error') return 'status-error';
  return '';
};

async function fetchJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.text();
}

function renderKpis(summary) {
  const cards = [
    {
      label: 'Requests',
      value: summary.total_requests || 0,
      foot: `${summary.query_stream_count || 0} stream / ${summary.query_count || 0} query`,
    },
    {
      label: 'Success Rate',
      value: formatPercent(summary.success_rate),
      foot: `${summary.error_count || 0} errors`,
    },
    {
      label: 'P95 Latency',
      value: formatMs(summary.p95_latency_ms),
      foot: `avg ${formatMs(summary.avg_latency_ms)} / p50 ${formatMs(summary.p50_latency_ms)}`,
    },
    {
      label: 'Avg Cost',
      value: formatUsd(summary.avg_cost_usd),
      foot: `total ${formatUsd(summary.total_cost_usd)}`,
    },
    {
      label: 'Avg Tokens',
      value: Math.round(summary.avg_tokens || 0).toLocaleString(),
      foot: 'generation tokens/request',
    },
    {
      label: 'Cache Hit Rate',
      value: formatPercent(summary.cache_hit_rate),
      foot: 'observed non-skipped cache checks',
    },
    {
      label: 'Image Grounding',
      value: formatPercent(summary.image_grounding_rate),
      foot: `${summary.image_grounded_count || 0}/${summary.image_triggered_count || 0} image-triggered requests`,
    },
    {
      label: 'Max Latency',
      value: formatMs(summary.max_latency_ms),
      foot: 'slowest request in selected window',
    },
  ];

  document.getElementById('kpi-grid').innerHTML = cards.map(card => `
    <article class="kpi-card">
      <div class="kpi-label">${card.label}</div>
      <div class="kpi-value">${card.value}</div>
      <div class="kpi-foot">${card.foot}</div>
    </article>
  `).join('');
}

function renderStepChart(steps) {
  const container = document.getElementById('step-chart');
  const topSteps = (steps || []).slice(0, 10);
  const max = Math.max(...topSteps.map(step => Number(step.avg_ms || 0)), 1);

  if (!topSteps.length) {
    container.innerHTML = '<div class="subtle">No step metrics yet.</div>';
    return;
  }

  container.innerHTML = topSteps.map(step => {
    const width = Math.max(2, (Number(step.avg_ms || 0) / max) * 100);
    return `
      <div class="bar-row">
        <div class="bar-name" title="${step.step}">${step.step}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <div class="bar-value">${formatMs(step.avg_ms)}</div>
      </div>
    `;
  }).join('');
}

function renderLatest(latest) {
  const latestStatus = document.getElementById('latest-status');
  const container = document.getElementById('latest-request');
  if (!latest) {
    latestStatus.textContent = 'empty';
    latestStatus.className = 'status-pill';
    container.innerHTML = '<div class="subtle">No requests logged yet.</div>';
    return;
  }

  latestStatus.textContent = latest.status || 'unknown';
  latestStatus.className = `status-pill ${statusClass(latest.status)}`;

  const paths = latest.selected_image_paths || [];
  container.innerHTML = `
    <div class="latest-question">${shortText(latest.question, 260)}</div>
    <div class="meta-grid">
      <div class="meta-item"><div class="meta-label">Latency</div><div class="meta-value">${formatMs(latest.total_latency_ms)}</div></div>
      <div class="meta-item"><div class="meta-label">Slowest Step</div><div class="meta-value">${latest.slowest_step || '-'}</div></div>
      <div class="meta-item"><div class="meta-label">Images</div><div class="meta-value">${latest.selected_image_count || 0}</div></div>
      <div class="meta-item"><div class="meta-label">Cost</div><div class="meta-value">${formatUsd(latest.estimated_usd)}</div></div>
      <div class="meta-item"><div class="meta-label">Cache</div><div class="meta-value">${latest.cache_skipped ? 'skipped' : latest.cache_hit === true ? 'hit' : 'miss'}</div></div>
      <div class="meta-item"><div class="meta-label">Mode</div><div class="meta-value">${latest.mode || '-'}</div></div>
    </div>
    ${paths.length ? `<div class="image-path">${paths.map(path => shortText(path, 120)).join('<br>')}</div>` : ''}
  `;
}

function renderRecent(rows) {
  const body = document.getElementById('recent-body');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="subtle">No requests logged yet.</td></tr>';
    return;
  }

  body.innerHTML = rows.map(row => {
    const paths = row.selected_image_paths || [];
    return `
      <tr>
        <td class="num">${row.started_at || '-'}</td>
        <td>${row.mode || '-'}</td>
        <td><span class="status-pill ${statusClass(row.status)}">${row.status || '-'}</span></td>
        <td class="num">${formatMs(row.total_latency_ms)}</td>
        <td>${row.slowest_step || '-'}<div class="subtle">${formatMs(row.slowest_step_ms)}</div></td>
        <td class="num">${row.selected_image_count || 0}<div class="subtle">${row.context_selection_image_hits || 0} in context</div></td>
        <td class="num">${formatUsd(row.estimated_usd)}</td>
        <td class="question-cell">
          ${shortText(row.question, 220)}
          ${paths.length ? `<div class="image-path">${shortText(paths[0], 120)}</div>` : ''}
        </td>
      </tr>
    `;
  }).join('');
}

async function loadMonitor() {
  const limit = Number(document.getElementById('window-size').value || 200);
  const [summaryData, recentData, latestData] = await Promise.all([
    fetchJson(`${API_BASE}/observability/summary?limit=${limit}`),
    fetchJson(`${API_BASE}/observability/recent?limit=50`),
    fetchJson(`${API_BASE}/observability/latest`),
  ]);

  renderKpis(summaryData.summary || {});
  renderStepChart(summaryData.step_breakdown || []);
  renderLatest(latestData.summary);
  renderRecent(recentData.requests || []);
}

document.getElementById('refresh-btn').addEventListener('click', () => {
  loadActiveView();
});

document.getElementById('window-size').addEventListener('change', () => {
  loadMonitor().catch(showError);
});

function showError(error) {
  document.getElementById('latest-status').textContent = 'error';
  document.getElementById('latest-status').className = 'status-pill status-error';
  document.getElementById('latest-request').innerHTML = `<div class="latest-question">${error.message}</div>`;
}

function renderMetricBars(containerId, items) {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = '<div class="subtle">Chưa có dữ liệu.</div>';
    return;
  }

  container.innerHTML = items.map(item => `
    <div class="bar-row">
      <div class="bar-name">${escapeHtml(item.label)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, item.value * 100)}%"></div></div>
      <div class="bar-value">${(item.value * 100).toFixed(1)}%</div>
    </div>
  `).join('');
}

function renderEvaluation(report) {
  const summary = report.summary || {};
  const total = Number(summary.total || 0);
  const cards = [
    { label: 'Total cases', value: total, foot: 'retrieval baseline' },
    { label: 'Passed', value: summary.passed || 0, foot: `${summary.failed || 0} failed` },
    { label: 'Pass rate', value: formatPercent((summary.pass_rate || 0) * 100), foot: 'all categories' },
    { label: 'Page hit', value: formatPercent((summary.check_rates?.page_hit || 0) * 100), foot: 'expected page retrieved' },
  ];

  document.getElementById('eval-kpis').innerHTML = cards.map(card => `
    <article class="kpi-card">
      <div class="kpi-label">${card.label}</div>
      <div class="kpi-value">${card.value}</div>
      <div class="kpi-foot">${card.foot}</div>
    </article>
  `).join('');

  document.getElementById('eval-run-time').textContent = report.created_at
    ? `run ${report.created_at}`
    : '';

  const categories = Object.entries(summary.by_category || {}).map(([label, value]) => ({
    label,
    value: Number(value.pass_rate || 0),
  }));
  renderMetricBars('eval-category-chart', categories);

  const checks = Object.entries(summary.check_rates || {}).map(([label, value]) => ({
    label: label.replaceAll('_', ' '),
    value: Number(value || 0),
  }));
  renderMetricBars('eval-check-chart', checks);

  const failures = (report.results || []).filter(item => !item.passed);
  document.getElementById('eval-failure-count').textContent = failures.length;
  document.getElementById('eval-failures').innerHTML = failures.length
    ? failures.map(renderFailureCard).join('')
    : '<div class="subtle">Không có case thất bại.</div>';
}

function renderFailureCard(item) {
  const failedChecks = Object.entries(item.checks || {})
    .filter(([, passed]) => !passed)
    .map(([name]) => name.replaceAll('_', ' '));
  const expected = item.expected || {};
  const topResult = (item.top_results || [])[0];

  return `
    <details class="case-card">
      <summary>
        <span class="case-id">${escapeHtml(item.id)}</span>
        <span class="case-question">${escapeHtml(item.question)}</span>
        <span class="case-tags">
          <span class="tag">${escapeHtml(item.category)}</span>
          ${failedChecks.map(check => `<span class="tag status-error">${escapeHtml(check)}</span>`).join('')}
        </span>
      </summary>
      <div class="case-details">
        <div class="detail-block">
          <div class="detail-title">Expected</div>
          <div class="detail-content">
            Pages: ${escapeHtml((expected.pages || []).join(', ') || '—')}<br>
            Types: ${escapeHtml((expected.chunk_types || []).join(', ') || '—')}<br>
            Numbers: ${escapeHtml((expected.numbers || []).join(', ') || '—')}
          </div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Top retrieved result</div>
          <div class="detail-content">
            ${topResult
              ? `Page ${escapeHtml(topResult.page ?? '—')} · ${escapeHtml(topResult.chunk_type || '—')} · score ${escapeHtml(topResult.score ?? '—')}<br>${escapeHtml(topResult.preview || '')}`
              : 'Không có kết quả retrieval.'}
          </div>
        </div>
      </div>
    </details>
  `;
}

async function loadEvaluation() {
  renderEvaluation(await fetchJson(EVAL_REPORT_PATH));
}

function parseJsonl(text) {
  return text.split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`JSONL không hợp lệ tại dòng ${index + 1}: ${error.message}`);
      }
    });
}

function renderTestSummary(rows) {
  const categoryCount = rows.reduce((counts, row) => {
    counts[row.category] = (counts[row.category] || 0) + 1;
    return counts;
  }, {});
  const cards = [
    { label: 'Total', value: rows.length, foot: 'test cases' },
    { label: 'Text', value: categoryCount.text || 0, foot: 'text evidence' },
    { label: 'Table', value: categoryCount.table || 0, foot: 'table evidence' },
    { label: 'Image', value: categoryCount.image || 0, foot: 'chart evidence' },
    { label: 'Mixed / No answer', value: `${categoryCount.mixed || 0} / ${categoryCount.unanswerable || 0}`, foot: 'cross-source / refusal' },
  ];

  document.getElementById('test-kpis').innerHTML = cards.map(card => `
    <article class="kpi-card">
      <div class="kpi-label">${card.label}</div>
      <div class="kpi-value">${card.value}</div>
      <div class="kpi-foot">${card.foot}</div>
    </article>
  `).join('');
}

function populateFilter(id, values) {
  const select = document.getElementById(id);
  const current = select.value;
  const firstOption = select.options[0].outerHTML;
  select.innerHTML = firstOption + values
    .map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
    .join('');
  select.value = current;
}

function renderTestCard(item) {
  return `
    <details class="case-card">
      <summary>
        <span class="case-id">${escapeHtml(item.id)}</span>
        <span class="case-question">${escapeHtml(item.question)}</span>
        <span class="case-tags">
          <span class="tag">${escapeHtml(item.category)}</span>
          <span class="tag">${escapeHtml(item.difficulty)}</span>
          <span class="tag ${item.answerable ? 'status-success' : 'status-warn'}">${item.answerable ? 'answerable' : 'no answer'}</span>
          <span class="tag">page ${escapeHtml((item.expected_pages || []).join(', ') || '—')}</span>
        </span>
      </summary>
      <div class="case-details">
        <div class="detail-block">
          <div class="detail-title">Ground truth answer</div>
          <div class="detail-content">${escapeHtml(item.ground_truth_answer)}</div>
        </div>
        <div class="detail-block">
          <div class="detail-title">Evidence</div>
          <div class="detail-content">${escapeHtml(item.evidence)}</div>
        </div>
      </div>
    </details>
  `;
}

function filterTestCases() {
  const query = document.getElementById('test-search').value.trim().toLowerCase();
  const category = document.getElementById('test-category').value;
  const difficulty = document.getElementById('test-difficulty').value;
  const answerable = document.getElementById('test-answerable').value;

  const visible = testCases.filter(item => {
    const searchable = `${item.id} ${item.question} ${item.ground_truth_answer} ${item.evidence}`.toLowerCase();
    return (!query || searchable.includes(query))
      && (!category || item.category === category)
      && (!difficulty || item.difficulty === difficulty)
      && (!answerable || String(item.answerable) === answerable);
  });

  document.getElementById('test-visible-count').textContent = `${visible.length}/${testCases.length} câu`;
  document.getElementById('test-cases').innerHTML = visible.length
    ? visible.map(renderTestCard).join('')
    : '<div class="subtle">Không tìm thấy câu phù hợp.</div>';
}

async function loadTestSet() {
  if (!testCases.length) {
    testCases = parseJsonl(await fetchText(TEST_SET_PATH));
    renderTestSummary(testCases);
    populateFilter('test-category', [...new Set(testCases.map(item => item.category))].sort());
    populateFilter('test-difficulty', [...new Set(testCases.map(item => item.difficulty))].sort());
  }
  filterTestCases();
}

async function loadActiveView() {
  try {
    if (activeView === 'monitoring') await loadMonitor();
    if (activeView === 'evaluation') await loadEvaluation();
    if (activeView === 'test-set') await loadTestSet();
  } catch (error) {
    if (activeView === 'monitoring') showError(error);
    else {
      const target = activeView === 'evaluation' ? 'eval-failures' : 'test-cases';
      document.getElementById(target).innerHTML = `<div class="latest-question">${escapeHtml(error.message)}</div>`;
    }
  }
}

function switchView(view) {
  activeView = view;
  document.querySelectorAll('.workspace-view').forEach(element => element.classList.add('hidden'));
  document.querySelectorAll('.workspace-tab').forEach(element => element.classList.remove('active'));
  document.getElementById(`view-${view}`).classList.remove('hidden');
  document.querySelector(`[data-view="${view}"]`).classList.add('active');
  window.location.hash = view;
  loadActiveView();
}

document.querySelectorAll('.workspace-tab').forEach(tab => {
  tab.addEventListener('click', () => switchView(tab.dataset.view));
});

['test-search', 'test-category', 'test-difficulty', 'test-answerable'].forEach(id => {
  document.getElementById(id).addEventListener(id === 'test-search' ? 'input' : 'change', filterTestCases);
});

const initialView = window.location.hash.slice(1);
const validViews = ['monitoring', 'evaluation', 'test-set', 'traces', 'documents'];
switchView(validViews.includes(initialView) ? initialView : 'monitoring');

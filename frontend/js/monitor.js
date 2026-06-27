const API_BASE = '/api/v1';

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
  loadMonitor().catch(showError);
});

document.getElementById('window-size').addEventListener('change', () => {
  loadMonitor().catch(showError);
});

function showError(error) {
  document.getElementById('latest-status').textContent = 'error';
  document.getElementById('latest-status').className = 'status-pill status-error';
  document.getElementById('latest-request').innerHTML = `<div class="latest-question">${error.message}</div>`;
}

loadMonitor().catch(showError);

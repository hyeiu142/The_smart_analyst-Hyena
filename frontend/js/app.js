/* ═══════════════════════════════════════ APP MAIN ══ */

document.addEventListener('DOMContentLoaded', () => {

  // Init modules
  Upload.init();
  Chat.init();

  // Load document list
  DocList.refresh();

  // Theme toggle
  const btnTheme = document.getElementById('btn-theme');
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.body.className = savedTheme;
  btnTheme.textContent = savedTheme === 'dark' ? '☀️' : '🌙';

  btnTheme.addEventListener('click', () => {
    const isDark = document.body.classList.toggle('dark');
    document.body.classList.toggle('light', !isDark);
    btnTheme.textContent = isDark ? '☀️' : '🌙';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });

  // Health check
  checkSystemHealth();

  // Auto-refresh docs every 30s (pick up processing docs)
  setInterval(() => DocList.refresh(), 30000);
});

async function checkSystemHealth() {
  const dot = document.getElementById('system-status');
  dot.className = 'status-dot dot-checking';
  dot.title = 'Checking...';

  const health = await Api.checkHealth();
  if (!health) {
    dot.className = 'status-dot dot-error';
    dot.title = 'Cannot reach backend';
    showToast('Cannot connect to backend API', 'error');
    return;
  }

  const qdrantOk = health.qdrant?.status === 'connected';
  const redisOk  = health.redis?.status === 'connected';

  if (qdrantOk && redisOk) {
    dot.className = 'status-dot dot-ok';
    dot.title = `API ✓ | Qdrant ✓ (${(health.qdrant.collections || []).join(', ')}) | Redis ✓`;
  } else {
    dot.className = 'status-dot dot-error';
    const issues = [];
    if (!qdrantOk) issues.push(`Qdrant: ${health.qdrant?.message || 'error'}`);
    if (!redisOk)  issues.push(`Redis: ${health.redis?.message || 'error'}`);
    dot.title = issues.join(' | ');
    showToast('Service issue: ' + issues.join(', '), 'error');
  }
}

/* ═══════════════════════════════════════ UTILS ══ */

/** Hiển thị toast notification */
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

/** Auto-resize textarea theo nội dung */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/** Escape HTML để tránh XSS */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/** Render Markdown đơn giản (bold, italic, code, table, list) */
function renderMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Code blocks
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Tables (| col | col |)
  const tableRegex = /((?:\|.*\|\n?)+)/g;
  html = html.replace(tableRegex, (match) => {
    const rows = match.trim().split('\n').filter(r => r.trim());
    if (rows.length < 2) return match;
    let tableHtml = '<table>';
    rows.forEach((row, i) => {
      if (/^\|[-|\s:]+\|$/.test(row.trim())) return; // separator row
      const cells = row.split('|').filter((_, j, arr) => j > 0 && j < arr.length - 1);
      const tag = i === 0 ? 'th' : 'td';
      tableHtml += '<tr>' + cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('') + '</tr>';
    });
    tableHtml += '</table>';
    return tableHtml;
  });

  // Lists
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>')
             .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Line breaks
  html = html.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br/>');
  html = '<p>' + html + '</p>';
  html = html.replace(/<p><\/p>/g, '');

  return html;
}

/** Scroll element to bottom */
function scrollToBottom(el) {
  el.scrollTop = el.scrollHeight;
}

/** Debounce */
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/** Format số (1234567 → 1,234,567) */
function formatNumber(n) {
  return Number(n).toLocaleString('vi-VN');
}

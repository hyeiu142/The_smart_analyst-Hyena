/* ═══════════════════════════════════════ CHAT MODULE ══ */

const Chat = (() => {
  let isLoading = false;
  let pendingSources = [];

  function init() {
    const input  = document.getElementById('chat-input');
    const btnSend = document.getElementById('btn-send');

    // Auto-resize
    input.addEventListener('input', () => autoResize(input));

    // Send on Enter (not Shift+Enter)
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    btnSend.addEventListener('click', send);

    // Close source viewer
    document.getElementById('btn-close-sources').addEventListener('click', () => {
      document.getElementById('source-viewer').classList.add('hidden');
    });

    // Example queries
    document.querySelectorAll('.example-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        input.value = btn.dataset.q;
        autoResize(input);
        send();
      });
    });
  }

  async function send() {
    if (isLoading) return;

    const input    = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;

    const company = document.getElementById('chat-company').value;
    const year    = document.getElementById('chat-year').value;

    // Clear input
    input.value = '';
    autoResize(input);

    // Remove welcome message
    const welcome = document.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    // Append user message
    appendMessage('user', question);

    // Show typing
    const typingId = appendTyping();
    setLoading(true);

    try {
      // Use streaming
      const assistantMsgEl = createAssistantBubble(typingId);
      let fullText = '';

      await Api.queryStream(
        question,
        { company: company || null, year: year || null },
        // onToken
        (token) => {
          fullText += token;
          assistantMsgEl.querySelector('.msg-bubble').innerHTML = renderMarkdown(fullText);
          scrollToBottom(document.getElementById('chat-messages'));
        },
        // onDone
        () => {
          // Fetch full result for citations via non-streaming query
          fetchCitations(question, company, year, assistantMsgEl);
          setLoading(false);
        },
        // onError
        async (err) => {
          console.warn('Stream failed, falling back to regular query:', err);
          // Fallback to regular query
          try {
            const result = await Api.query(question, { company: company || null, year: year || null });
            assistantMsgEl.querySelector('.msg-bubble').innerHTML = renderMarkdown(result.answer);
            if (result.sources && result.sources.length > 0) {
              pendingSources = result.sources;
              renderCitations(assistantMsgEl, result.sources);
            }
          } catch (e2) {
            assistantMsgEl.querySelector('.msg-bubble').textContent = `❌ Lỗi: ${e2.message}`;
          }
          setLoading(false);
        }
      );
    } catch (e) {
      removeTyping(typingId);
      appendMessage('assistant', `❌ Lỗi: ${e.message}`, []);
      setLoading(false);
    }
  }

  async function fetchCitations(question, company, year, msgEl) {
    try {
      const result = await Api.query(question, { company: company || null, year: year || null });
      if (result.sources && result.sources.length > 0) {
        pendingSources = result.sources;
        renderCitations(msgEl, result.sources);
      }
    } catch (e) {
      console.warn('Failed to fetch citations:', e);
    }
  }

  function appendMessage(role, content, sources = []) {
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `
      <div class="msg-avatar">${role === 'user' ? 'U' : '🐍'}</div>
      <div class="msg-body">
        <div class="msg-bubble">${role === 'user' ? escapeHtml(content) : renderMarkdown(content)}</div>
        ${sources.length ? renderCitationsHtml(sources) : ''}
      </div>
    `;
    messages.appendChild(div);
    scrollToBottom(messages);
    return div;
  }

  function appendTyping() {
    const id = 'typing-' + Date.now();
    const messages = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.id = id;
    div.innerHTML = `
      <div class="msg-avatar">🐍</div>
      <div class="msg-body">
        <div class="msg-bubble">
          <div class="typing-dots"><span></span><span></span><span></span></div>
        </div>
      </div>
    `;
    messages.appendChild(div);
    scrollToBottom(messages);
    return id;
  }

  function createAssistantBubble(typingId) {
    const typing = document.getElementById(typingId);
    if (typing) {
      typing.querySelector('.msg-bubble').innerHTML = '';
      return typing;
    }
    return appendMessage('assistant', '');
  }

  function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function renderCitations(msgEl, sources) {
    const body = msgEl.querySelector('.msg-body');
    const existing = body.querySelector('.msg-citations');
    if (existing) existing.remove();
    const div = document.createElement('div');
    div.innerHTML = renderCitationsHtml(sources);
    body.appendChild(div.firstElementChild);
    attachCitationClickHandlers(body);
  }

  function renderCitationsHtml(sources) {
    if (!sources || !sources.length) return '';
    const tags = sources.map(s => {
      const icon = { table: '📈', image: '📉', text: '📝' }[s.type] || '📌';
      return `<button class="citation-tag type-${s.type}" data-index="${s.index}"
        title="Page ${s.page} | Score: ${s.score}">
        ${icon} ${s.type === 'image' ? 'Chart' : s.type.charAt(0).toUpperCase() + s.type.slice(1)} · p.${s.page}
      </button>`;
    }).join('');
    return `<div class="msg-citations">📎 Sources: ${tags}</div>`;
  }

  function attachCitationClickHandlers(container) {
    container.querySelectorAll('.citation-tag').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.index);
        const source = pendingSources.find(s => s.index === idx);
        if (source) showSourceViewer([source]);
      });
    });
  }

  function showSourceViewer(sources) {
    const viewer = document.getElementById('source-viewer');
    const content = document.getElementById('source-viewer-content');
    const title   = document.getElementById('source-viewer-title');

    title.textContent = `📎 Sources (${sources.length})`;
    content.innerHTML = sources.map(s => {
      const icon = { table: '📈 TABLE', image: '📉 CHART', text: '📝 TEXT' }[s.type] || '📌';
      return `
        <div class="source-card">
          <div class="source-card-header">
            <span>${icon}</span>
            <span>${escapeHtml(s.company)} · Page ${s.page}</span>
            <span style="margin-left:auto;color:var(--text3)">score: ${s.score}</span>
          </div>
          <div class="source-card-body ${s.type === 'table' ? 'table-content' : ''}">${escapeHtml(s.preview)}</div>
        </div>
      `;
    }).join('');

    viewer.classList.remove('hidden');
  }

  function setLoading(val) {
    isLoading = val;
    const btn = document.getElementById('btn-send');
    btn.disabled = val;
    document.getElementById('chat-input').disabled = val;
  }

  return { init };
})();

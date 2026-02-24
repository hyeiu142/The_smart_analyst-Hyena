/* ═══════════════════════════════════════ API CLIENT ══ */

const API_BASE = '/api/v1';

const Api = {

  /* ─── Documents ─── */

  async uploadDocument(file, company, year, quarter) {
    const form = new FormData();
    form.append('file', file);
    form.append('company', company);
    form.append('year', year);
    if (quarter) form.append('quarter', quarter);

    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST', body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async listDocuments() {
    const res = await fetch(`${API_BASE}/documents/`);
    if (!res.ok) throw new Error('Failed to load documents');
    return res.json();
  },

  async getDocumentStatus(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}/status`);
    if (!res.ok) throw new Error('Failed to get status');
    return res.json();
  },

  async deleteDocument(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete document');
    return res.json();
  },

  /* ─── Query ─── */

  async query(question, options = {}) {
    const body = {
      question,
      top_k: options.topK || 5,
      company: options.company || null,
      year: options.year ? parseInt(options.year) : null,
      quarter: options.quarter || null,
    };
    const res = await fetch(`${API_BASE}/query/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Query failed');
    }
    return res.json();
  },

  /** Streaming query — gọi callback(token) với mỗi token */
  async queryStream(question, options = {}, onToken, onDone, onError) {
    const body = {
      question,
      top_k: options.topK || 5,
      company: options.company || null,
      year: options.year ? parseInt(options.year) : null,
      quarter: options.quarter || null,
    };
    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') { onDone && onDone(); return; }
          try {
            const json = JSON.parse(data);
            if (json.token) onToken && onToken(json.token);
            if (json.error) throw new Error(json.error);
          } catch (e) { /* skip malformed */ }
        }
      }
      onDone && onDone();
    } catch (e) {
      onError && onError(e);
    }
  },

  /* ─── Health ─── */

  async checkHealth() {
    try {
      const [api, qdrant, redis] = await Promise.all([
        fetch(`${API_BASE}/health/`).then(r => r.json()),
        fetch(`${API_BASE}/health/qdrant`).then(r => r.json()),
        fetch(`${API_BASE}/health/redis`).then(r => r.json()),
      ]);
      return { api, qdrant, redis };
    } catch {
      return null;
    }
  },
};

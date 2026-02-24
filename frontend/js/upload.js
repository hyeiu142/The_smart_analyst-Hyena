/* ═══════════════════════════════════════ UPLOAD MODULE ══ */

const Upload = (() => {
  let selectedFile = null;
  let pollingIntervals = {};

  function init() {
    // Open modal buttons
    document.getElementById('btn-upload-header').addEventListener('click', openModal);
    document.getElementById('btn-upload-sidebar').addEventListener('click', openModal);

    // Close modal
    document.getElementById('btn-close-modal').addEventListener('click', closeModal);
    document.getElementById('modal-backdrop').addEventListener('click', closeModal);
    document.getElementById('btn-cancel-upload').addEventListener('click', resetForm);

    // File pick via button
    document.getElementById('file-input').addEventListener('change', e => {
      if (e.target.files[0]) selectFile(e.target.files[0]);
    });

    // Drag & Drop
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave',  () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.pdf')) selectFile(file);
      else showToast('Chỉ hỗ trợ file PDF', 'error');
    });

    // Submit
    document.getElementById('btn-submit-upload').addEventListener('click', submitUpload);
  }

  function openModal() {
    document.getElementById('upload-modal').classList.remove('hidden');
  }

  function closeModal() {
    document.getElementById('upload-modal').classList.add('hidden');
    resetForm();
  }

  function selectFile(file) {
    selectedFile = file;
    document.getElementById('drop-zone').classList.add('hidden');
    document.getElementById('upload-form').classList.remove('hidden');
    document.getElementById('selected-file').textContent = `📄 ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;

    // Auto-fill year
    const yearInput = document.getElementById('input-year');
    if (!yearInput.value) yearInput.value = new Date().getFullYear();
  }

  function resetForm() {
    selectedFile = null;
    document.getElementById('drop-zone').classList.remove('hidden');
    document.getElementById('upload-form').classList.add('hidden');
    document.getElementById('upload-progress').classList.add('hidden');
    document.getElementById('file-input').value = '';
    document.getElementById('input-company').value = '';
    document.getElementById('input-year').value = '';
    document.getElementById('input-quarter').value = '';
  }

  async function submitUpload() {
    const company = document.getElementById('input-company').value.trim();
    const year    = document.getElementById('input-year').value.trim();
    const quarter = document.getElementById('input-quarter').value;

    if (!selectedFile) return showToast('Chọn file trước', 'error');
    if (!company)       return showToast('Nhập tên công ty', 'error');
    if (!year)          return showToast('Nhập năm', 'error');

    // Show progress
    document.getElementById('upload-form').classList.add('hidden');
    document.getElementById('upload-progress').classList.remove('hidden');
    setProgress(20, 'Uploading file...');

    try {
      const result = await Api.uploadDocument(selectedFile, company, year, quarter);
      setProgress(60, 'File uploaded. Processing started...');
      document.getElementById('upload-doc-id').textContent = `doc_id: ${result.doc_id}`;

      showToast('Upload thành công! Đang xử lý...', 'success');

      // Add to UI immediately
      DocList.addOrUpdate({
        doc_id: result.doc_id,
        filename: result.filename,
        company: result.company,
        year: result.year,
        quarter: result.quarter,
        status: 'processing',
        total_chunks: 0, text_chunks: 0, table_chunks: 0, image_chunks: 0,
        created_at: new Date().toISOString(),
      });

      // Poll status
      startPolling(result.doc_id);
      setTimeout(closeModal, 2000);

    } catch (e) {
      setProgress(0, '');
      document.getElementById('upload-form').classList.remove('hidden');
      document.getElementById('upload-progress').classList.add('hidden');
      showToast(`Upload thất bại: ${e.message}`, 'error');
    }
  }

  function setProgress(pct, text) {
    document.getElementById('progress-bar').style.width = pct + '%';
    document.getElementById('upload-status-text').textContent = text;
  }

  function startPolling(docId) {
    if (pollingIntervals[docId]) return;
    let attempts = 0;
    pollingIntervals[docId] = setInterval(async () => {
      attempts++;
      if (attempts > 120) { clearInterval(pollingIntervals[docId]); return; } // 10 min timeout
      try {
        const status = await Api.getDocumentStatus(docId);
        DocList.addOrUpdate({ ...status, doc_id: docId });

        if (status.status === 'completed') {
          clearInterval(pollingIntervals[docId]);
          delete pollingIntervals[docId];
          showToast(`✅ Xử lý xong! ${status.total_chunks} chunks`, 'success');
          DocList.refresh();
        } else if (status.status === 'failed') {
          clearInterval(pollingIntervals[docId]);
          delete pollingIntervals[docId];
          showToast(`❌ Xử lý thất bại: ${status.error || 'unknown error'}`, 'error');
          DocList.refresh();
        }
      } catch (e) { /* network blip, retry */ }
    }, 5000);
  }

  return { init, startPolling };
})();

/* ═══════════════════════════════════════ DOC LIST MODULE ══ */

const DocList = (() => {
  let docs = {}; // docId → doc object

  async function refresh() {
    try {
      const list = await Api.listDocuments();
      docs = {};
      list.forEach(d => { docs[d.doc_id] = d; });
      render();
      updateFilters();
    } catch (e) {
      console.error('Failed to load docs', e);
    }
  }

  function addOrUpdate(doc) {
    docs[doc.doc_id] = { ...(docs[doc.doc_id] || {}), ...doc };
    render();
    updateFilters();
  }

  function render() {
    const container = document.getElementById('doc-list');
    const filterCompany = document.getElementById('filter-company').value;
    const filterYear    = document.getElementById('filter-year').value;

    const filtered = Object.values(docs).filter(d => {
      if (filterCompany && d.company !== filterCompany) return false;
      if (filterYear    && String(d.year) !== filterYear) return false;
      return true;
    });

    if (filtered.length === 0) {
      container.innerHTML = '<div class="empty-state">No documents found.</div>';
      return;
    }

    container.innerHTML = filtered.map(d => `
      <div class="doc-item" data-id="${d.doc_id}">
        <div class="doc-item-header">
          <span class="doc-item-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
          <span class="doc-status status-${d.status}">${statusLabel(d.status)}</span>
        </div>
        <div class="doc-item-badges">
          <span class="badge badge-company">${escapeHtml(d.company)}</span>
          <span class="badge badge-year">${d.year}</span>
          ${d.quarter ? `<span class="badge badge-quarter">${d.quarter}</span>` : ''}
          ${d.total_chunks ? `<span class="badge badge-chunks">${d.total_chunks} chunks</span>` : ''}
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.doc-item').forEach(el => {
      el.addEventListener('click', () => {
        container.querySelectorAll('.doc-item').forEach(e => e.classList.remove('active'));
        el.classList.add('active');
        const doc = docs[el.dataset.id];
        // Pre-fill chat filters
        document.getElementById('chat-company').value = doc.company;
        if (doc.year) document.getElementById('chat-year').value = doc.year;
      });
    });
  }

  function statusLabel(s) {
    return { pending: '⏳', processing: '⏳⏳', completed: '✓', failed: '✗' }[s] || s;
  }

  function updateFilters() {
    const companies = [...new Set(Object.values(docs).map(d => d.company))];
    const years     = [...new Set(Object.values(docs).map(d => String(d.year)))].sort().reverse();

    const updateSelect = (id, values) => {
      const sel = document.getElementById(id);
      const cur = sel.value;
      sel.innerHTML = `<option value="">All ${id.includes('company') ? 'companies' : 'years'}</option>`
        + values.map(v => `<option value="${v}" ${v === cur ? 'selected' : ''}>${v}</option>`).join('');
    };

    updateSelect('filter-company', companies);
    updateSelect('filter-year', years);
    updateSelect('chat-company', companies);
    updateSelect('chat-year', years);
  }

  // Filter change
  document.getElementById('filter-company').addEventListener('change', render);
  document.getElementById('filter-year').addEventListener('change', render);

  return { refresh, addOrUpdate, render };
})();

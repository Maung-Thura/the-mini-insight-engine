(() => {
  const askBtn = document.getElementById('askBtn');
  const loading = document.getElementById('loading');
  const out = document.getElementById('output');

  async function ask() {
    const q = document.getElementById('question').value.trim();
    const cat = document.getElementById('category').value.trim();
    const body = { question: q };
    if (cat) body.filters = { category: cat };

    try {
      askBtn.disabled = true;
      loading.style.display = 'inline-block';
      out.innerHTML = '';

      const res = await fetch('/qa', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) {
        const text = await res.text();
        out.innerHTML = `<p style="color:#b00">Request failed (${res.status}). ${escapeHtml(text)}</p>`;
        return;
      }
      const data = await res.json();
      if (data.error) { out.innerHTML = `<p style=\"color:#b00\">${data.error}</p>`; return; }

      const result = data.result || {};
      const metrics = data.metrics || null;
      const metricsStatus = data.metrics_status || null;

      let html = '';
      html += `<div class="answer"><strong>Answer</strong><br>${escapeHtml(result.answer || '')}</div>`;

      if (metrics) {
        html += `<div class="metrics"><strong>Metrics (RAGAS)</strong><div class="kvs">`;
        for (const [k,v] of Object.entries(metrics)) {
          html += `<div>${escapeHtml(k)}</div><div>${Number(v).toFixed(3)}</div>`;
        }
        html += `</div></div>`;
      } else {
        const msg = metricsStatus ? `Metrics not available: ${escapeHtml(metricsStatus)}` : `Metrics not available. Ensure ragas is installed and configured.`;
        html += `<div class="metrics"><em>${msg}</em></div>`;
      }

      const ctxs = result.contexts || [];
      if (ctxs.length) {
        html += `<h3>Retrieved Context</h3>`;
        for (const c of ctxs) {
          const id = (c.metadata && (c.metadata.recommendation_id || c.metadata.id)) || c.id || '';
          html += `<div class="ctx"><strong>${escapeHtml(id)}</strong><br>${escapeHtml(c.text || '')}</div>`;
        }
      }

      out.innerHTML = html;
    } catch (err) {
      out.innerHTML = `<p style=\"color:#b00\">${escapeHtml(String(err))}</p>`;
    } finally {
      askBtn.disabled = false;
      loading.style.display = 'none';
    }
  }

  function escapeHtml(str) {
    return (str || '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\'':'&#39;','"':'&quot;'}[c]));
  }

  askBtn.addEventListener('click', ask);
})();


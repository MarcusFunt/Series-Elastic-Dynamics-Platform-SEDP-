export const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const badge = (label, kind='') => `<span class="badge ${esc(kind)}">${esc(label)}</span>`;
export const sourceLink = (ctx, id, label) => {
  const item = ctx.catalog.sources.find(x => x.id === id);
  return item ? `<a class="source-pill" target="_blank" rel="noopener" href="/api/project/source/${encodeURIComponent(id)}">${esc(label || item.name)}</a>` : '';
};
export const evidenceLink = (ctx, id, label) => {
  const item = ctx.evidence.find(x => x.id === id);
  return item?.available ? `<a class="source-pill" target="_blank" rel="noopener" href="${item.sourceUrl}">${esc(label || item.name)}</a>` : '';
};
export const pageHead = (eyebrow, title, description, actions='') => `<div class="page-head"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div><div class="head-actions">${actions}</div></div>`;
export const panelHead = (title, sub='') => `<div class="panel-head"><div><h2>${esc(title)}</h2>${sub ? `<p>${esc(sub)}</p>` : ''}</div></div>`;
export const statCard = (label, value, sub, symbol='◈') => `<section class="panel stat-card"><span class="stat-accent">${symbol}</span><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div><div class="stat-sub">${esc(sub)}</div></section>`;
export const fmt = (x, digits=3) => typeof x === 'number' && Number.isFinite(x) ? x.toFixed(digits) : '—';
export const metric = (label, value, unit='') => `<span class="metric-chip"><b>${esc(value)}</b>${esc(label)}${unit ? ` · ${esc(unit)}` : ''}</span>`;

export async function getEvidence(ctx, id) {
  if (ctx.cache.has(id)) return ctx.cache.get(id);
  const response = await fetch(`/api/project/evidence/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(`Could not load ${id} (${response.status})`);
  const value = await response.json();
  ctx.cache.set(id, value.data);
  return value.data;
}

export function plot(id, traces, options={}) {
  const target = document.getElementById(id);
  if (!target) return;
  target.setAttribute('role','img');
  target.setAttribute('aria-label',options.ariaLabel||`${options.yaxis?.title||'Data'} chart`);
  if (!window.Plotly) {
    target.innerHTML = '<div class="empty-state"><b>Chart library unavailable</b>Reload the page while connected to the project server.</div>';
    return;
  }
  const layout = {
    paper_bgcolor:'#111a24', plot_bgcolor:'#111a24',
    font:{color:'#aabac8',size:10},
    margin:{l:52,r:18,t:18,b:48},
    xaxis:{gridcolor:'#253541',zerolinecolor:'#344957'},
    yaxis:{gridcolor:'#253541',zerolinecolor:'#344957'},
    legend:{orientation:'h',x:0,y:1.13}, hovermode:'x unified',
    ...options,
  };
  window.Plotly.react(target, traces, layout, {responsive:true,displayModeBar:false});
}

export const COLORS = {ppo:'#69d4df',lqr:'#88adff',mpc:'#c09cff',safe_servo:'#f1b36c',energy:'#75d69e',other:'#92a4b3'};

export function sourceNotes(ctx, sourceIds=[], evidenceIds=[]) {
  const links = [...sourceIds.map(id => sourceLink(ctx,id)), ...evidenceIds.map(id => evidenceLink(ctx,id))].filter(Boolean);
  return links.length ? `<div class="source-pills">${links.join('')}</div>` : '';
}

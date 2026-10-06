import {badge, pageHead, sourceLink, evidenceLink} from '../shared.js';

const labels={
  'live-simulation':['Live simulation','live'],
  'simulation-and-source':['Simulation + source','source'],
  'simulation-and-evidence':['Simulation + evidence','recorded'],
  'simulated-and-designed':['Simulated + designed','source'],
  'recorded-and-runnable':['Recorded + runnable','recorded'],
  recorded:['Recorded evidence','recorded'],
  'planned-no-telemetry':['Planned · no telemetry','planned'],
  'design-stage':['Design stage','planned'],
  'planned-physical-calibration':['Calibration planned','planned'],
};

export async function renderProjectMap(ctx) {
  ctx.mounts.push(()=>{
    const input=document.getElementById('mapSearch'),systems=document.getElementById('subsystemRows'),sources=document.getElementById('sourceRows'),evidence=document.getElementById('evidenceRows'),media=document.getElementById('mediaRows');
    const update=()=>{
      const q=input.value.trim().toLowerCase();
      systems.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.includes(q));
      sources.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.includes(q));
      evidence.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.includes(q));
      media.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.includes(q));
    };
    input.addEventListener('input',update);
  });
  const systems=ctx.catalog.subsystems.map(item=>{
    const [label,kind]=labels[item.status]||['Source-defined','source'];
    const sourceButtons=item.sources.map(id=>sourceLink(ctx,id)).filter(Boolean).join(' ');
    const evidenceButtons=item.evidence.map(id=>evidenceLink(ctx,id)).filter(Boolean).join(' ');
    return `<article class="project-row" data-search="${(item.name+' '+item.group+' '+item.summary).toLowerCase()}"><div><b>${item.name}</b><div style="margin-top:5px">${badge(label,kind)}</div></div><p>${item.summary}</p><div class="row-links">${sourceButtons}${evidenceButtons}</div></article>`;
  }).join('');
  const sources=ctx.catalog.sources.map(item=>`<article class="project-row" data-search="${(item.name+' '+item.path).toLowerCase()}"><div><b>${item.name}</b></div><p>${item.path}</p><div class="row-links">${sourceLink(ctx,item.id,'View source')}</div></article>`).join('');
  const evidence=ctx.evidence.map(item=>`<article class="project-row" data-search="${(item.name+' '+item.kind+' '+item.dataType).toLowerCase()}"><div><b>${item.name}</b><div style="margin-top:5px">${badge(item.available?'Available':'Not in this checkout',item.available?'recorded':'planned')}</div></div><p>${item.kind} · ${item.dataType}</p><div class="row-links">${item.available?`<a class="source-pill" href="${item.sourceUrl}" target="_blank" rel="noopener">Open file</a><a class="source-pill" href="/api/project/evidence/${item.id}" target="_blank" rel="noopener">View data</a>`:''}</div></article>`).join('');
  const media=ctx.media.map(item=>`<article class="project-row" data-search="${(item.name+' '+item.stage+' '+item.description).toLowerCase()}"><div><b>${item.name}</b><div style="margin-top:5px">${badge(item.available?'Available':'Not in this checkout',item.available?'recorded':'planned')}</div></div><p>${item.stage} · ${item.description}</p><div class="row-links">${item.available?`<a class="source-pill" href="${item.url}" target="_blank" rel="noopener">Play video</a>`:''}</div></article>`).join('');
  return `${pageHead('SOURCE CATALOG','Project map','Search the complete module, document, and evidence catalog. Links return only explicitly listed files from the repository.',`<span class="badge source">${ctx.catalog.sources.length} sources</span><span class="badge recorded">${ctx.evidence.length} evidence items</span>`)}
    <input id="mapSearch" class="search" type="search" placeholder="Search controls, sensor, Phase 5, firmware…" aria-label="Search project catalog">
    <div class="section-kicker">SUBSYSTEMS</div><div id="subsystemRows" class="project-list">${systems}</div>
    <div class="section-kicker">PROJECT SOURCES</div><div id="sourceRows" class="project-list">${sources}</div>
    <div class="section-kicker">RECORDED EVIDENCE</div><div id="evidenceRows" class="project-list">${evidence}</div>
    <div class="section-kicker">GENERATED VIDEO ARTIFACTS</div><div id="mediaRows" class="project-list">${media}</div>
    <div class="notice" style="margin-top:14px">The catalog is allowlisted in <code>software/python/ui/project_catalog.json</code>. It does not expose arbitrary filesystem paths or run directories.</div>`;
}

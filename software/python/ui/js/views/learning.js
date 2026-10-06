import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt} from '../shared.js';

const controllerName={policy:'PPO residual',lqr:'Constrained LQR',mpc:'Repository MPC'};
const colorFor={policy:COLORS.ppo,lqr:COLORS.lqr,mpc:COLORS.mpc};
const formatPercent=x=>`${(Number(x||0)*100).toFixed(1)}%`;

function railMarkup(targets=[]) {
  return `<div class="rail-wrap"><div class="rail-labels"><span>−65 mm</span><span>0</span><span>+65 mm</span></div><div class="rail"><i class="rail-center"></i>${targets.map((x,i)=>`<i class="rail-target" title="Goal ${i+1}: ${fmt(x,1)} mm" style="left:${((x+65)/130*100).toFixed(2)}%;background:${i%2?'#c09cff':'#68d7df'}"></i>`).join('')}</div><div class="rail-labels"><span>15 mm exclusion</span><span>Allowed target interval: −48 to +48 mm</span><span>15 mm exclusion</span></div></div>`;
}

function drawRuns(data,metricKey,seed=null) {
  const rows=data.rows||[];
  const chosen=seed?rows.filter(r=>r.seed===Number(seed)):rows;
  const controllers=[...new Set(chosen.map(r=>r.controller))];
  const x=[...new Set(chosen.map(r=>r.seed))].sort((a,b)=>a-b).map(String);
  plot('goalMetricChart',controllers.map(controller=>({
    x:x.map(s=>s.slice(-4)),y:x.map(s=>chosen.find(r=>r.controller===controller&&String(r.seed)===s)?.[metricKey]??null),
    name:controllerName[controller]||controller,type:'bar',marker:{color:colorFor[controller]||COLORS.other},
  })),{barmode:'group',xaxis:{title:seed?'Selected confirmation seed':'Paired confirmation seed'},yaxis:{title:metricKey==='integrated_resonator_energy_mJs'?'Integrated modeled resonator energy [mJ·s]':metricKey==='position_rmse_mm'?'Position RMSE [mm]':'Peak angle [°]'}});
}

function renderGoalHold(ctx,data) {
  const summary=data.summary,ppo=summary.policy,lqr=summary.lqr;
  const rows=data.rows.filter(r=>r.controller==='policy'), seed=rows[0]?.seed;
  const target=rows[0]?.goal_targets_mm||[];
  const ppoRows=data.rows.filter(r=>r.controller==='policy'||r.controller==='ppo');
  const completedHolds=ppoRows.reduce((total,row)=>total+(row.completed_two_second_holds||0),0);
  const goalCount=ppoRows.reduce((total,row)=>total+(row.goal_count||0),0);
  const exclusionIncursions=data.rows.reduce((total,row)=>total+(row.exclusion_zone_violations||0),0);
  return `<div class="grid cols-4">
    <section class="panel stat-card"><div class="stat-label">PPO vs LQR · paired energy ratio</div><div class="stat-value">−${((1-data.promotion.policy_lqr_integrated_resonator_energy_ratio)*100).toFixed(1)}%</div><div class="stat-sub">paired promotion-record ratio</div></section>
    <section class="panel stat-card"><div class="stat-label">PPO vs LQR · tracking</div><div class="stat-value">−${((1-ppo.mean_position_rmse_mm/lqr.mean_position_rmse_mm)*100).toFixed(1)}%</div><div class="stat-sub">paired mean position RMSE</div></section>
    <section class="panel stat-card"><div class="stat-label">PPO hold success</div><div class="stat-value">${completedHolds} / ${goalCount}</div><div class="stat-sub">continuous two-second holds</div></section>
    <section class="panel stat-card"><div class="stat-label">Exclusion-zone incursions</div><div class="stat-value">${exclusionIncursions}</div><div class="stat-sub">across all paired controller runs</div></section>
  </div>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Paired controller results</h2><p>Independent confirmation · same randomized goals and disturbances</p></div></div><div class="metric-strip">${badge('PPO promoted vs LQR','recorded')}${badge('MPC remains stronger on this task','warn')}</div><div id="goalMetricChart" class="chart"></div></section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Random destination task</h2><p>Move, settle, then continue after the dwell</p></div></div>
      <label class="toolbar small muted">Controller <select id="goalController" class="selector"><option value="policy">PPO residual</option><option value="lqr">Constrained LQR</option><option value="mpc">Repository MPC</option></select></label>
      <label class="toolbar small muted" style="margin-top:9px">Confirmation seed <select id="goalSeed" class="selector">${[...new Set(data.rows.map(r=>r.seed))].sort((a,b)=>a-b).map(s=>`<option value="${s}" ${s===seed?'selected':''}>${s}</option>`).join('')}</select></label>
      <div id="targetRail">${railMarkup(target)}</div>
      <div class="metric-strip">${badge('One-second minimum-jerk move','source')}${badge(`${data.task.configuredMinimumHoldSeconds} s configured dwell`,'source')}${badge(`±${data.holdToleranceMm} mm tolerance`,'source')}</div>
      <p class="small muted" id="goalSequenceText">Target sequence is shown for the selected controller and seed.</p>
    </section>
  </div>
  <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Per-seed evidence</h2><p>Stored rows from all eight confirmation seeds</p></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Controller</th><th>Seed</th><th>Energy [mJ·s]</th><th>Position RMSE [mm]</th><th>Angle RMS [°]</th><th>Peak angle [°]</th><th>Goals completed</th><th>Hold fraction</th><th>Zone violations</th><th>MPC p95 [ms]</th></tr></thead><tbody>${data.rows.map(r=>`<tr><td>${controllerName[r.controller]||r.controller}</td><td>${r.seed}</td><td>${fmt(r.integrated_resonator_energy_mJs,5)}</td><td>${fmt(r.position_rmse_mm,4)}</td><td>${fmt(r.angle_rms_deg,4)}</td><td>${fmt(r.peak_angle_deg,4)}</td><td>${r.completed_two_second_holds??'—'} / ${r.goal_count??'—'}</td><td>${formatPercent(r.two_second_hold_fraction)}</td><td>${r.exclusion_zone_violations??'—'}</td><td>${r.mpc_solve_p95_ms==null?'—':fmt(r.mpc_solve_p95_ms,2)}</td></tr>`).join('')}</tbody></table></div></section>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Promotion record</h2><p>Thresholds and gate outcomes from confirmation JSON</p></div>${badge('Accepted vs LQR','recorded')}</div>
      <div class="metric-strip">${badge('≥5% energy improvement','source')}${badge('Paired tracking gate','source')}${badge('Peak-angle gate','source')}${badge('Safety & exclusion gate','source')}${badge('Two-second holds','source')}</div>
      <div class="rule"></div><div class="table-wrap"><table class="data-table"><tbody><tr><td>Paired PPO / LQR integrated resonator energy ratio</td><td>${fmt(data.promotion.policy_lqr_integrated_resonator_energy_ratio*100,2)}%</td></tr><tr><td>Paired PPO / MPC integrated resonator energy ratio</td><td>${fmt(data.promotion.policy_mpc_integrated_resonator_energy_ratio*100,2)}%</td></tr><tr><td>Passed gates vs LQR</td><td>${data.promotion.accepted_vs_lqr?'Yes':'No'}</td></tr><tr><td>Passed gates vs MPC</td><td>No; PPO energy is higher</td></tr></tbody></table></div>
      <p class="small muted">The experiment’s reported “energy” is integrated resonator kinetic-plus-spring energy. MPC records the lowest mean vibration energy and tracking RMSE of these three controllers.</p>
    </section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Run provenance</h2><p>Reproducible study context and files</p></div></div>
      <div class="metric-strip">${badge(data.benchmarkVersion,'recorded')}${badge('Source revision '+String(data.sourceRevision||'').slice(0,7),'source')}${badge('8 confirmation seeds','recorded')}</div>
      <p class="small muted">Task: randomized goals with a 15 mm rail-edge exclusion band; targets sampled inside ±48 mm on a rail with 65 mm half-travel. Successful holds require two continuous seconds within tolerance; the environment config asks for at least 2.5 seconds per dwell.</p>
      ${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation','goal-hold-development','goal-hold-training','goal-hold-metadata'])}
    </section>
  </div>`;
}

export async function renderLearning(ctx) {
  const items=['goal-hold-confirmation','goal-hold-development','goal-hold-training','goal-hold-metadata','ppo-training','ppo-ablation','teacher-fit','teacher-40-epochs'].map(id=>ctx.evidence.find(e=>e.id===id)).filter(Boolean);
  const videos=ctx.media.map(item=>`<section class="panel video-card"><div class="panel-title"><b>${item.name}</b>${badge(item.stage,'recorded')}</div>${item.available?`<video controls preload="metadata" src="${item.url}" aria-label="${item.name}"></video>`:`<div class="empty-state"><b>Video unavailable</b>Recorded file is not in this checkout.</div>`}<div class="panel-pad"><p class="small muted">${item.description}</p></div></section>`).join('');
  ctx.mounts.push(()=>{
    const select=document.getElementById('learningEvidence');
    const content=document.getElementById('learningDetail');
    const mountDetail=async()=>{
      try {
        const id=select.value,data=await getEvidence(ctx,id);
        if(id==='goal-hold-confirmation'||id==='goal-hold-development') {
          data.id=id;content.innerHTML=renderGoalHold(ctx,data);
          const draw=()=>{
            const seedSel=document.getElementById('goalSeed'),controllerSel=document.getElementById('goalController');
            drawRuns(data,'integrated_resonator_energy_mJs');
            const updateSequence=()=>{
              const row=data.rows.find(r=>r.seed===Number(seedSel.value)&&r.controller===controllerSel.value);
              document.getElementById('targetRail').innerHTML=railMarkup(row?.goal_targets_mm||[]);
              document.getElementById('goalSequenceText').textContent=row?`Targets for ${controllerName[row.controller]||row.controller}, seed ${row.seed}: ${(row.goal_targets_mm||[]).map(v=>`${fmt(v,1)} mm`).join(' · ')}. Completed ${row.completed_two_second_holds} of ${row.goal_count} in-tolerance holds.`:'No target sequence in this result.';
            };
            seedSel.addEventListener('change',updateSequence);controllerSel.addEventListener('change',updateSequence);updateSequence();
          };
          draw();
        } else if(Array.isArray(data)) {
          const points=data.filter(row=>row&&typeof row==='object');
          const keys=['mean_reward','explained_variance','action_mean_abs','kl','policy','value'].filter(key=>points.some(row=>typeof row[key]==='number'));
          content.innerHTML=`<section class="panel panel-pad"><div class="panel-head"><div><h2>Saved training curve</h2><p>Metrics emitted by the recorded PPO updates; horizontal axis is environment steps.</p></div>${badge(`${points.length} updates`,'recorded')}</div><label class="toolbar small muted">Metric <select id="trainingMetric" class="selector">${keys.map(key=>`<option value="${key}" ${key==='mean_reward'?'selected':''}>${key.replaceAll('_',' ')}</option>`).join('')}</select></label><div id="savedTrainingChart" class="chart"></div><details><summary class="small">Show saved update records</summary><pre class="source-preview">${JSON.stringify(points,null,2).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</pre></details></section>`;
          const metricSelect=document.getElementById('trainingMetric');
          const updateCurve=()=>plot('savedTrainingChart',[{x:points.map(row=>row.steps),y:points.map(row=>row[metricSelect.value]),name:metricSelect.value.replaceAll('_',' '),mode:'lines+markers',line:{color:COLORS.ppo}}],{xaxis:{title:'Environment steps'},yaxis:{title:metricSelect.value.replaceAll('_',' ')}});
          metricSelect.addEventListener('change',updateCurve);updateCurve();
        } else {
          const text=JSON.stringify(data,null,2);
          content.innerHTML=`<section class="panel panel-pad"><div class="panel-head"><div><h2>${id==='goal-hold-metadata'?'Run configuration':'Historical experiment evidence'}</h2><p>Source JSON is shown as committed; machine-local artifact paths are filtered by the server.</p></div>${badge('Source data','recorded')}</div><pre class="source-preview">${text.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</pre></section>`;
        }
      } catch(err) {content.innerHTML=`<div class="empty-state"><b>Evidence unavailable</b>${String(err.message||err)}</div>`;}
    };
    select.addEventListener('change',mountDetail);mountDetail();
  });
  return `${pageHead('EXPERIMENT LIBRARY','Learning & experiments','Inspect versioned PPO studies, MPC teacher work, training progress, paired benchmark outcomes, and local v4 training controls. Historical results remain tied to their own task definitions.',`<a class="button primary" href="/live?view=training" target="_blank" rel="noopener">Start or evaluate a run ↗</a>`)}
    <div class="notice blue"><b>Scope of the highlighted result:</b> PPO passed its 5% vibration-energy gate against LQR in the random-goal task. Repository MPC also passes the holds and has lower mean modeled vibration energy and tracking RMSE; the evidence does not claim PPO beats MPC.</div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="controls-line"><div class="panel-head"><div><h2>Saved experiment</h2><p>Choose an evaluation, training history, or run configuration</p></div></div><label class="toolbar small muted">Evidence <select id="learningEvidence" class="selector">${items.map(item=>`<option value="${item.id}" ${item.id==='goal-hold-confirmation'?'selected':''} ${item.available?'':'disabled'}>${item.name}${item.available?'':' · unavailable'}</option>`).join('')}</select></label></div><div id="learningDetail" style="margin-top:14px"></div></section>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Live v4 workflow</h2><p>Start training, MPC teacher collection, checkpoint evaluation, or a standalone MPC reference</p></div>${badge('Local job controls','live')}</div><p class="small muted">Training runs save progress and artifacts on the machine serving this page. The original controls include cancellation, promotion results, and checkpoint loading into the live simulator.</p><div class="head-actions"><a class="button primary" href="/live?view=training" target="_blank" rel="noopener">Open v4 training controls ↗</a></div></section>
    ${videos?`<div class="section-kicker">SELECTED PPO TRAINING STAGES · ${ctx.media.filter(v=>v.available).length} LOCAL VIDEOS</div><div class="video-grid">${videos}</div>`:''}
    <div class="section-kicker">OTHER PROJECT EVIDENCE</div><div class="evidence-list">${ctx.evidence.filter(e=>e.available&&e.id!=='goal-hold-confirmation').map(item=>`<a class="evidence-row" href="/api/project/evidence/${item.id}/source" target="_blank" rel="noopener"><span><b>${item.name}</b><small>${item.kind} · ${item.dataType}</small></span>${badge('Open source','recorded')}</a>`).join('')}</div>`;
}

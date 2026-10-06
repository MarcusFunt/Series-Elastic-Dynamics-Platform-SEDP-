import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt} from '../shared.js';

const metricMap={
  energy:{field:'integrated_resonator_energy_mJs',label:'Integrated resonator energy [mJ·s]'},
  tracking:{field:'position_rmse_mm',label:'Position tracking RMSE [mm]'},
  angle:{field:'angle_rms_deg',label:'Resonator angle RMS [°]'},
  peak:{field:'peak_angle_deg',label:'Peak resonator angle [°]'},
  rail:{field:'peak_rail_fraction',label:'Peak rail fraction'},
  holds:{field:'two_second_hold_fraction',label:'Two-second hold fraction'},
};
const title={policy:'PPO residual',ppo:'PPO residual',lqr:'Constrained LQR',mpc:'Repository MPC',safe_servo:'Safe servo',energy:'Energy damping'};

function displayRows(data,kind) {
  const rows=data.rows||[];
  return rows.map(r=>({...r,
    controller:r.controller==='policy'?'ppo':r.controller,
    integrated_resonator_energy_mJs:r.integrated_resonator_energy_mJs??r.energy_integral_mJs,
    mpc_solve_p95_ms:r.mpc_solve_p95_ms??r.solve_p95_ms,
  }));
}
function draw(ctx,data,kind) {
  const metricName=document.getElementById('controlMetric').value,def=metricMap[metricName];
  const rows=displayRows(data,kind);
  const controllers=[...new Set(rows.map(r=>r.controller))];
  const seeds=[...new Set(rows.map(r=>r.seed).filter(x=>x!==undefined))];
  const categories=kind==='goal-hold'?seeds:([...new Set(rows.map(r=>r.scenario||''))]);
  const traces=controllers.map(controller=>({
    x:categories.map(cat=>kind==='goal-hold'?String(cat):String(cat).replace('_',' ')),
    y:categories.map(cat=>{
      const matching=rows.filter(r=>r.controller===controller&&(kind==='goal-hold'?r.seed===cat:(r.scenario||'')===cat));
      if(!matching.length)return null;
      return matching.reduce((n,r)=>n+(Number(r[def.field])||0),0)/matching.length;
    }),
    name:title[controller]||controller, type:'bar',
    marker:{color:COLORS[controller]||COLORS.other},
  }));
  plot('controllerChart',traces,{barmode:kind==='goal-hold'?'group':'group',xaxis:{title:kind==='goal-hold'?'Paired confirmation seed':'Scenario'},yaxis:{title:def.label}});
  const table=document.getElementById('controllerRows');
  if(table) table.innerHTML=rows.map(r=>`<tr><td>${title[r.controller]||r.controller}</td><td>${r.seed??r.scenario??'—'}</td><td>${fmt(r[def.field],4)}</td><td>${fmt(r.position_rmse_mm,4)}</td><td>${fmt(r.angle_rms_deg,4)}</td><td>${r.accepted===undefined?'—':r.accepted?'Yes':'No'}</td></tr>`).join('');
}

export async function renderControls(ctx) {
  const items=['goal-hold-confirmation','benchmark-b1','mpc-reference','mpc-phase4'].map(id=>ctx.evidence.find(e=>e.id===id)).filter(Boolean);
  const options=items.map(e=>`<option value="${e.id}" ${e.id==='goal-hold-confirmation'?'selected':''}>${e.name}</option>`).join('');
  ctx.mounts.push(()=>{
    const select=document.getElementById('controlEvidence'),metric=document.getElementById('controlMetric');
    async function refresh(){
      const item=items.find(e=>e.id===select.value);if(!item)return;
      const value=await getEvidence(ctx,item.id);const kind=item.dataType;
      const availableMetrics=Object.keys(metricMap).filter(key=>displayRows(value,kind).some(row=>typeof row[metricMap[key].field]==='number'));
      if(!availableMetrics.includes(metric.value))metric.value=availableMetrics.includes('tracking')?'tracking':availableMetrics[0]||'tracking';
      const details=document.getElementById('controlSummary');
      const reportName=[value.benchmark_version,value.report,value.experiment].find(x=>typeof x==='string')||item.kind;
      details.innerHTML=kind==='goal-hold'?`<div class="metric-strip">${badge('8 paired seeds','recorded')}${badge('PPO accepted vs LQR','recorded')}${badge('MPC reference','recorded')}${badge('MPC energy & tracking lower than PPO','warn')}</div><p class="small muted">Each seed uses the same randomized targets and disturbances across controllers. For this task, PPO’s modeled resonator energy is ${((value.promotion.policy_mpc_integrated_resonator_energy_ratio-1)*100).toFixed(1)}% higher than MPC’s; its promotion result is against LQR only.</p>`:`<div class="metric-strip">${badge(reportName,'recorded')}${badge(`${(value.rows||[]).length} recorded comparison rows`,'recorded')}</div><p class="small muted">Metrics come from the selected committed report. Scenario sets differ between benchmark versions.</p>`;
      draw(ctx,value,kind);
    }
    select.addEventListener('change',refresh);metric.addEventListener('change',refresh);refresh();
  });
  return `${pageHead('CONTROLLER EVIDENCE','Control lab','Compare recorded controller outcomes and live simulator behavior. Stored comparisons preserve their original scenario and seed sets.',`<a class="button" href="#/rig">Try controllers in simulator →</a>`)}
    <div class="notice blue"><b>Comparison rule:</b> reports are compared only within the selected evidence set. B1, Phase 4 MPC, and random-goal-and-hold use different task definitions. “Resonator energy” is modeled vibration energy, not electrical power.</div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="controls-line"><div class="toolbar"><label>Evidence set <select id="controlEvidence" class="selector">${options}</select></label><label>Metric <select id="controlMetric" class="selector"><option value="energy">Integrated resonator energy</option><option value="tracking">Position tracking RMSE</option><option value="angle">Angle RMS</option><option value="peak">Peak angle</option><option value="rail">Peak rail fraction</option><option value="holds">Two-second hold fraction</option></select></label></div>${badge('Stored evidence','recorded')}</div>
      <div id="controlSummary" style="margin-top:14px"></div><div id="controllerChart" class="chart tall"></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Controller</th><th>Seed / scenario</th><th>Selected metric</th><th>Position RMSE [mm]</th><th>Angle RMS [°]</th><th>Accepted</th></tr></thead><tbody id="controllerRows"></tbody></table></div>
    </section>
    <div class="grid cols-2" style="margin-top:14px">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Controller ladder</h2><p>Project-defined comparison from simple servo through learned feedback</p></div></div><div class="flow-vertical"><div class="flow-card"><b>Trajectory servo / direct damping</b><span>classical baseline and vibration feedback</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Constrained LQR / LQG</b><span>reduced dynamics with rail safety constraints</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Constrained MPC</b><span>repository implementation · solver status measured in saved runs</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Adaptive / residual RL</b><span>candidate refinement with the safety layer retained</span></div></div>${sourceNotes(ctx,['project-overview','residual-v4'],['benchmark-b1','mpc-reference'])}</section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>What the current evidence says</h2><p>Random goal-hold task</p></div>${badge('8-seed confirmation','recorded')}</div><ul class="small muted"><li>PPO clears the 5% integrated resonator-energy gate against LQR.</li><li>It completes all in-tolerance holds and avoids the exclusion zone.</li><li>Repository MPC has lower integrated resonator energy, lower tracking error, and lower angle metrics in this study.</li><li>The PPO promotion record is therefore explicitly scoped to LQR; the evidence does not claim PPO beats MPC.</li></ul>${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation'])}</section>
    </div>`;
}

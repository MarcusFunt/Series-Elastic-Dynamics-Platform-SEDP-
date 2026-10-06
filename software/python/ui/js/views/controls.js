import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt} from '../shared.js';

const metricMap={
  energy:{field:'integrated_resonator_energy_mJs',label:'Integrated resonator energy [mJ·s]'},
  tracking:{field:'position_rmse_mm',label:'Position tracking RMSE [mm]'},
  angle:{field:'angle_rms_deg',label:'Resonator angle RMS [°]'},
  peak:{field:'peak_angle_deg',label:'Peak resonator angle [°]'},
  rail:{field:'peak_rail_fraction',label:'Peak rail fraction'},
  holds:{field:'two_second_hold_fraction',label:'Two-second hold fraction'},
  solve:{field:'mpc_solve_p95_ms',label:'MPC solve p95 [ms]'},
  fallback:{field:'mpc_fallback_fraction',label:'MPC fallback fraction'},
};
const title={policy:'PPO residual',ppo:'PPO residual',lqr:'Constrained LQR',mpc:'Repository MPC',
  mpc_baseline:'Preserved MPC baseline',safe_servo:'Safe servo',energy:'Energy damping'};

function displayRows(data) {
  return (data.rows||[]).map(row=>({...row,
    controller:row.controller==='policy'?'ppo':row.controller,
    integrated_resonator_energy_mJs:row.integrated_resonator_energy_mJs??row.energy_integral_mJs,
    mpc_solve_p95_ms:row.mpc_solve_p95_ms??row.solve_p95_ms,
  }));
}

function conditionKey(row,kind) {
  if(kind==='goal-hold')return `seed:${row.seed}`;
  return JSON.stringify([
    row.scenario??'',row.seed??'',row.randomized??'',row.actuator_mode??'',
    row.command_delay_seconds??0,row.command_jitter_seconds??0,
  ]);
}

function conditionLabel(row,kind) {
  if(kind==='goal-hold')return `seed ${row.seed??'—'}`;
  const delay=Number(row.command_delay_seconds||0),jitter=Number(row.command_jitter_seconds||0);
  const actuator=row.actuator_mode==='step_dir'?'STEP/DIR':row.actuator_mode;
  return [row.scenario||'scenario',row.seed!==undefined?`seed ${row.seed}`:null,actuator,
    delay>0?`${(delay*1000).toFixed(0)} ms delay`:null,
    jitter>0?`${(jitter*1000).toFixed(0)} ms jitter`:null].filter(Boolean).join(' · ');
}

function tableCondition(row,kind) {
  if(kind==='goal-hold')return 'Random goal-hold';
  return conditionLabel({...row,seed:undefined},kind);
}

function draw(ctx,data,kind) {
  const metricName=document.getElementById('controlMetric').value,def=metricMap[metricName];
  const rows=displayRows(data),controllers=[...new Set(rows.map(row=>row.controller))];
  const categories=[...new Map(rows.map(row=>[conditionKey(row,kind),row])).entries()];
  const traces=controllers.map(controller=>({
    x:categories.map(([,row])=>conditionLabel(row,kind)),
    y:categories.map(([key])=>{
      const matching=rows.filter(row=>row.controller===controller&&conditionKey(row,kind)===key);
      if(!matching.length)return null;
      return matching.reduce((sum,row)=>sum+(Number(row[def.field])||0),0)/matching.length;
    }),
    name:title[controller]||controller,type:'bar',
    marker:{color:COLORS[controller]||COLORS.other},
  }));
  plot('controllerChart',traces,{barmode:'group',xaxis:{title:kind==='goal-hold'?'Paired seed':'Scenario, seed, actuator, and latency condition'},yaxis:{title:def.label}});
  const table=document.getElementById('controllerRows');
  if(table)table.innerHTML=rows.map(row=>{
    const isMpc=['mpc','mpc_baseline'].includes(row.controller);
    return `<tr><td>${title[row.controller]||row.controller}</td><td>${tableCondition(row,kind)}</td><td>${row.seed??'—'}</td><td>${fmt(row[def.field],4)}</td><td>${fmt(row.position_rmse_mm,4)}</td><td>${fmt(row.angle_rms_deg,4)}</td><td>${isMpc?fmt(row.mpc_solve_p95_ms,2):'—'}</td><td>${isMpc&&typeof row.mpc_fallback_fraction==='number'?`${(row.mpc_fallback_fraction*100).toFixed(1)}%`:'—'}</td><td>${row.accepted===undefined?'—':row.accepted?'Yes':'No'}</td></tr>`;
  }).join('')||'<tr><td colspan="9">No comparison rows are available for this report.</td></tr>';
}

export async function renderControls(ctx) {
  const items=['goal-hold-confirmation','benchmark-b1','mpc-reference','mpc-phase4']
    .map(id=>ctx.evidence.find(item=>item.id===id)).filter(item=>item?.available);
  const options=items.map(item=>`<option value="${item.id}" ${item.id==='goal-hold-confirmation'?'selected':''}>${item.name}</option>`).join('');
  ctx.mounts.push(()=>{
    const select=document.getElementById('controlEvidence'),metric=document.getElementById('controlMetric');
    if(!items.length){
      document.getElementById('controlSummary').innerHTML='<div class="empty-state"><b>Evidence unavailable</b>No supported controller reports are available in this checkout.</div>';
      return;
    }
    let refreshGeneration=0;
    async function refresh(){
      const generation=++refreshGeneration;
      const item=items.find(entry=>entry.id===select.value);if(!item)return;
      try {
        const value=await getEvidence(ctx,item.id);
        if(!ctx.isCurrent()||generation!==refreshGeneration)return;
        const kind=item.dataType,rows=displayRows(value);
        const availableMetrics=Object.keys(metricMap).filter(key=>rows.some(row=>typeof row[metricMap[key].field]==='number'));
        if(!availableMetrics.includes(metric.value))metric.value=availableMetrics.includes('tracking')?'tracking':availableMetrics[0]||'tracking';
        const details=document.getElementById('controlSummary');
        const reportName=[value.benchmark_version,value.report,value.experiment].find(entry=>typeof entry==='string')||item.kind;
        const hasConditions=rows.some(row=>row.actuator_mode||Number(row.command_delay_seconds||0)>0);
        details.innerHTML=kind==='goal-hold'
          ?`<div class="metric-strip">${badge(`${new Set(rows.map(row=>row.seed)).size} paired seeds`,'recorded')}${badge('PPO accepted vs LQR','recorded')}${badge('MPC reference','recorded')}${badge('MPC energy & tracking lower than PPO','warn')}</div><p class="small muted">Each seed uses the same randomized targets and disturbances across controllers. For this task, PPO’s modeled resonator energy is ${((value.promotion.policy_mpc_integrated_resonator_energy_ratio-1)*100).toFixed(1)}% higher than MPC’s; its promotion result is against LQR only.</p>`
          :`<div class="metric-strip">${badge(reportName,'recorded')}${badge(`${rows.length} recorded comparison rows`,'recorded')}</div><p class="small muted">Metrics come from the selected committed report. Chart categories retain scenario, seed, actuator, delay, and jitter so unlike conditions are not averaged together.${hasConditions?' MPC timing and fallback diagnostics are shown when recorded.':''}</p>`;
        draw(ctx,value,kind);
      } catch(error) {
        if(!ctx.isCurrent()||generation!==refreshGeneration)return;
        const details=document.getElementById('controlSummary'),table=document.getElementById('controllerRows');
        if(details)details.innerHTML=`<div class="empty-state"><b>Evidence unavailable</b>${String(error.message||error)}</div>`;
        if(table)table.innerHTML='<tr><td colspan="9">The selected report could not be loaded.</td></tr>';
      }
    }
    select.addEventListener('change',refresh);metric.addEventListener('change',refresh);refresh();
  });
  return `${pageHead('CONTROLLER EVIDENCE','Control lab','Compare recorded controller outcomes and live simulator behavior. Stored comparisons preserve their original scenario, seed, actuator, and timing conditions.',`<a class="button" href="#/rig">Try controllers in simulator →</a>`)}
    <div class="notice blue"><b>Comparison rule:</b> reports are compared only within the selected evidence set. B1, Phase 4 MPC, and random-goal-and-hold use different task definitions. “Resonator energy” is modeled vibration energy, not electrical power.</div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="controls-line"><div class="toolbar"><label>Evidence set <select id="controlEvidence" class="selector">${options}</select></label><label>Metric <select id="controlMetric" class="selector"><option value="energy">Integrated resonator energy</option><option value="tracking">Position tracking RMSE</option><option value="angle">Angle RMS</option><option value="peak">Peak angle</option><option value="rail">Peak rail fraction</option><option value="holds">Two-second hold fraction</option><option value="solve">MPC solve p95 [ms]</option><option value="fallback">MPC fallback fraction</option></select></label></div>${badge('Stored evidence','recorded')}</div>
      <div id="controlSummary" style="margin-top:14px"></div><div id="controllerChart" class="chart tall"></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Controller</th><th>Scenario / actuator / latency</th><th>Seed</th><th>Selected metric</th><th>Position RMSE [mm]</th><th>Angle RMS [°]</th><th>MPC solve p95 [ms]</th><th>MPC fallback</th><th>Accepted</th></tr></thead><tbody id="controllerRows"></tbody></table></div>
    </section>
    <div class="grid cols-2" style="margin-top:14px">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Controller ladder</h2><p>Project-defined comparison from simple servo through learned feedback</p></div></div><div class="flow-vertical"><div class="flow-card"><b>Trajectory servo / direct damping</b><span>classical baseline and vibration feedback</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Constrained LQR / LQG</b><span>reduced dynamics with rail safety constraints</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Constrained MPC</b><span>repository implementation · solver status measured in saved runs</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Adaptive / residual RL</b><span>candidate refinement with the safety layer retained</span></div></div>${sourceNotes(ctx,['project-overview','residual-v4'],['benchmark-b1','mpc-reference'])}</section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>What the current evidence says</h2><p>Random goal-hold task</p></div>${badge('8-seed confirmation','recorded')}</div><ul class="small muted"><li>PPO clears the 5% integrated resonator-energy gate against LQR.</li><li>It completes all in-tolerance holds and avoids the exclusion zone.</li><li>Repository MPC has lower integrated resonator energy, lower tracking error, and lower angle metrics in this study.</li><li>The PPO promotion record is therefore explicitly scoped to LQR; the evidence does not claim PPO beats MPC.</li></ul>${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation'])}</section>
    </div>`;
}

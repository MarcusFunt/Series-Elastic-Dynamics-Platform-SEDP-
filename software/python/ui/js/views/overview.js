import {badge, sourceLink, statCard, sourceNotes, getEvidence, fmt} from '../shared.js';

const visualState = status => status==='live-simulation'?['Live simulation','live']:status==='recorded'?['Recorded evidence','recorded']:status==='planned-no-telemetry'||status==='planned-physical-calibration'||status==='design-stage'?['Planned / no telemetry','planned']:['Simulation + source','source'];
const viewBySubsystem={
  'rig-simulation':'rig','digital-twin':'models','controls-mpc':'controls',
  'sensors-timing':'sensors','learning-experiments':'learning',
  'benchmarks-evidence':'project-map','hardware-firmware':'hardware',
  haptics:'haptics','system-identification':'haptics',
};

export async function renderOverview(ctx) {
  let goal=null;
  try { goal=await getEvidence(ctx,'goal-hold-confirmation'); } catch { /* Show the rest of the workspace if this optional report is absent. */ }
  const summary=goal?.summary||{},promotion=goal?.promotion||{},policy=summary.policy||{},lqr=summary.lqr||{},mpc=summary.mpc||{};
  const policyRows=(goal?.rows||[]).filter(row=>['policy','ppo'].includes(row.controller));
  const confirmed=policyRows.reduce((n,row)=>n+(row.completed_two_second_holds||0),0);
  const goalCount=policyRows.reduce((n,row)=>n+(row.goal_count||0),0);
  const seedCount=new Set(policyRows.map(row=>row.seed)).size;
  const energyRatio=promotion.policy_lqr_integrated_resonator_energy_ratio;
  const accepted=Boolean(promotion.accepted_vs_lqr);
  const cards=ctx.catalog.subsystems.map(item=>{
    const [status,kind]=visualState(item.status);
    return `<button class="subsystem-card" data-go="${viewBySubsystem[item.id]||'project-map'}"><div class="card-top"><span class="iconbox">${item.group==='Dynamics'?'◉':item.group==='Control'?'⌁':item.group==='Runtime'?'⌗':item.group==='Learning'?'⌘':item.group==='Physical system'?'▤':'⌖'}</span>${badge(status,kind)}</div><h3>${item.name}</h3><p>${item.summary}</p></button>`;
  }).join('');
  const result=goal?`<div class="panel-head"><div><h2>Latest highlighted result</h2><p>Random destinations with a dwell period · paired confirmation</p></div>${badge(accepted?'Passed vs LQR':'Not promoted vs LQR',accepted?'recorded':'warn')}</div>
      <div class="grid cols-3">
        <div><div class="stat-label">PPO resonator energy</div><div class="stat-value" style="font-size:18px">${fmt(policy.mean_integrated_resonator_energy_mJs,5)}</div><div class="stat-sub">mJ·s integrated modeled vibration energy</div></div>
        <div><div class="stat-label">LQR comparator</div><div class="stat-value" style="font-size:18px">${fmt(lqr.mean_integrated_resonator_energy_mJs,5)}</div><div class="stat-sub">Same goal sequences and disturbances</div></div>
        <div><div class="stat-label">Repository MPC</div><div class="stat-value" style="font-size:18px">${fmt(mpc.mean_integrated_resonator_energy_mJs,5)}</div><div class="stat-sub">PPO vibration energy ${Number.isFinite(promotion.policy_mpc_integrated_resonator_energy_ratio)?`${((promotion.policy_mpc_integrated_resonator_energy_ratio-1)*100).toFixed(1)}% higher`: 'not reported'} than MPC</div></div>
      </div>
      <div class="notice blue" style="margin-top:14px"><b>Metric clarification.</b> “Energy” here is the integral of modeled resonator kinetic-plus-spring energy over time. It measures vibration, not electrical consumption or total motor energy.</div>
      ${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation'])}`
    :`<div class="empty-state"><b>Highlighted evaluation unavailable</b>The project coverage and source catalog are still available. The saved random-goal confirmation report is missing or could not be loaded.</div>`;
  return `
  <div class="page-head"><div><div class="eyebrow">PROJECT OVERVIEW · ${ctx.catalog.project.maturity.toUpperCase()}</div><h1>One workspace for the whole rig</h1><p>Explore the live plant, twin models, controllers, experiments, firmware design, and planned hardware. Each view says whether its data is live, recorded, or design-only.</p></div><div class="head-actions"><a class="button primary" href="#/rig">Open live simulator <span>→</span></a></div></div>
  <div class="grid cols-4">
    ${statCard('Project stage','Simulation first','Physical RP2350 rig not yet validated','◌')}
    ${statCard('Project areas',String(ctx.catalog.subsystems.length),'Dynamics through system identification','◈')}
    ${statCard('Goal holds completed',goal?`${confirmed} / ${goalCount||'—'}`:'Unavailable',goal?`${seedCount} paired confirmation seeds`:'Saved evaluation not available','✓')}
    ${statCard('PPO vs LQR',Number.isFinite(energyRatio)?`${((1-energyRatio)*100).toFixed(1)}%`:'Unavailable',goal?'Integrated resonator energy on this task':'Saved evaluation not available','↘')}
  </div>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad">${result}</section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>What the goal task asks</h2><p>Random target locations with protected end zones and time to settle</p></div>${badge(goal?'Recorded setup':'Task definition','recorded')}</div>
      <div class="metric-strip">${badge('15 mm edge exclusion','warn')}${badge('≥2 s continuous in tolerance','source')}${badge(`${goal?.task?.configuredMinimumHoldSeconds??2.5} s configured minimum dwell`,'source')}</div>
      <div class="rail-wrap"><div class="rail-labels"><span>−65 mm rail end</span><span>0</span><span>+65 mm rail end</span></div><div class="rail"><i class="rail-center"></i><i class="rail-target" style="left:16%"></i><i class="rail-target" style="left:78%;background:#b895ff"></i></div><div class="rail-labels"><span>15 mm exclusion band</span><span>goal target range: ±48 mm</span><span>15 mm exclusion band</span></div></div>
      <p class="small muted">The success check requires two continuous seconds within ${goal?.holdToleranceMm??5} mm of a target. The task config requests at least ${goal?.task?.configuredMinimumHoldSeconds??2.5} seconds dwell before advancing.</p>
      <div class="head-actions"><button class="button" data-go="learning">Explore this experiment →</button></div>
    </section>
  </div>
  <div class="section-kicker">PROJECT COVERAGE · ${ctx.catalog.subsystems.length} AREAS</div>
  <div class="subsystem-grid">${cards}</div>
  <div class="section-kicker">QUICK LINKS</div>
  <div class="grid cols-2">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Recorded evidence</h2><p>Benchmarks and reports committed with the project source</p></div></div><div class="evidence-list">${ctx.evidence.filter(e=>e.available).slice(0,5).map(e=>`<a class="evidence-row" href="/api/project/evidence/${e.id}/source" target="_blank" rel="noopener"><span><b>${e.name}</b><small>${e.kind}</small></span>${badge('Available','recorded')}</a>`).join('')||'<div class="empty-state">No saved evidence is available in this checkout.</div>'}</div><div class="head-actions" style="margin-top:12px"><button class="button" data-go="project-map">Browse all sources →</button></div></section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Project documents</h2><p>Core definitions and implementation plans</p></div></div><div class="source-pills">${sourceLink(ctx,'project-overview')}${sourceLink(ctx,'math-model')}${sourceLink(ctx,'hardware')}${sourceLink(ctx,'firmware')}${sourceLink(ctx,'openmodelica')}${sourceLink(ctx,'phase5-plan')}</div><div class="rule"></div><div class="notice"><b>Hardware state:</b> design documented; no physical rig has been calibrated or validated. Live values in the Rig simulator come from the Python simulation.</div></section>
  </div>`;
}

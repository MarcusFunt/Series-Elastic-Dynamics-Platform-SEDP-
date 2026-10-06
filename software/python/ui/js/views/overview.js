import {badge, sourceLink, evidenceLink, statCard, sourceNotes} from '../shared.js';

const visualState = status => status==='live-simulation'?['Live simulation','live']:status==='recorded'?['Recorded evidence','recorded']:status==='planned-no-telemetry'||status==='planned-physical-calibration'||status==='design-stage'?['Planned / no telemetry','planned']:['Simulation + source','source'];
const viewBySubsystem={
  'rig-simulation':'rig','digital-twin':'models','controls-mpc':'controls',
  'sensors-timing':'sensors','learning-experiments':'learning',
  'benchmarks-evidence':'project-map','hardware-firmware':'hardware',
  haptics:'haptics','system-identification':'haptics',
};

export async function renderOverview(ctx) {
  const goal = await (await fetch('/api/project/evidence/goal-hold-confirmation')).json();
  const p = goal.data.promotion, s=goal.data.summary, policy=s.policy, mpc=s.mpc;
  const cards=ctx.catalog.subsystems.map(item=>{
    const [status,kind]=visualState(item.status);
    return `<button class="subsystem-card" data-go="${viewBySubsystem[item.id]||'project-map'}"><div class="card-top"><span class="iconbox">${item.group==='Dynamics'?'◉':item.group==='Control'?'⌁':item.group==='Runtime'?'⌗':item.group==='Learning'?'⌘':item.group==='Physical system'?'▤':'⌖'}</span>${badge(status,kind)}</div><h3>${item.name}</h3><p>${item.summary}</p></button>`;
  }).join('');
  const confirmed=goal.data.rows.filter(r=>r.controller==='policy'||r.controller==='ppo').reduce((n,r)=>n+(r.completed_two_second_holds||0),0);
  return `
  <div class="page-head"><div><div class="eyebrow">PROJECT OVERVIEW · ${ctx.catalog.project.maturity.toUpperCase()}</div><h1>One workspace for the whole rig</h1><p>Explore the live plant, twin models, controllers, experiments, firmware design, and planned hardware. Each view says whether its data is live, recorded, or design-only.</p></div><div class="head-actions"><a class="button primary" href="#/rig">Open live simulator <span>→</span></a></div></div>
  <div class="grid cols-4">
    ${statCard('Project stage','Simulation first','Physical RP2350 rig not yet validated','◌')}
    ${statCard('Project areas',String(ctx.catalog.subsystems.length),'Dynamics through system identification','◈')}
    ${statCard('Goal holds completed',`${confirmed} / 24`,'Independent PPO confirmation · all inside tolerance','✓')}
    ${statCard('PPO vs LQR','−7.1%','Integrated resonator energy on this task','↘')}
  </div>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad">${`<div class="panel-head"><div><h2>Latest highlighted result</h2><p>Random destinations with a dwell period · independent paired confirmation</p></div>${badge('Passed vs LQR','recorded')}</div>`}
      <div class="grid cols-3">
        <div><div class="stat-label">PPO resonator energy</div><div class="stat-value" style="font-size:18px">${policy.mean_integrated_resonator_energy_mJs.toFixed(5)}</div><div class="stat-sub">mJ·s integrated modeled vibration energy</div></div>
        <div><div class="stat-label">LQR comparator</div><div class="stat-value" style="font-size:18px">${s.lqr.mean_integrated_resonator_energy_mJs.toFixed(5)}</div><div class="stat-sub">Same goal sequences and disturbances</div></div>
        <div><div class="stat-label">Repository MPC</div><div class="stat-value" style="font-size:18px">${mpc.mean_integrated_resonator_energy_mJs.toFixed(5)}</div><div class="stat-sub">PPO is ${((policy.mean_integrated_resonator_energy_mJs/mpc.mean_integrated_resonator_energy_mJs-1)*100).toFixed(1)}% higher than MPC energy</div></div>
      </div>
      <div class="notice blue" style="margin-top:14px"><b>Metric clarification.</b> “Energy” here is the integral of modeled resonator kinetic-plus-spring energy over time. It measures vibration, not electrical consumption or total motor energy.</div>
      ${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation'])}
    </section>
    <section class="panel panel-pad">${`<div class="panel-head"><div><h2>What the goal task asks</h2><p>Random target locations with protected end zones and time to settle</p></div>${badge('Recorded setup','recorded')}</div>`}
      <div class="metric-strip">${badge('±15 mm edge exclusion','warn')}${badge('≥2 s continuous in tolerance','source')}${badge('2.5 s configured minimum dwell','source')}</div>
      <div class="rail-wrap"><div class="rail-labels"><span>−65 mm rail end</span><span>0</span><span>+65 mm rail end</span></div><div class="rail"><i class="rail-center"></i><i class="rail-target" style="left:16%"></i><i class="rail-target" style="left:78%;background:#b895ff"></i></div><div class="rail-labels"><span>15 mm exclusion band</span><span>goal target range: ±48 mm</span><span>15 mm exclusion band</span></div></div>
      <p class="small muted">The success check requires two continuous seconds within 5 mm of a target. The task config requests at least 2.5 seconds dwell before advancing.</p>
      <div class="head-actions"><button class="button" data-go="learning">Explore this experiment →</button></div>
    </section>
  </div>
  <div class="section-kicker">PROJECT COVERAGE · ${ctx.catalog.subsystems.length} AREAS</div>
  <div class="subsystem-grid">${cards}</div>
  <div class="section-kicker">QUICK LINKS</div>
  <div class="grid cols-2">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Recorded evidence</h2><p>Benchmarks and reports committed with the project source</p></div></div><div class="evidence-list">${ctx.evidence.filter(e=>e.available).slice(0,5).map(e=>`<a class="evidence-row" href="/api/project/evidence/${e.id}/source" target="_blank" rel="noopener"><span><b>${e.name}</b><small>${e.kind}</small></span>${badge('Available','recorded')}</a>`).join('')}</div><div class="head-actions" style="margin-top:12px"><button class="button" data-go="project-map">Browse all sources →</button></div></section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Project documents</h2><p>Core definitions and implementation plans</p></div></div><div class="source-pills">${sourceLink(ctx,'project-overview')}${sourceLink(ctx,'math-model')}${sourceLink(ctx,'hardware')}${sourceLink(ctx,'firmware')}${sourceLink(ctx,'openmodelica')}${sourceLink(ctx,'phase5-plan')}</div><div class="rule"></div><div class="notice"><b>Hardware state:</b> design documented; no physical rig has been calibrated or validated. Live values in the Rig simulator come from the Python simulation.</div></section>
  </div>`;
}

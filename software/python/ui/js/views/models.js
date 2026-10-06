import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt, esc} from '../shared.js';

function renderConvergence(ctx,data) {
  const convergence=data.timestepConvergence||{},models=convergence.models||[];
  const scenarios=[...new Set(models.flatMap(model=>(model.scenarios||[]).map(scenario=>scenario.name)))];
  const limits=convergence.acceptanceLimits||{};
  const stateLimit=limits.max_normalized_state_error_rms,energyLimit=limits.max_relative_energy_drift_free_case;
  const rows=models.flatMap(model=>(model.scenarios||[]).flatMap(scenario=>(scenario.steps||[]).map(step=>({model,scenario,step}))))
    .sort((a,b)=>a.scenario.name.localeCompare(b.scenario.name)||a.model.name.localeCompare(b.model.name)||a.step.physicsDtSeconds-b.step.physicsDtSeconds);
  const modelName=model=>{
    const mode=model.springMode||model.name;
    return mode.includes('two_spring')||mode==='geometric'?'Explicit two-spring geometry':mode==='equivalent_torsion'?'Equivalent torsion':mode.replaceAll('_',' ');
  };
  const modelColor=model=>(model.springMode||model.name).includes('two_spring')||model.name==='geometric'?'#f1b36c':(model.springMode||model.name)==='equivalent_torsion'?COLORS.mpc:COLORS.other;
  const draw=()=>{
    if(!ctx.isCurrent())return;
    const scenarioName=document.getElementById('physicsScenario')?.value||scenarios[0];
    const series=models.map(model=>{
      const scenario=(model.scenarios||[]).find(item=>item.name===scenarioName),steps=(scenario?.steps||[]).slice().sort((a,b)=>a.physicsDtSeconds-b.physicsDtSeconds);
      return {x:steps.map(step=>step.physicsDtSeconds),y:steps.map(step=>step.maxNormalizedStateErrorRms),name:modelName(model),mode:'lines+markers',line:{color:modelColor(model)}};
    });
    plot('physicsStateErrorChart',series,{xaxis:{title:'Physics timestep [s]',type:'log'},yaxis:{title:'Maximum normalized state error RMS',type:'log'},shapes:Number.isFinite(stateLimit)?[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:stateLimit,y1:stateLimit,line:{color:'#ff8787',dash:'dash',width:1.5}}]:[]});
    const energySeries=models.map(model=>{
      const scenario=(model.scenarios||[]).find(item=>item.name===scenarioName),steps=(scenario?.steps||[]).slice().sort((a,b)=>a.physicsDtSeconds-b.physicsDtSeconds);
      return {x:steps.map(step=>step.physicsDtSeconds),y:steps.map(step=>step.maxRelativeEnergyDrift),name:modelName(model),mode:'lines+markers',line:{color:modelColor(model)}};
    });
    plot('physicsEnergyDriftChart',energySeries,{xaxis:{title:'Physics timestep [s]',type:'log'},yaxis:{title:'Maximum relative energy drift',type:'log'},shapes:scenarioName==='free'&&Number.isFinite(energyLimit)?[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:energyLimit,y1:energyLimit,line:{color:'#ff8787',dash:'dash',width:1.5}}]:[]});
    const table=document.getElementById('physicsConvergenceRows');
    if(table)table.innerHTML=rows.filter(({scenario})=>scenario.name===scenarioName).map(({model,scenario,step})=>{
      const statePass=Number.isFinite(stateLimit)&&step.maxNormalizedStateErrorRms<=stateLimit;
      const energyPass=scenario.name==='free'&&Number.isFinite(energyLimit)?step.maxRelativeEnergyDrift<=energyLimit:null;
      return `<tr><td>${esc(modelName(model))}</td><td>${esc(scenario.name)}</td><td>${fmt(step.physicsDtSeconds,5)}</td><td>${fmt(step.maxNormalizedStateErrorRms,7)}</td><td>${Number.isFinite(stateLimit)?fmt(stateLimit,7):'—'}</td><td>${statePass?'Pass':'Fail'}</td><td>${fmt(step.maxRelativeEnergyDrift,7)}</td><td>${energyPass===null?'—':energyPass?'Pass':'Fail'}</td><td>${fmt(model.validatedMaxPhysicsDtSeconds,5)}</td></tr>`;
    }).join('')||'<tr><td colspan="9">No timestep convergence rows are present in the report.</td></tr>';
  };
  const select=document.getElementById('physicsScenario');
  if(select){select.innerHTML=scenarios.map(name=>`<option value="${esc(name)}">${esc(name)}</option>`).join('');select.addEventListener('change',draw);}
  if(!models.length){
    for(const id of ['physicsStateErrorChart','physicsEnergyDriftChart']){const node=document.getElementById(id);if(node)node.innerHTML='<div class="empty-state">Timestep convergence evidence is unavailable.</div>';}
    return;
  }
  draw();
}

export async function renderModels(ctx) {
  const evidence=ctx.evidence.find(x=>x.id==='physics-validation');
  ctx.mounts.push(async()=>{
    try {
      const data=await getEvidence(ctx,'physics-validation');
      if(!ctx.isCurrent())return;
      const curves=data.springModelComparison?.torque_comparison||[];
      if(curves.length){
        plot('springModelChart',[
          {x:curves.map(r=>r.angle_deg),y:curves.map(r=>r.geometric_torque_nm),name:'Geometric two-spring torque',mode:'lines+markers',line:{color:COLORS.mpc}},
          {x:curves.map(r=>r.angle_deg),y:curves.map(r=>r.equivalent_torque_nm),name:'Equivalent torsion torque',mode:'lines+markers',line:{color:COLORS.lqr}},
        ],{xaxis:{title:'Resonator angle [°]'},yaxis:{title:'Spring torque [N·m]'}});
      } else {
        const node=document.getElementById('springModelChart');if(node)node.innerHTML='<div class="empty-state">Spring comparison data is not present in this report.</div>';
      }
      renderConvergence(ctx,data);
    } catch {
      if(!ctx.isCurrent())return;
      for(const id of ['springModelChart','physicsStateErrorChart','physicsEnergyDriftChart']){const node=document.getElementById(id);if(node)node.innerHTML='<div class="empty-state">Physics validation evidence is unavailable.</div>';}
      const table=document.getElementById('physicsConvergenceRows');if(table)table.innerHTML='<tr><td colspan="9">Physics validation evidence could not be loaded.</td></tr>';
    }
  });
  return `${pageHead('MODELS','Digital twin','Inspect how the reduced control model, nonlinear Python plant, and OpenModelica model relate. Placeholder parameters and uncalibrated values are identified explicitly.',`<a class="button quiet" href="/api/project/source/openmodelica" target="_blank" rel="noopener">Open Modelica notes ↗</a>`)}
    <div class="grid cols-3">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Reduced dynamics</h2><p>Used for feedback control and analysis</p></div>${badge('Source model','source')}</div><div class="metric-strip">${badge('q = [x, θ]','source')}${badge('base + resonator','source')}</div><p class="small muted">Captures base motion coupled to the compliant resonator. Supports reduced LQR and estimation work.</p>${sourceNotes(ctx,['math-model'])}</section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Nonlinear Python plant</h2><p>Interactive simulator and RL environment source</p></div>${badge('Live simulation','live')}</div><div class="metric-strip">${badge('7-state model','live')}${badge('motor + torque lag','live')}${badge('belt elasticity','live')}</div><p class="small muted">Models motor inertia, torque limits, belt compliance, friction, travel stops, sensors, and resonator coupling.</p>${sourceNotes(ctx,['python-plant','browser-ui'])}</section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>OpenModelica</h2><p>Equation-based twin and example systems</p></div>${badge('Not rig-calibrated','planned')}</div><div class="metric-strip">${badge('Stepper / transmission','source')}${badge('Sensors','source')}${badge('Active damping examples','source')}</div><p class="small muted">The geometric spring and material values are still placeholders. TMC2209 chopper behavior and missed steps are not modeled.</p>${sourceNotes(ctx,['openmodelica','openmodelica-model','modelica-parameters','modelica-examples','physics-validation'])}</section>
    </div>
    <div class="grid cols-2" style="margin-top:14px">
      <section class="panel"><div class="panel-pad"><div class="panel-head"><div><h2>Spring model comparison</h2><p>Stored physics validation · placeholder OpenModelica geometry</p></div>${evidence?.available?badge('Recorded','recorded'):badge('Unavailable','planned')}</div></div><div id="springModelChart" class="chart"></div></section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Calibration status</h2><p>Parameter source and validation state</p></div>${badge('Physical fit pending','planned')}</div>
        <div class="table-wrap"><table class="data-table"><thead><tr><th>Parameter / quantity</th><th>Value in current report</th><th>State</th></tr></thead><tbody>
        <tr><td>Model geometric spring stiffness</td><td>−0.5491 N·m/rad</td><td class="bad">Placeholder setup unstable upright</td></tr>
        <tr><td>Equivalent torsion stiffness</td><td>0.3082 N·m/rad</td><td>Simulation parameter</td></tr>
        <tr><td>OpenModelica anchor/rate values</td><td>documented in validation JSON</td><td class="bad">Placeholder</td></tr>
        <tr><td>Assembled spring / damping / friction</td><td>—</td><td class="bad">No rig measurement yet</td></tr>
        </tbody></table></div>
        <div class="notice" style="margin-top:13px"><b>Model difference is visible:</b> geometric and equivalent spring models have opposite torque signs in the recorded comparison. The report traces this to placeholder geometry; do not treat it as calibrated hardware behavior.</div>
        ${sourceNotes(ctx,['math-model','physics-validation'],['physics-validation'])}
      </section>
    </div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Physics timestep convergence</h2><p>Recorded integration error against the report’s acceptance limits; dashed lines mark the threshold where applicable.</p></div>${badge(evidence?.available?'Recorded validation':'Unavailable','recorded')}</div>
      <label class="toolbar small muted">Scenario <select id="physicsScenario" class="selector"></select></label>
      <div class="grid cols-2"><section><h3>Normalized state error</h3><div id="physicsStateErrorChart" class="chart"></div></section><section><h3>Relative energy drift</h3><div id="physicsEnergyDriftChart" class="chart"></div></section></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Model</th><th>Scenario</th><th>Timestep [s]</th><th>State error RMS</th><th>Limit</th><th>State gate</th><th>Energy drift</th><th>Free energy gate</th><th>Validated max dt [s]</th></tr></thead><tbody id="physicsConvergenceRows"></tbody></table></div>
      ${sourceNotes(ctx,[],['physics-validation'])}
    </section>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>End-to-end model path</h2><p>From geometry and actuator limits to simulated state and control</p></div></div>
      <div class="flow-row"><div class="flow-card"><b>Plant parameters</b><span>mass · spring · damping · belt</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Nonlinear plant</b><span>7-state ODE / RK integration</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Sensors & observer</b><span>encoder · gyro · estimator</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Controller</b><span>servo · LQR · MPC · residual PPO</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Actuator</b><span>torque and STEP/DIR models</span></div></div>
    </section>`;
}

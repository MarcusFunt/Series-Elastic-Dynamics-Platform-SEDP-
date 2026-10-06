import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt} from '../shared.js';

export async function renderModels(ctx) {
  const evidence=ctx.evidence.find(x=>x.id==='physics-validation');
  const subsys=ctx.catalog.subsystems.find(x=>x.id==='digital-twin');
  ctx.mounts.push(async()=>{
    try {
      const data=await getEvidence(ctx,'physics-validation');
      const curves=data.springModelComparison.torque_comparison||[];
      plot('springModelChart',[
        {x:curves.map(r=>r.angle_deg),y:curves.map(r=>r.geometric_torque_nm),name:'Geometric two-spring torque',mode:'lines+markers',line:{color:COLORS.mpc}},
        {x:curves.map(r=>r.angle_deg),y:curves.map(r=>r.equivalent_torque_nm),name:'Equivalent torsion torque',mode:'lines+markers',line:{color:COLORS.lqr}},
      ],{xaxis:{title:'Resonator angle [°]'},yaxis:{title:'Spring torque [N·m]'}});
    } catch { const node=document.getElementById('springModelChart');if(node)node.innerHTML='<div class="empty-state">Physics comparison evidence is unavailable.</div>'; }
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
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>End-to-end model path</h2><p>From geometry and actuator limits to simulated state and control</p></div></div>
      <div class="flow-row"><div class="flow-card"><b>Plant parameters</b><span>mass · spring · damping · belt</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Nonlinear plant</b><span>7-state ODE / RK integration</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Sensors & observer</b><span>encoder · gyro · estimator</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Controller</b><span>servo · LQR · MPC · residual PPO</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>Actuator</b><span>torque and STEP/DIR models</span></div></div>
    </section>`;
}

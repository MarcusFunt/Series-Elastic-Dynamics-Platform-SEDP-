import {badge, pageHead, sourceNotes, getEvidence, plot, COLORS, fmt, esc} from '../shared.js';

const controllerName={policy:'PPO residual',ppo:'PPO residual',lqr:'Constrained LQR',mpc:'Repository MPC'};
const colorFor={...COLORS,policy:COLORS.ppo,ppo:COLORS.ppo,lqr:COLORS.lqr,mpc:COLORS.mpc,teacher:COLORS.mpc,student:COLORS.ppo};
const formatPercent=x=>`${(Number(x||0)*100).toFixed(1)}%`;
const ablationConfigName=run=>typeof run.configuration==='string'?run.configuration:run.configuration?.name||'Unnamed configuration';
const jsonDetails=(data,label='Show saved source JSON')=>`<details><summary class="small">${esc(label)}</summary><pre class="source-preview">${esc(JSON.stringify(data,null,2))}</pre></details>`;

function railMarkup(targets=[]) {
  return `<div class="rail-wrap"><div class="rail-labels"><span>−65 mm</span><span>0</span><span>+65 mm</span></div><div class="rail"><i class="rail-center"></i>${targets.map((x,i)=>`<i class="rail-target" title="Goal ${i+1}: ${fmt(x,1)} mm" style="left:${((x+65)/130*100).toFixed(2)}%;background:${i%2?'#c09cff':'#68d7df'}"></i>`).join('')}</div><div class="rail-labels"><span>15 mm exclusion</span><span>Allowed target interval: −48 to +48 mm</span><span>15 mm exclusion</span></div></div>`;
}

function drawRuns(data,metricKey) {
  const rows=data.rows||[],controllers=[...new Set(rows.map(r=>r.controller))];
  const seeds=[...new Set(rows.map(r=>r.seed))].sort((a,b)=>a-b);
  plot('goalMetricChart',controllers.map(controller=>({
    x:seeds.map(String),y:seeds.map(seed=>rows.find(r=>r.controller===controller&&Number(r.seed)===seed)?.[metricKey]??null),
    name:controllerName[controller]||controller,type:'bar',marker:{color:colorFor[controller]||COLORS.other},
  })),{barmode:'group',xaxis:{title:data.id==='goal-hold-confirmation'?'Confirmation seed':'Development seed'},yaxis:{title:metricKey==='integrated_resonator_energy_mJs'?'Integrated modeled resonator energy [mJ·s]':metricKey==='position_rmse_mm'?'Position RMSE [mm]':'Peak angle [°]'}});
}

function renderGoalHold(ctx,data) {
  const summary=data.summary||{},ppo=summary.policy||summary.ppo||{},lqr=summary.lqr||{},mpc=summary.mpc||{};
  const rows=data.rows||[],policyRows=rows.filter(r=>['policy','ppo'].includes(r.controller));
  const seed=policyRows[0]?.seed,seeds=[...new Set(rows.map(row=>row.seed))].sort((a,b)=>a-b);
  const target=policyRows[0]?.goal_targets_mm||[];
  const completedHolds=policyRows.reduce((total,row)=>total+(row.completed_two_second_holds||0),0);
  const goalCount=policyRows.reduce((total,row)=>total+(row.goal_count||0),0);
  const exclusionIncursions=rows.reduce((total,row)=>total+(row.exclusion_zone_violations||0),0);
  const promotion=data.promotion||{},energyRatio=promotion.policy_lqr_integrated_resonator_energy_ratio??(ppo.mean_integrated_resonator_energy_mJs/lqr.mean_integrated_resonator_energy_mJs);
  const mpcRatio=promotion.policy_mpc_integrated_resonator_energy_ratio??(ppo.mean_integrated_resonator_energy_mJs/mpc.mean_integrated_resonator_energy_mJs);
  const isConfirmation=data.id==='goal-hold-confirmation',studyName=isConfirmation?'Independent confirmation':'Development study';
  const accepted=Boolean(promotion.accepted_vs_lqr);
  const controllers=[...new Set(rows.map(r=>r.controller))];
  const selectedController=controllers.find(value=>['policy','ppo'].includes(value))||controllers[0]||'policy';
  return `<div class="grid cols-4">
    <section class="panel stat-card"><div class="stat-label">PPO vs LQR · paired energy ratio</div><div class="stat-value">${Number.isFinite(energyRatio)?`${energyRatio<1?'−':'+'}${Math.abs((1-energyRatio)*100).toFixed(1)}%`: '—'}</div><div class="stat-sub">${studyName.toLowerCase()} promotion record</div></section>
    <section class="panel stat-card"><div class="stat-label">PPO vs LQR · tracking</div><div class="stat-value">${Number.isFinite(ppo.mean_position_rmse_mm)&&Number.isFinite(lqr.mean_position_rmse_mm)?`${((1-ppo.mean_position_rmse_mm/lqr.mean_position_rmse_mm)*100).toFixed(1)}%`: '—'}</div><div class="stat-sub">paired mean position RMSE improvement</div></section>
    <section class="panel stat-card"><div class="stat-label">PPO continuous holds</div><div class="stat-value">${completedHolds} / ${goalCount||'—'}</div><div class="stat-sub">two-second in-tolerance holds</div></section>
    <section class="panel stat-card"><div class="stat-label">Exclusion-zone incursions</div><div class="stat-value">${exclusionIncursions}</div><div class="stat-sub">across paired controller runs</div></section>
  </div>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Paired controller results</h2><p>${studyName} · same randomized goals and disturbances</p></div></div><div class="metric-strip">${badge(accepted?'PPO promoted vs LQR':'PPO not promoted vs LQR',accepted?'recorded':'warn')}${badge('MPC remains stronger on this task','warn')}</div><div id="goalMetricChart" class="chart"></div></section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Random destination task</h2><p>Move, settle, then continue after the dwell</p></div></div>
      <label class="toolbar small muted">Controller <select id="goalController" class="selector">${controllers.map(value=>`<option value="${esc(value)}" ${value===selectedController?'selected':''}>${esc(controllerName[value]||value)}</option>`).join('')}</select></label>
      <label class="toolbar small muted" style="margin-top:9px">${isConfirmation?'Confirmation':'Development'} seed <select id="goalSeed" class="selector">${seeds.map(s=>`<option value="${s}" ${s===seed?'selected':''}>${s}</option>`).join('')}</select></label>
      <div id="targetRail">${railMarkup(target)}</div>
      <div class="metric-strip">${badge('One-second minimum-jerk move','source')}${badge(`${data.task?.configuredMinimumHoldSeconds??'—'} s configured dwell`,'source')}${badge(`±${data.holdToleranceMm??'—'} mm tolerance`,'source')}</div>
      <p class="small muted" id="goalSequenceText">Target sequence is shown for the selected controller and seed.</p>
    </section>
  </div>
  <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Per-seed evidence</h2><p>Stored rows from all ${seeds.length} ${isConfirmation?'confirmation':'development'} seeds</p></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Controller</th><th>Seed</th><th>Energy [mJ·s]</th><th>Position RMSE [mm]</th><th>Angle RMS [°]</th><th>Peak angle [°]</th><th>Goals completed</th><th>Hold fraction</th><th>Zone violations</th><th>MPC p95 [ms]</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${controllerName[r.controller]||esc(r.controller)}</td><td>${r.seed}</td><td>${fmt(r.integrated_resonator_energy_mJs,5)}</td><td>${fmt(r.position_rmse_mm,4)}</td><td>${fmt(r.angle_rms_deg,4)}</td><td>${fmt(r.peak_angle_deg,4)}</td><td>${r.completed_two_second_holds??'—'} / ${r.goal_count??'—'}</td><td>${formatPercent(r.two_second_hold_fraction)}</td><td>${r.exclusion_zone_violations??'—'}</td><td>${r.mpc_solve_p95_ms==null?'—':fmt(r.mpc_solve_p95_ms,2)}</td></tr>`).join('')}</tbody></table></div></section>
  <div class="grid cols-2" style="margin-top:14px">
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Promotion record</h2><p>Thresholds and gate outcomes from this evaluation</p></div>${badge(accepted?'Accepted vs LQR':'Not accepted vs LQR',accepted?'recorded':'warn')}</div>
      <div class="metric-strip">${badge('≥5% energy improvement','source')}${badge('Paired tracking gate','source')}${badge('Peak-angle gate','source')}${badge('Safety & exclusion gate','source')}${badge('Two-second holds','source')}</div>
      <div class="rule"></div><div class="table-wrap"><table class="data-table"><tbody><tr><td>Paired PPO / LQR integrated resonator energy ratio</td><td>${fmt(energyRatio*100,2)}%</td></tr><tr><td>Paired PPO / MPC integrated resonator energy ratio</td><td>${fmt(mpcRatio*100,2)}%</td></tr><tr><td>Passed gates vs LQR</td><td>${accepted?'Yes':'No'}</td></tr><tr><td>Passed gates vs MPC</td><td>${Number.isFinite(mpcRatio)&&mpcRatio<1?'PPO energy lower; no separate promotion record':'No; PPO energy is higher'}</td></tr></tbody></table></div>
      <p class="small muted">The experiment’s reported “energy” is integrated resonator kinetic-plus-spring energy. It measures vibration, not electrical consumption or total motor energy. ${Number.isFinite(mpc.mean_integrated_resonator_energy_mJs)?'MPC records the lowest mean vibration energy and tracking RMSE of these three controllers.':''}</p>
    </section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Run provenance</h2><p>Reproducible study context and files</p></div></div>
      <div class="metric-strip">${badge(data.benchmarkVersion||'Unknown benchmark','recorded')}${badge('Source revision '+String(data.sourceRevision||'').slice(0,7),'source')}${badge(`${seeds.length} ${isConfirmation?'confirmation':'development'} seeds`,'recorded')}</div>
      <p class="small muted">Task: randomized goals with a ${data.task?.edgeExclusionMm??'—'} mm rail-edge exclusion band; targets sampled inside ±48 mm on a rail with ${data.task?.railHalfTravelMm??'—'} mm half-travel. Successful holds require two continuous seconds within tolerance; the environment config asks for at least ${data.task?.configuredMinimumHoldSeconds??'—'} seconds per dwell.</p>
      ${sourceNotes(ctx,['goal-hold-readme'],[data.id,'goal-hold-training','goal-hold-metadata'])}
    </section>
  </div>`;
}

function renderTeacherFit(data) {
  const losses=data.losses||[];
  const traces=[
    {x:losses.map(row=>row.epoch),y:losses.map(row=>row.train_action_mae),name:'Training action MAE',mode:'lines+markers',line:{color:COLORS.ppo}},
    {x:losses.map(row=>row.epoch),y:losses.map(row=>row.heldout_action_mae),name:'Held-out action MAE',mode:'lines+markers',line:{color:COLORS.mpc}},
  ];
  return `<section class="panel panel-pad"><div class="panel-head"><div><h2>MPC teacher fit</h2><p>Action imitation error over distillation epochs</p></div>${badge(`${losses.length} epochs`,'recorded')}</div>
    <div class="metric-strip">${badge(`Best held-out MAE ${fmt(data.best_heldout_action_mae,4)}`,'recorded')}${badge(`${fmt(data.train_samples,0)} training samples`,'source')}${badge(`${fmt(data.validation_samples,0)} validation samples`,'source')}${badge(`${Array.isArray(data.heldout_episodes)?data.heldout_episodes.length:fmt(data.heldout_episodes,0)} held-out episodes`,'source')}</div>
    <div id="teacherFitChart" class="chart"></div><div class="table-wrap"><table class="data-table"><thead><tr><th>Epoch</th><th>Train action MAE</th><th>Held-out action MAE</th></tr></thead><tbody>${losses.map(row=>`<tr><td>${row.epoch}</td><td>${fmt(row.train_action_mae,5)}</td><td>${fmt(row.heldout_action_mae,5)}</td></tr>`).join('')}</tbody></table></div>
    <p class="small muted">Fallback MPC labels excluded from the fit: ${fmt(data.excluded_fallback_labels,0)}.</p>${jsonDetails(data)}</section>`;
}

function teacherCondition(row) { return `${row.scenario||'scenario'} · seed ${row.seed??'—'}${row.randomized?' · randomized':''}`; }
function drawTeacherMetric(rows,field,id,label) {
  const conditions=[...new Map(rows.map(row=>[teacherCondition(row),row])).keys()];
  const controllers=[...new Set(rows.map(row=>row.controller))];
  plot(id,controllers.map(controller=>({
    x:conditions,y:conditions.map(condition=>rows.find(row=>row.controller===controller&&teacherCondition(row)===condition)?.[field]??null),
    name:controllerName[controller]||controller,type:'bar',marker:{color:colorFor[controller]||COLORS.other},
  })),{barmode:'group',xaxis:{title:'Scenario and evaluation seed'},yaxis:{title:label}});
}

function renderTeacherEvaluation(data) {
  const rows=data.rows||[],promotion=data.promotion||{};
  return `<div class="grid cols-2"><section class="panel panel-pad"><div class="panel-head"><div><h2>Teacher energy by condition</h2><p>Distilled controller compared with recorded baselines</p></div></div><div id="teacherEnergyChart" class="chart"></div></section>
    <section class="panel panel-pad"><div class="panel-head"><div><h2>Teacher tracking by condition</h2><p>Position RMSE retained per scenario and seed</p></div></div><div id="teacherTrackingChart" class="chart"></div></section></div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Teacher evaluation</h2><p>${data.benchmark_version||'Distillation evaluation'} · ${rows.length} recorded controller rows</p></div>${badge(promotion.accepted?'Accepted':'Not accepted',promotion.accepted?'recorded':'warn')}</div>
    <div class="metric-strip">${badge(`Mean energy ratio ${fmt(promotion.mean_energy_ratio,3)}`,'recorded')}${badge(`${(promotion.reasons||[]).length} gate reasons`,'warn')}</div>
    <div class="table-wrap"><table class="data-table"><thead><tr><th>Controller</th><th>Condition</th><th>Energy integral [mJ·s]</th><th>Position RMSE [mm]</th><th>Angle RMS [°]</th><th>Peak angle [°]</th><th>MPC solve p95 [ms]</th><th>MPC fallback</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${controllerName[row.controller]||esc(row.controller)}</td><td>${esc(teacherCondition(row))}</td><td>${fmt(row.energy_integral_mJs,5)}</td><td>${fmt(row.position_rmse_mm,4)}</td><td>${fmt(row.angle_rms_deg,4)}</td><td>${fmt(row.peak_angle_deg,4)}</td><td>${fmt(row.solve_p95_ms,2)}</td><td>${typeof row.mpc_fallback_fraction==='number'?formatPercent(row.mpc_fallback_fraction):'—'}</td></tr>`).join('')}</tbody></table></div>
    ${(promotion.reasons||[]).length?`<details><summary class="small">Show promotion gate reasons</summary><ul class="small muted">${promotion.reasons.map(reason=>`<li>${esc(reason)}</li>`).join('')}</ul></details>`:''}${jsonDetails(data)}</section>`;
}

function renderAblation(data) {
  const runs=Object.entries(data.runs||{}).map(([id,run])=>({id,run}));
  const first=runs[0]?.run,firstTraining=first?.training||[];
  const trainingKeys=['mean_reward','explained_variance','policy','value','entropy','kl','clipfrac'].filter(key=>firstTraining.some(row=>typeof row[key]==='number'));
  const evalKeys=['energy_integral_mJs','position_rmse_mm','angle_rms_deg','peak_angle_deg','completion_rate'].filter(key=>typeof first?.evaluation_summary?.aggregate?.policy?.[key]==='number');
  return `<section class="panel panel-pad"><div class="panel-head"><div><h2>PPO ablation study</h2><p>${esc(data.experiment||'Saved configuration and seed comparisons')} · recorded training and paired evaluation</p></div>${badge(`${runs.length} runs`,'recorded')}</div>
    ${runs.length?`<div class="controls-line"><label class="toolbar small muted">Run / configuration <select id="ablationRun" class="selector">${runs.map(({id,run})=>`<option value="${esc(id)}">${esc(ablationConfigName(run))} · seed ${esc(run.training_seed)}</option>`).join('')}</select></label><label class="toolbar small muted">Training metric <select id="ablationTrainingMetric" class="selector">${trainingKeys.map(key=>`<option value="${key}" ${key==='mean_reward'?'selected':''}>${key.replaceAll('_',' ')}</option>`).join('')}</select></label><label class="toolbar small muted">Evaluation metric <select id="ablationEvalMetric" class="selector">${evalKeys.map(key=>`<option value="${key}" ${key==='energy_integral_mJs'?'selected':''}>${key.replaceAll('_',' ')}</option>`).join('')}</select></label></div>
      <div id="ablationSummary" class="metric-strip" style="margin-top:12px"></div><div class="grid cols-2"><div><h3>Training history</h3><div id="ablationTrainingChart" class="chart"></div></div><div><h3>Paired evaluation</h3><div id="ablationEvalChart" class="chart"></div></div></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Scenario / case</th><th>Policy</th><th>LQR</th><th>Policy − LQR</th></tr></thead><tbody id="ablationCaseRows"></tbody></table></div><div id="ablationRunSource"></div>`:`<div class="empty-state">No ablation runs are present in this report.</div>`}
    ${jsonDetails({experiment:data.experiment,source_revision:data.source_revision,budget:data.budget,training_seeds:data.training_seeds,evaluation_seeds:data.evaluation_seeds,configurations:data.configurations})}</section>`;
}

function mountAblation(ctx,data) {
  const runSelect=document.getElementById('ablationRun'),trainMetric=document.getElementById('ablationTrainingMetric'),evalMetric=document.getElementById('ablationEvalMetric');
  if(!runSelect)return;
  const runs=data.runs||{};
  function update() {
    if(!ctx.isCurrent())return;
    const run=runs[runSelect.value];if(!run)return;
    const training=run.training||[],trainKey=trainMetric.value;
    plot('ablationTrainingChart',[{x:training.map(row=>row.steps),y:training.map(row=>row[trainKey]),name:trainKey.replaceAll('_',' '),mode:'lines+markers',line:{color:COLORS.ppo}}],{xaxis:{title:'Environment steps'},yaxis:{title:trainKey.replaceAll('_',' ')} });
    const metric=evalMetric.value,aggregate=run.evaluation_summary?.aggregate||{},policy=aggregate.policy||{},lqr=aggregate.lqr||{},cases=run.evaluation_summary?.case_metrics||[];
    const accepted=Boolean(run.promotion?.accepted);
    document.getElementById('ablationSummary').innerHTML=`${badge(ablationConfigName(run),'recorded')}${badge(`Training seed ${run.training_seed}`,'source')}${badge(accepted?'Promotion accepted':'Not promoted',accepted?'recorded':'warn')}${badge(`Energy ratio ${fmt(policy.energy_integral_mJs/lqr.energy_integral_mJs,3)}`,'recorded')}`;
    const labels=cases.map(row=>`${row.scenario} · ${row.evaluation_seed}${row.randomized?' · rand':''}`);
    plot('ablationEvalChart',[
      {x:labels,y:cases.map(row=>row.policy?.[metric]??null),name:'PPO policy',type:'bar',marker:{color:COLORS.ppo}},
      {x:labels,y:cases.map(row=>row.lqr?.[metric]??null),name:'LQR',type:'bar',marker:{color:COLORS.lqr}},
    ],{barmode:'group',xaxis:{title:'Scenario and evaluation seed'},yaxis:{title:metric.replaceAll('_',' ')} });
    const table=document.getElementById('ablationCaseRows');
    table.innerHTML=cases.map((row,index)=>{
      const p=row.policy?.[metric],b=row.lqr?.[metric],delta=typeof p==='number'&&typeof b==='number'?p-b:null;
      return `<tr><td>${esc(labels[index])}</td><td>${fmt(p,5)}</td><td>${fmt(b,5)}</td><td>${fmt(delta,5)}</td></tr>`;
    }).join('')||'<tr><td colspan="4">No paired evaluation cases are present.</td></tr>';
    document.getElementById('ablationRunSource').innerHTML=jsonDetails(run,'Show selected run records');
  }
  runSelect.addEventListener('change',update);trainMetric.addEventListener('change',update);evalMetric.addEventListener('change',update);update();
}

export async function renderLearning(ctx) {
  const ids=['goal-hold-confirmation','goal-hold-development','goal-hold-training','goal-hold-metadata','ppo-training','ppo-ablation','teacher-fit','teacher-40-epochs'];
  const items=ids.map(id=>ctx.evidence.find(e=>e.id===id)).filter(item=>item?.available);
  const videos=ctx.media.map(item=>`<section class="panel video-card"><div class="panel-title"><b>${esc(item.name)}</b>${badge(item.stage,'recorded')}</div>${item.available?`<video controls preload="metadata" src="${item.url}" aria-label="${esc(item.name)}"></video>`:`<div class="empty-state"><b>Video unavailable</b>Recorded file is not in this checkout.</div>`}<div class="panel-pad"><p class="small muted">${esc(item.description)}</p></div></section>`).join('');
  ctx.mounts.push(()=>{
    const select=document.getElementById('learningEvidence'),content=document.getElementById('learningDetail');
    if(!items.length){content.innerHTML='<div class="empty-state"><b>No saved evidence available</b>Evidence records are missing or unavailable in this checkout.</div>';return;}
    let detailGeneration=0;
    const mountDetail=async()=>{
      const generation=++detailGeneration;
      try {
        const id=select.value,data=await getEvidence(ctx,id);
        if(!ctx.isCurrent()||generation!==detailGeneration)return;
        if(id==='goal-hold-confirmation'||id==='goal-hold-development') {
          data.id=id;content.innerHTML=renderGoalHold(ctx,data);
          const seedSel=document.getElementById('goalSeed'),controllerSel=document.getElementById('goalController');
          drawRuns(data,'integrated_resonator_energy_mJs');
          const updateSequence=()=>{
            const row=data.rows.find(r=>r.seed===Number(seedSel.value)&&r.controller===controllerSel.value);
            document.getElementById('targetRail').innerHTML=railMarkup(row?.goal_targets_mm||[]);
            document.getElementById('goalSequenceText').textContent=row?`Targets for ${controllerName[row.controller]||row.controller}, seed ${row.seed}: ${(row.goal_targets_mm||[]).map(v=>`${fmt(v,1)} mm`).join(' · ')}. Completed ${row.completed_two_second_holds} of ${row.goal_count} in-tolerance holds.`:'No target sequence in this result.';
          };
          seedSel.addEventListener('change',updateSequence);controllerSel.addEventListener('change',updateSequence);updateSequence();
        } else if(id==='teacher-fit') {
          content.innerHTML=renderTeacherFit(data);
          plot('teacherFitChart',[
            {x:(data.losses||[]).map(row=>row.epoch),y:(data.losses||[]).map(row=>row.train_action_mae),name:'Training action MAE',mode:'lines+markers',line:{color:COLORS.ppo}},
            {x:(data.losses||[]).map(row=>row.epoch),y:(data.losses||[]).map(row=>row.heldout_action_mae),name:'Held-out action MAE',mode:'lines+markers',line:{color:COLORS.mpc}},
          ],{xaxis:{title:'Distillation epoch'},yaxis:{title:'Action mean absolute error'}});
        } else if(id==='teacher-40-epochs') {
          content.innerHTML=renderTeacherEvaluation(data);
          drawTeacherMetric(data.rows||[],'energy_integral_mJs','teacherEnergyChart','Integrated vibration energy [mJ·s]');
          drawTeacherMetric(data.rows||[],'position_rmse_mm','teacherTrackingChart','Position RMSE [mm]');
        } else if(id==='ppo-ablation') {
          content.innerHTML=renderAblation(data);mountAblation(ctx,data);
        } else if(Array.isArray(data)) {
          const points=data.filter(row=>row&&typeof row==='object');
          const keys=['mean_reward','explained_variance','action_mean_abs','kl','policy','value'].filter(key=>points.some(row=>typeof row[key]==='number'));
          if(!keys.length){content.innerHTML='<div class="empty-state"><b>No numeric training metrics</b>This saved history has no chartable PPO fields.</div>';return;}
          content.innerHTML=`<section class="panel panel-pad"><div class="panel-head"><div><h2>Saved training curve</h2><p>Metrics emitted by the recorded PPO updates; horizontal axis is environment steps.</p></div>${badge(`${points.length} updates`,'recorded')}</div><label class="toolbar small muted">Metric <select id="trainingMetric" class="selector">${keys.map(key=>`<option value="${key}" ${key==='mean_reward'?'selected':''}>${key.replaceAll('_',' ')}</option>`).join('')}</select></label><div id="savedTrainingChart" class="chart"></div>${jsonDetails(points,'Show saved update records')}</section>`;
          const metricSelect=document.getElementById('trainingMetric');
          const updateCurve=()=>plot('savedTrainingChart',[{x:points.map(row=>row.steps),y:points.map(row=>row[metricSelect.value]),name:metricSelect.value.replaceAll('_',' '),mode:'lines+markers',line:{color:COLORS.ppo}}],{xaxis:{title:'Environment steps'},yaxis:{title:metricSelect.value.replaceAll('_',' ')}});
          metricSelect.addEventListener('change',updateCurve);updateCurve();
        } else {
          content.innerHTML=`<section class="panel panel-pad"><div class="panel-head"><div><h2>${id==='goal-hold-metadata'?'Run configuration':'Historical experiment evidence'}</h2><p>Source JSON is shown as committed; machine-local artifact paths are filtered by the server.</p></div>${badge('Source data','recorded')}</div>${jsonDetails(data,'Show saved configuration')}</section>`;
        }
      } catch(err) {
        if(!ctx.isCurrent()||generation!==detailGeneration)return;
        content.innerHTML=`<div class="empty-state"><b>Evidence unavailable</b>${esc(err.message||err)}</div>`;
      }
    };
    select.addEventListener('change',mountDetail);mountDetail();
  });
  return `${pageHead('EXPERIMENT LIBRARY','Learning & experiments','Inspect versioned PPO studies, MPC teacher work, training progress, paired benchmark outcomes, and local v4 training controls. Historical results remain tied to their own task definitions.',`<a class="button primary" href="/live?view=training" target="_blank" rel="noopener">Start or evaluate a run ↗</a>`)}
    <div class="notice blue"><b>Scope of the highlighted result:</b> PPO passed its 5% vibration-energy gate against LQR in the random-goal task. Repository MPC also passes the holds and has lower mean modeled vibration energy and tracking RMSE; the evidence does not claim PPO beats MPC.</div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="controls-line"><div class="panel-head"><div><h2>Saved experiment</h2><p>Choose an evaluation, training history, or run configuration</p></div></div><label class="toolbar small muted">Evidence <select id="learningEvidence" class="selector">${items.map(item=>`<option value="${item.id}" ${item.id==='goal-hold-confirmation'?'selected':''}>${esc(item.name)}</option>`).join('')}</select></label></div><div id="learningDetail" style="margin-top:14px"></div></section>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Live v4 workflow</h2><p>Start training, MPC teacher collection, checkpoint evaluation, or a standalone MPC reference</p></div>${badge('Local job controls','live')}</div><p class="small muted">Training runs save progress and artifacts on the machine serving this page. The original controls include cancellation, promotion results, and checkpoint loading into the live simulator.</p><div class="head-actions"><a class="button primary" href="/live?view=training" target="_blank" rel="noopener">Open v4 training controls ↗</a></div></section>
    ${videos?`<div class="section-kicker">RANDOM-GOAL-AND-HOLD CHECKPOINT PLAYBACKS · ${ctx.media.filter(v=>v.available).length} LOCAL VIDEOS</div><div class="video-grid">${videos}</div>`:''}
    <div class="section-kicker">OTHER PROJECT EVIDENCE</div><div class="evidence-list">${ctx.evidence.filter(e=>e.available&&e.id!=='goal-hold-confirmation').map(item=>`<a class="evidence-row" href="/api/project/evidence/${item.id}/source" target="_blank" rel="noopener"><span><b>${esc(item.name)}</b><small>${esc(item.kind)} · ${esc(item.dataType)}</small></span>${badge('Open source','recorded')}</a>`).join('')}</div>`;
}

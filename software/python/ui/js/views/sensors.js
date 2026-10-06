import {badge, pageHead, sourceNotes, getEvidence, fmt} from '../shared.js';

const freq = seconds => typeof seconds==='number'&&seconds>0?`${(1/seconds).toFixed(0)} Hz`:'—';

export async function renderSensors(ctx) {
  const item=ctx.evidence.find(e=>e.id==='goal-hold-confirmation');
  ctx.mounts.push(async()=>{
    const select=document.getElementById('sensorEvidence');
    const table=document.getElementById('sensorTimingRows');
    async function update(){
      const data=await getEvidence(ctx,select.value);
      const channels=data.task?.sensorTiming||{};
      table.innerHTML=Object.entries(channels).map(([name,value])=>`<tr><td>${name.replace('_',' ')}</td><td>${freq(value.sample_period)}</td><td>${fmt(value.sample_period*1000,1)}</td><td>${fmt(value.delay*1000,1)}</td><td>${fmt(value.jitter*1000,1)}</td><td>${fmt(value.dropout_probability*100,2)}%</td></tr>`).join('')||'<tr><td colspan="6">No sensor timing in this evidence item.</td></tr>';
    }
    select.addEventListener('change',update);update();
  });
  return `${pageHead('RUNTIME PATH','Sensors & timing','Follow simulated measurements through timestamping and estimation into control. Planned hardware sample rates are shown separately from saved simulation configuration.',`<a class="button quiet" href="/api/project/source/async-timing" target="_blank" rel="noopener">Read timing model ↗</a>`)}
    <div class="grid cols-2">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Measurement and estimate flow</h2><p>Firmware target architecture; live bench channels are unavailable</p></div>${badge('Architecture view','source')}</div>
        <div class="flow-vertical">
          <div class="flow-card"><b>Two AS5600 encoders</b><span>motor angle · resonator angle</span></div><div class="flow-arrow">↓</div>
          <div class="flow-card"><b>6-axis IMU</b><span>gyro and specific force; acceleration includes base motion</span></div><div class="flow-arrow">↓</div>
          <div class="flow-card"><b>Timestamped asynchronous channels</b><span>sample time · delay · jitter · dropout</span></div><div class="flow-arrow">↓</div>
          <div class="flow-card"><b>Angle unwrap + state estimator</b><span>encoder / gyro fusion; EKF supports delayed measurements</span></div><div class="flow-arrow">↓</div>
          <div class="flow-card"><b>State and force estimate</b><span>position · velocity · resonator state · elastic force</span></div>
        </div>${sourceNotes(ctx,['firmware','hardware','async-timing','residual-v4'])}
      </section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Firmware rate targets</h2><p>Design guidance from firmware notes · not measured timing</p></div>${badge('Planned rates','planned')}</div>
        <div class="table-wrap"><table class="data-table"><thead><tr><th>Pipeline</th><th>Target rate</th><th>Source status</th></tr></thead><tbody>
        <tr><td>IMU sampling</td><td>~2–4 kHz</td><td>Firmware proposal</td></tr><tr><td>State prediction</td><td>~2–4 kHz</td><td>Firmware proposal</td></tr><tr><td>Inner loop</td><td>~1–2 kHz</td><td>Firmware proposal</td></tr><tr><td>AS5600 correction</td><td>~1 kHz starting point</td><td>Firmware proposal</td></tr><tr><td>LQR / haptics</td><td>~500 Hz–1 kHz</td><td>Firmware proposal</td></tr><tr><td>MPC</td><td>~100–500 Hz</td><td>Firmware proposal</td></tr><tr><td>RL inference</td><td>hundreds of Hz–~1 kHz</td><td>Firmware proposal</td></tr>
        </tbody></table></div>
        <div class="notice" style="margin-top:12px"><b>Core 0 safety rule:</b> remain safe when Core 1 stalls. Avoid blocking logging and USB work on the fast path.</div>${sourceNotes(ctx,['firmware'])}
      </section>
    </div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="controls-line"><div><div class="panel-head"><div><h2>Saved sensor timing configuration</h2><p>Training environment config, not physical sensor measurements</p></div></div></div><label class="toolbar small muted">Evidence <select id="sensorEvidence" class="selector"><option value="goal-hold-confirmation">Random-goal confirmation</option><option value="goal-hold-development">Random-goal development</option></select></label></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Channel</th><th>Sample rate</th><th>Period [ms]</th><th>Delay [ms]</th><th>Jitter [ms]</th><th>Dropout probability</th></tr></thead><tbody id="sensorTimingRows"></tbody></table></div>
      <div class="notice blue" style="margin-top:12px">The saved training run uses simulated timing/noise. These values describe that run’s config and must not be read as bench telemetry.</div>
      ${sourceNotes(ctx,['goal-hold-readme'],['goal-hold-confirmation'])}
    </section>`;
}

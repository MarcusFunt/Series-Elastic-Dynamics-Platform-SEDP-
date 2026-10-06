import {badge, pageHead, sourceNotes} from '../shared.js';

export async function renderHardware(ctx) {
  return `${pageHead('PHYSICAL SYSTEM DESIGN','Hardware & firmware','Review the planned mechanism, electronics, RP2350 workload split, motion generation, safety chain, and the present build/validation state.',`<a class="button quiet" href="/api/project/source/hardware" target="_blank" rel="noopener">Open hardware notes ↗</a>`)}
    <div class="notice"><b>Build state:</b> mechanical and electronics architecture is documented, but no physical rig has been assembled and validated. There is no hardware telemetry to plot yet.</div>
    <div class="grid cols-2" style="margin-top:14px">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Mechanical load path</h2><p>Series-elastic resonator mounted on the driven carriage</p></div>${badge('Design','planned')}</div>
        <div class="flow-row"><div class="flow-card"><b>NEMA 17</b><span>stepper motor</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>GT2 drive</b><span>belt · nominal 20T pulley</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>MGN7 carriage</b><span>about 150 mm rail travel</span></div></div>
        <div style="text-align:center;color:#7590a1;padding:9px">↳ compliant one-axis resonator · two diagonal springs · interchangeable mass/spring geometry</div>
        <div class="metric-strip">${badge('2020 extrusion','source')}${badge('TMC2209 driver','source')}${badge('AS5600 motor encoder','source')}${badge('AS5600 resonator encoder','source')}${badge('6-axis IMU','source')}</div>${sourceNotes(ctx,['hardware','system-model'])}
      </section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Electrical and sensor map</h2><p>Proposed buses and embedded board</p></div>${badge('Not connected','planned')}</div>
        <div class="flow-vertical"><div class="flow-card"><b>24 V input</b><span>stepper driver supply · 5 V buck · 3.3 V logic</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>RP2350 controller</b><span>I²C0 motor AS5600 · I²C1 resonator AS5600 · SPI IMU</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>PIO STEP/DIR</b><span>TMC2209 → NEMA17 → GT2 belt</span></div></div>
        <p class="small muted">GPIO travel-limit switches · UART driver diagnostics · USB telemetry · SWD debug.</p>${sourceNotes(ctx,['hardware','firmware'])}
      </section>
    </div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>Firmware execution split</h2><p>Deterministic motion and safety remain on the fast side of the architecture</p></div>${badge('Design proposal','source')}</div>
      <div class="architecture-grid">
        <div class="architecture-card"><h3>PIO / peripherals</h3><ul><li>Event-driven STEP/DIR timing</li><li>Timestamped sensor snapshots</li><li>Independent travel-limit inputs</li></ul></div>
        <div class="architecture-card"><h3>Core 0 · deterministic fast path</h3><ul><li>Angle unwrap and sensor fusion</li><li>State prediction and elastic force model</li><li>Inner motion loop and safety checks</li><li>Queue motor commands to PIO</li></ul></div>
        <div class="architecture-card"><h3>Core 1 · supervisory tasks</h3><ul><li>Trajectory generation and control</li><li>LQR / MPC / learned inference</li><li>Haptics and system identification</li><li>USB, experiment sequencing, logging</li></ul></div>
      </div>
      <div class="rule"></div><div class="panel-head"><div><h2>Safety path</h2><p>Firmware design requires independent mechanical and software safeguards</p></div></div>
      <div class="safety-list"><div class="safety-step">Mechanical end stops limit travel if control fails.</div><div class="safety-step">Independent left/right travel switches report rail limits.</div><div class="safety-step">Software braking envelope estimates stopping distance before commanding motion.</div><div class="safety-step">Commanded-versus-measured motor position detects tracking faults.</div><div class="safety-step">Watchdog leaves Core 0 safe if supervisory Core 1 stalls.</div></div>
      ${sourceNotes(ctx,['firmware','hardware'])}
    </section>
    <div class="grid cols-3" style="margin-top:14px">
      <section class="panel panel-pad"><div class="stat-label">Physical build</div><div class="stat-value" style="font-size:17px">Not validated</div><div class="stat-sub">No bench measurements are committed.</div></section>
      <section class="panel panel-pad"><div class="stat-label">Calibration</div><div class="stat-value" style="font-size:17px">Awaiting rig</div><div class="stat-sub">Spring, belt, friction, and latency values are provisional.</div></section>
      <section class="panel panel-pad"><div class="stat-label">Telemetry</div><div class="stat-value" style="font-size:17px">Unavailable</div><div class="stat-sub">All live values currently come from Python simulation.</div></section>
    </div>`;
}

export async function renderHaptics(ctx) {
  return `${pageHead('APPLICATIONS & IDENTIFICATION','Haptics & system identification','See how the same compliant mechanism can sense force and render a virtual feel, and how physical measurements will update the digital twin.',`<a class="button quiet" href="/api/project/source/hardware" target="_blank" rel="noopener">Read calibration procedure ↗</a>`)}
    <div class="grid cols-2">
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Elastic force-feedback path</h2><p>Spring deformation is treated as a calibrated force signal</p></div>${badge('Design concept','planned')}</div>
        <div class="flow-vertical"><div class="flow-card"><b>User or external load deflects resonator</b><span>relative displacement from encoder/IMU estimates</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Calibrated spring geometry</b><span>force estimate accounts for assembled springs and friction</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Requested virtual / remote force</b><span>compare desired feel with measured elastic force</span></div><div class="flow-arrow">↓</div><div class="flow-card"><b>Motor position / torque response</b><span>render the selected haptic effect through the carriage</span></div></div>
        <div class="notice" style="margin-top:13px"><b>Calibration is pending:</b> the repo describes applying known weights and fitting the assembled force/torque relationship. No physical calibration samples are available.</div>${sourceNotes(ctx,['project-overview','hardware','system-model'])}
      </section>
      <section class="panel panel-pad"><div class="panel-head"><div><h2>Haptic modes in the concept</h2><p>Different force laws can share the same elastic sensor path</p></div>${badge('Physical behavior unverified','planned')}</div>
        <div class="grid cols-2"><div class="architecture-card"><h3>Virtual centering</h3><p class="small muted">Return force toward a reference position.</p></div><div class="architecture-card"><h3>Damping</h3><p class="small muted">Resist motion with velocity-dependent feedback.</p></div><div class="architecture-card"><h3>Detents</h3><p class="small muted">Render tactile positions or mode boundaries.</p></div><div class="architecture-card"><h3>Virtual stops</h3><p class="small muted">Increase opposing force near a software boundary.</p></div><div class="architecture-card"><h3>Vibration</h3><p class="small muted">Apply controlled oscillatory feedback.</p></div><div class="architecture-card"><h3>Remote force</h3><p class="small muted">Transmit measured interaction force to a remote controller.</p></div></div>
      </section>
    </div>
    <section class="panel panel-pad" style="margin-top:14px"><div class="panel-head"><div><h2>System identification workflow</h2><p>Replace provisional twin parameters with measurements from the assembled mechanism</p></div>${badge('Physical calibration planned','planned')}</div>
      <div class="flow-row"><div class="flow-card"><b>1 · Assemble and instrument</b><span>encoders · IMU · limit switches</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>2 · Measure static force</b><span>apply known dead weights: F = mg</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>3 · Fit dynamic terms</b><span>spring · damping · friction · belt · delay</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>4 · Recalibrate both twins</b><span>Python plant and OpenModelica</span></div><div class="flow-arrow">→</div><div class="flow-card"><b>5 · Repeat benchmark gates</b><span>compare on hardware-observable states</span></div></div>
      <div class="rule"></div><div class="metric-strip">${badge('Spring rates · unmeasured','planned')}${badge('Belt compliance · unmeasured','planned')}${badge('Rail / motor friction · unmeasured','planned')}${badge('Command latency · unmeasured','planned')}</div>${sourceNotes(ctx,['hardware','project-overview','physics-validation'])}
    </section>`;
}

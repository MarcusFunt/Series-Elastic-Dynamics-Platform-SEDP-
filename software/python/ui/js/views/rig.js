import {badge, pageHead} from '../shared.js';

export async function renderRig() {
  return `${pageHead('LIVE SIMULATION','Rig simulator','Run the nonlinear Python plant and inspect the animated mechanism, controller output, carriage tracking, resonator motion, motor torque, and belt force.',`<a class="button" href="/live?view=training" target="_blank" rel="noopener">Open training controls ↗</a>`)}
    <div class="controls-line" style="margin:0 0 11px"><span>${badge('Live Python simulation','live')} <span class="small muted">The physical rig is not connected to this view.</span></span><a class="text-link small" href="/live" target="_blank" rel="noopener">Open simulator in a full browser tab ↗</a></div>
    <section class="iframe-shell"><iframe src="/live" title="Interactive nonlinear active vibration rig simulator" loading="lazy"></iframe></section>`;
}

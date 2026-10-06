import {renderOverview} from './views/overview.js';
import {renderRig} from './views/rig.js';
import {renderModels} from './views/models.js';
import {renderControls} from './views/controls.js';
import {renderSensors} from './views/sensors.js';
import {renderLearning} from './views/learning.js';
import {renderHardware, renderHaptics} from './views/architecture.js';
import {renderProjectMap} from './views/project-map.js';

const names={
  overview:'Overview',rig:'Rig simulator',models:'Digital twin',controls:'Control lab',
  sensors:'Sensors & timing',learning:'Learning & experiments',hardware:'Hardware & firmware',
  haptics:'Haptics & system ID','project-map':'Project map',
};
const renderers={
  overview:renderOverview,rig:renderRig,models:renderModels,controls:renderControls,
  sensors:renderSensors,learning:renderLearning,hardware:renderHardware,
  haptics:renderHaptics,'project-map':renderProjectMap,
};
const ctx={catalog:null,evidence:[],media:[],cache:new Map(),navigate:null};
const root=document.getElementById('viewRoot');
let navigationGeneration=0;

async function navigate(view,updateHash=true) {
  if(!renderers[view])view='overview';
  const generation=++navigationGeneration;
  const isCurrent=()=>generation===navigationGeneration;
  document.querySelectorAll('.nav-item').forEach(item=>item.classList.toggle('active',item.dataset.view===view));
  document.getElementById('currentViewName').textContent=names[view];
  if(updateHash)history.replaceState(null,'',`#/${view}`);
  root.innerHTML='<div class="loading">Loading view…</div>';
  const viewCtx={...ctx,mounts:[],isCurrent,navigate:id=>navigate(id)};
  try {
    const markup=await renderers[view](viewCtx);
    if(!isCurrent())return;
    root.innerHTML=markup;
    root.focus({preventScroll:true});
    root.scrollIntoView({behavior:'smooth',block:'start'});
    for(const mount of viewCtx.mounts){
      if(!isCurrent())return;
      await mount();
    }
  } catch(error) {
    if(!isCurrent())return;
    root.innerHTML=`<section class="panel panel-pad"><h2>View unavailable</h2><p class="muted">${String(error.message||error)}</p><p class="small">The rest of the project workspace is still available from the navigation.</p></section>`;
  }
}
ctx.navigate=id=>navigate(id);

document.getElementById('primaryNav').addEventListener('click',event=>{
  const button=event.target.closest('[data-view]');
  if(button)navigate(button.dataset.view);
});
root.addEventListener('click',event=>{
  const button=event.target.closest('[data-go]');
  if(button)navigate(button.dataset.go);
});
window.addEventListener('hashchange',()=>navigate(location.hash.replace(/^#\//,''),false));

try {
  const [catalogResponse,evidenceResponse,mediaResponse]=await Promise.all([
    fetch('/api/project/catalog'),fetch('/api/project/evidence'),fetch('/api/project/media'),
  ]);
  if(!catalogResponse.ok||!evidenceResponse.ok||!mediaResponse.ok)throw new Error('Project catalog is unavailable.');
  ctx.catalog=await catalogResponse.json();
  ctx.evidence=await evidenceResponse.json();
  ctx.media=await mediaResponse.json();
  await navigate(location.hash.replace(/^#\//,'')||'overview',false);
} catch(error) {
  root.innerHTML=`<section class="panel panel-pad"><h2>Workspace data could not load</h2><p class="muted">${String(error.message||error)}</p><p class="small">Start the browser server from <code>software/python</code> and refresh this page.</p></section>`;
}

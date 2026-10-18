'use strict';
const $ = id => document.getElementById(id);
let token = '', jobs = [], selected = null, connected = false, refreshing = false;
const notice = (text, error=false) => { $('notice').textContent=text; $('notice').className=error?'error':''; };
async function api(path, options={}) {
  const response = await fetch('/api'+path, {...options, headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json',...options.headers}});
  if (!response.ok) { let body=await response.json().catch(()=>({})); throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail || response.status)); }
  return response.json();
}
const node=(tag,text,cls)=>{const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(cls)e.className=cls; return e;};
function renderJobs(){
  const list=$('jobs'); list.replaceChildren();
  const filtered=jobs.filter(j=>$('filter').value==='all'||j.status===$('filter').value);
  if(!filtered.length)list.append(node('div','No jobs in this view. Submit an experiment to begin.','empty'));
  for(const job of filtered){
    const card=node('article',undefined,'job'), head=node('div',undefined,'job-heading');
    head.append(node('h3',job.spec.name),node('span',job.status,'badge '+job.status)); card.append(head);
    card.append(node('p',`${job.spec.model} · ${job.spec.dataset} · ${job.spec.epochs} epochs · ${new Date(job.created*1000).toLocaleString()}`,'job-meta'));
    const p=job.progress; if(p.epoch){const bar=node('progress');bar.max=p.epochs;bar.value=p.epoch;bar.setAttribute('aria-label','Epoch progress');card.append(bar,node('p',`${p.simulated?'Simulated ':''}Epoch ${p.epoch} / ${p.epochs}`,'job-meta'));}
    if(job.error)card.append(node('p',job.error,'job-meta'));
    const actions=node('div',undefined,'job-actions'), view=node('button','View logs & results');view.onclick=()=>{selected=job.id;showDetail();};actions.append(view);
    if(['queued','starting','running','cancelling'].includes(job.status)){const cancel=node('button',job.cancel?'Cancellation requested':'Cancel');cancel.disabled=!!job.cancel;cancel.onclick=async()=>{try{await api(`/jobs/${job.id}/cancel`,{method:'POST'});await refresh();}catch(e){notice(e.message,true);}};actions.append(cancel);}
    card.append(actions);list.append(card);
  }
  $('queued').textContent=jobs.filter(j=>j.status==='queued').length;
  $('running').textContent=jobs.filter(j=>['starting','running','cancelling'].includes(j.status)).length;
  $('completed').textContent=jobs.filter(j=>j.status==='completed').length;
}
async function showDetail(){
  if(!selected)return;
  const job=jobs.find(j=>j.id===selected); if(!job)return;
  $('detail').hidden=false;$('detail-title').textContent=job.spec.name;
  $('detail-meta').textContent=`${job.id} · ${job.status} · GPU: ${job.gpu||'waiting'}${job.tracking.run_id?' · MLflow run: '+job.tracking.run_id:''}`;
  $('metrics').replaceChildren();for(const [k,v] of Object.entries(job.progress.metrics||{})){const m=node('div',k,'metric');m.append(node('b',Number(v).toFixed(4)));$('metrics').append(m);}
  $('artifacts').replaceChildren();for(const filename of job.artifacts){const button=node('button','↓ '+filename);button.onclick=async()=>{try{const r=await fetch(`/api/jobs/${job.id}/artifacts/${filename}`,{headers:{Authorization:'Bearer '+token}});if(!r.ok)throw new Error('Download failed');const url=URL.createObjectURL(await r.blob()),a=node('a');a.href=url;a.download=filename.split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice(e.message,true);}};$('artifacts').append(button);}
  try{$('logs').textContent=(await api(`/jobs/${job.id}/logs`)).text;}catch(e){notice(e.message,true);}
}
async function refresh(){
  if(!connected||refreshing)return;refreshing=true;
  try{const [all,system]=await Promise.all([api('/jobs'),api('/system')]);jobs=all;renderJobs();$('worker').textContent=system.worker_alive?'Online':'Offline';
    const reservations=new Set(jobs.filter(j=>['starting','running','cancelling'].includes(j.status)).map(j=>j.gpu));
    $('gpus').replaceChildren();for(const gpu of system.worker.gpus||[]){const el=node('div',undefined,'gpu');el.append(node('b',gpu.name),node('span',`${gpu.uuid} · ${gpu.free_mb} MB free · ${reservations.has(gpu.uuid)?'Reserved':gpu.available?'Available':'Busy'}`));$('gpus').append(el);}
    if(!(system.worker.gpus||[]).length)$('gpus').textContent=system.worker.error||'Waiting for GPU telemetry.';
    notice(!system.worker_alive?'Worker offline. Submitted jobs will remain queued.':system.mode==='demo'?'SIMULATION MODE · No real training, model weights or MLflow runs.':'Worker online · Training jobs are assigned to available GPUs.');
    if(selected)await showDetail();
  }catch(e){notice(e.message,true);}finally{refreshing=false;}
}
$('connect').onclick=async()=>{token=$('token').value;try{const data=await api('/datasets');$('dataset').replaceChildren();for(const d of data.datasets){const option=node('option',d);option.value=d;$('dataset').append(option);}if(!data.datasets.length){const option=node('option','No datasets found');option.value='';$('dataset').append(option);}$('root').textContent='Server dataset root: '+data.root;connected=true;await refresh();}catch(e){connected=false;notice(e.message,true);}};
$('submit').onsubmit=async event=>{event.preventDefault();const button=event.target.querySelector('button[type=submit]');button.disabled=true;try{const spec=Object.fromEntries(new FormData(event.target));for(const key of ['epochs','batch','imgsz','seed'])spec[key]=Number(spec[key]);await api('/jobs',{method:'POST',body:JSON.stringify(spec)});await refresh();notice('Job queued. The worker will start it when a GPU is available.');}catch(e){notice(e.message,true);}finally{button.disabled=false;}};
$('filter').onchange=renderJobs;$('close').onclick=()=>{selected=null;$('detail').hidden=true;};setInterval(refresh,2500);

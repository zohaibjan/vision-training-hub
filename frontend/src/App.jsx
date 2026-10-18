import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createApi } from './api.js';
import { previewJobs, previewSystem } from './preview.js';
const active = ['starting', 'running', 'cancelling'];
const defaults = { name: 'YOLOv8 baseline', dataset: '', model: 'yolov8n.pt', epochs: 50, batch: 8, imgsz: 640, seed: 42 };
export default function App({ preview = false, pollMs = 2500 }) {
  const [tokenInput, setTokenInput] = useState(''), [token, setToken] = useState('');
  const [jobs, setJobs] = useState(preview ? previewJobs : []), [system, setSystem] = useState(preview ? previewSystem : null);
  const [datasets, setDatasets] = useState(preview ? ['synthetic-boxes'] : []), [root, setRoot] = useState('');
  const [form, setForm] = useState({ ...defaults, dataset: preview ? 'synthetic-boxes' : '' });
  const [filter, setFilter] = useState('all'), [selected, setSelected] = useState(null), [logs, setLogs] = useState('');
  const [error, setError] = useState(''), [message, setMessage] = useState(''), [busy, setBusy] = useState(false), [revision, setRevision] = useState(0);
  const api = useMemo(() => createApi(token), [token]);
  const connectionAttempt = useRef(0);
  useEffect(() => {
    if (!token || preview) return;
    const abort = new AbortController(); let timer;
    async function poll() {
      try {
        const [all, status] = await Promise.all([api.get('/jobs', abort.signal), api.get('/system', abort.signal)]);
        if (!abort.signal.aborted) { setJobs(all); setSystem(status); setError(''); }
      } catch (e) { if (!abort.signal.aborted) setError(e.message); }
      finally { if (!abort.signal.aborted) timer = setTimeout(poll, pollMs); }
    }
    poll(); return () => { abort.abort(); clearTimeout(timer); };
  }, [api, token, preview, revision, pollMs]);
  const job = jobs.find(j => j.id === selected);
  useEffect(() => {
    if (!selected) return;
    if (preview) { setLogs('READ-ONLY PREVIEW\nSample interface state. No training was performed and no model metrics or artifacts were generated.'); return; }
    const abort = new AbortController(); let timer;
    async function pollLog() {
      try { const result = await api.get(`/jobs/${selected}/logs`, abort.signal); if (!abort.signal.aborted) setLogs(result.text); }
      catch (e) { if (!abort.signal.aborted) setLogs(e.message); }
      finally { if (!abort.signal.aborted) timer = setTimeout(pollLog, pollMs); }
    }
    setLogs('Loading log…'); pollLog(); return () => { abort.abort(); clearTimeout(timer); };
  }, [selected, api, preview, pollMs]);
  async function connect(event) {
    event.preventDefault(); const attempt = ++connectionAttempt.current;
    setError(''); setBusy(true); setToken(''); setJobs([]); setSystem(null); setSelected(null); setDatasets([]);
    try {
      const result = await createApi(tokenInput).get('/datasets');
      if (attempt !== connectionAttempt.current) return;
      setDatasets(result.datasets); setRoot(result.root); setForm(f => ({ ...f, dataset: result.datasets[0] || '' })); setToken(tokenInput); setTokenInput('');
    } catch (e) { if (attempt === connectionAttempt.current) setError(e.message); }
    finally { if (attempt === connectionAttempt.current) setBusy(false); }
  }
  function disconnect() { connectionAttempt.current++; setToken(''); setJobs([]); setDatasets([]); setSystem(null); setSelected(null); setError(''); setMessage(''); }
  async function submit(event) {
    event.preventDefault(); if (preview || !token) return;
    setBusy(true); setError(''); setMessage('');
    try { const spec = { ...form }; for (const k of ['epochs', 'batch', 'imgsz', 'seed']) spec[k] = Number(spec[k]);
      await api.post('/jobs', spec); setMessage('Job queued. It will start when a GPU is available.'); setRevision(x => x + 1);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  async function cancel(id) { try { await api.post(`/jobs/${id}/cancel`); setMessage('Cancellation requested.'); setRevision(x => x + 1); } catch (e) { setError(e.message); } }
  async function download(filename) { try { const blob = await api.download(selected, filename); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = filename.split('/').pop(); a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); } catch (e) { setError(e.message); } }
  const reserved = new Set(jobs.filter(j => active.includes(j.status)).map(j => j.gpu));
  const filtered = jobs.filter(j => filter === 'all' || j.status === filter);
  const statusText = preview ? 'READ-ONLY PREVIEW · Sample jobs and GPU states. No training or real metrics.' : !token ? 'Connect to see your training workspace.' : !system?.worker_alive ? 'Worker offline. Submitted jobs will remain queued.' : system.mode === 'demo' ? 'SIMULATION MODE · No real training, model weights or MLflow runs.' : 'Worker online · Training jobs are assigned to available GPUs.';
  const field = (key, value) => setForm(f => ({ ...f, [key]: value }));
  return <>
    <header><a className="brand" href="./">V<span>▧</span>H <b>Vision Training Hub</b></a><span className="tag">React · FastAPI · MLflow</span></header>
    <main><section className="intro"><div><p className="eyebrow">COMPUTER VISION / TRAINING OPERATIONS</p><h1>From labelled data<br />to your next model.</h1><p className="muted">Submit an experiment. We’ll find a GPU and take it from there.</p></div>
      {preview ? <div className="preview-note"><span className="tag">PORTFOLIO PREVIEW</span><p>Explore the training workspace.</p><a href="https://github.com/zohaibjan/vision-training-hub">View source & setup instructions ↗</a></div> : <form className="connection" onSubmit={connect}><label htmlFor="token">Portal access token</label><div className="inline"><input id="token" type="password" autoComplete="off" value={tokenInput} onChange={e => setTokenInput(e.target.value)} required placeholder="Enter your configured token" /><button disabled={busy}>Connect</button></div><p className="muted small">Held in this tab’s memory only. {token && <button type="button" onClick={disconnect}>Disconnect</button>}</p></form>}
    </section>
    <p id="notice" role={error ? 'alert' : 'status'} className={error ? 'error' : ''}>{error || message || statusText}</p>
    {message && !error && <p className="small muted">{statusText}</p>}
    <section className="stats" aria-label="Workspace summary">{[['QUEUED', jobs.filter(j => j.status === 'queued').length], ['TRAINING', jobs.filter(j => active.includes(j.status)).length], ['COMPLETED', jobs.filter(j => j.status === 'completed').length], ['WORKER', preview ? 'Preview' : system?.worker_alive ? 'Online' : 'Offline']].map(([title, value]) => <article key={title}><span>{title}</span><strong className={title === 'WORKER' ? 'worker' : ''}>{!token && !preview && title !== 'WORKER' ? '—' : value}</strong></article>)}</section>
    <div className="layout"><aside><section className="panel"><p className="eyebrow">NEW EXPERIMENT</p><h2>Set up training</h2><form onSubmit={submit} aria-label="Training configuration">
      <label>Experiment name<input required maxLength={100} value={form.name} onChange={e => field('name', e.target.value)} /></label>
      <label>Dataset folder<select required value={form.dataset} onChange={e => field('dataset', e.target.value)}>{!datasets.length && <option value="">{token ? 'No datasets found' : 'Connect to load datasets'}</option>}{datasets.map(d => <option key={d}>{d}</option>)}</select></label><p className="small muted">{root ? `Server dataset root: ${root}` : 'Server folders containing images, labels and data.yaml.'}</p>
      <label>Base model<select value={form.model} onChange={e => field('model', e.target.value)}>{['n', 's', 'm', 'l', 'x'].map(n => <option key={n}>yolov8{n}.pt</option>)}</select></label>
      <div className="pair">{[['epochs','Epochs',1,1000,1],['batch','Batch size',1,256,1],['imgsz','Image size',32,2048,32],['seed','Seed',0,2147483647,1]].map(([key,label,min,max,step]) => <label key={key}>{label}<input type="number" required min={min} max={max} step={step} value={form[key]} onChange={e => field(key,e.target.value)} /></label>)}</div>
      <button className="primary" disabled={preview || !token || busy || !datasets.length}>{preview ? 'Training disabled in preview' : busy ? 'Validating dataset…' : 'Queue training job ↗'}</button><p className="small muted">One job per GPU. Busy devices keep your job in the queue.</p>
    </form></section><section className="panel"><h2>GPU pool</h2>{(system?.worker?.gpus || []).map(gpu => <div className="gpu" key={gpu.uuid}><b>{gpu.name}</b><span>{gpu.uuid} · {gpu.free_mb} MB free · {reserved.has(gpu.uuid) ? 'Reserved' : gpu.available ? 'Available' : 'Busy'}</span></div>)}{!(system?.worker?.gpus || []).length && <p className="muted small">{system?.worker?.error || 'Waiting for GPU telemetry.'}</p>}</section></aside>
    <section className="panel jobs"><div className="section-title"><div><p className="eyebrow">EXPERIMENTS</p><h2>Training jobs</h2></div><select aria-label="Filter jobs" value={filter} onChange={e => setFilter(e.target.value)}>{['all','queued','starting','running','cancelling','completed','failed','cancelled','interrupted'].map(s => <option key={s} value={s}>{s === 'all' ? 'All jobs' : s}</option>)}</select></div>
      {!filtered.length && <div className="empty">No jobs in this view.<br /><span className="muted">Choose a labelled dataset to get started.</span></div>}
      {filtered.map(j => <article className="job" key={j.id}><div className="job-heading"><h3>{j.spec.name}</h3><span className={'badge '+j.status}>{j.status}</span></div><p className="job-meta">{j.spec.model} · {j.spec.dataset} · {j.spec.epochs} epochs · {new Date(j.created * 1000).toLocaleString()}</p>{j.progress.epoch > 0 && <><progress max={j.progress.epochs} value={j.progress.epoch} aria-label={`${j.spec.name} epoch progress`} /><p className="job-meta">{j.progress.simulated ? 'Simulated ' : ''}Epoch {j.progress.epoch} / {j.progress.epochs}</p></>}{j.error && <p className="job-meta">{j.error}</p>}<div className="job-actions"><button onClick={() => setSelected(j.id)}>View logs & results<span className="sr-only"> for {j.spec.name}</span></button>{['queued', ...active].includes(j.status) && <button disabled={preview || !!j.cancel} onClick={() => cancel(j.id)}>{j.cancel ? 'Cancellation requested' : 'Cancel'}<span className="sr-only"> {j.spec.name}</span></button>}</div></article>)}
    </section></div>
    {job && <section className="panel" aria-label="Job details"><div className="section-title"><h2>{job.spec.name}</h2><button onClick={() => setSelected(null)}>Close details</button></div><p className="muted small">{job.id} · {job.status} · GPU: {job.gpu || 'waiting'}{job.tracking.run_id ? ` · MLflow run: ${job.tracking.run_id}` : ''}</p><div className="metrics">{Object.entries(job.progress.metrics || {}).map(([k,v]) => <div className="metric" key={k}>{k}<b>{Number(v).toFixed(4)}</b></div>)}</div><div>{job.artifacts.map(file => <button key={file} onClick={() => download(file)}>↓ {file}</button>)}</div><h3>Training log</h3><pre>{logs}</pre></section>}
    <footer>Vision Training Hub · React frontend / FastAPI backend · Independent reference implementation</footer></main>
  </>;
}

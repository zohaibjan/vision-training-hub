import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';
import { createApi } from './api.js';
const example = { id:'j1',created:1,status:'running',spec:{name:'Baseline',model:'yolov8n.pt',dataset:'boxes',epochs:2},progress:{epoch:1,epochs:2,metrics:{loss:0.5}},tracking:{run_id:'run-123'},artifacts:['weights/best.pt'],gpu:'GPU-a' };
function mockServer({ badToken=false, failSubmit=false, jobs=[example] }={}) {
  const server=vi.fn(async (url, opts={})=>{
    let data;
    if(url==='/api/datasets') {
      if(badToken)return {ok:false,status:401,json:async()=>({detail:'Enter the portal access token'})};
      data={root:'/datasets',datasets:['boxes']};
    } else if(url==='/api/system')data={mode:'demo',worker_alive:true,worker:{gpus:[{uuid:'GPU-a',name:'Test GPU',free_mb:8192,available:true}]}};
    else if(url==='/api/jobs'&&opts.method==='POST') {
      if(failSubmit)return {ok:false,status:422,json:async()=>({detail:'Missing label: labels/train/a.txt'})};
      data={id:'new-job'};
    } else if(url==='/api/jobs')data=jobs;
    else if(url.endsWith('/cancel'))data={...example,cancel:1};
    else if(url.endsWith('/logs'))data={text:'Epoch 1/2: training'};
    else if(url.includes('/artifacts/'))return {ok:true,blob:async()=>new Blob(['checkpoint'])};
    else throw Error('Unexpected request '+url);
    return {ok:true,json:async()=>data};
  });
  vi.stubGlobal('fetch',server);return server;
}
async function connect(user) {await user.type(screen.getByLabelText('Portal access token'),'local-token');await user.click(screen.getByRole('button',{name:'Connect',exact:true}));await screen.findByRole('heading',{name:'Baseline',exact:true});}

describe('React training portal',()=>{
  it('shows a safe disconnected state',()=>{render(<App/>);expect(screen.getByRole('button',{name:/Queue training/})).toBeDisabled();expect(screen.getByText('Connect to see your training workspace.')).toBeVisible();});
  it('connects with bearer auth and displays telemetry',async()=>{const server=mockServer();const user=userEvent.setup();render(<App/>);await connect(user);expect(screen.getByText('Test GPU')).toBeVisible();expect(server).toHaveBeenCalledWith('/api/jobs',expect.objectContaining({headers:expect.objectContaining({Authorization:'Bearer local-token'})}));expect(screen.getByLabelText('Portal access token')).toHaveValue('');});
  it('surfaces authentication failures',async()=>{mockServer({badToken:true});const user=userEvent.setup();render(<App/>);await user.type(screen.getByLabelText('Portal access token'),'bad');await user.click(screen.getByRole('button',{name:'Connect',exact:true}));expect(await screen.findByRole('alert')).toHaveTextContent('Enter the portal access token');});
  it('submits numeric parameters to FastAPI',async()=>{const server=mockServer();const user=userEvent.setup();render(<App/>);await connect(user);await user.click(screen.getByRole('button',{name:/Queue training/}));await screen.findByText(/Job queued/);const [,opts]=server.mock.calls.find(([url,o])=>url==='/api/jobs'&&o.method==='POST');expect(JSON.parse(opts.body)).toMatchObject({dataset:'boxes',epochs:50,batch:8,imgsz:640,seed:42,model:'yolov8n.pt'});});
  it('shows dataset validation errors',async()=>{mockServer({failSubmit:true});const user=userEvent.setup();render(<App/>);await connect(user);await user.click(screen.getByRole('button',{name:/Queue training/}));expect(await screen.findByRole('alert')).toHaveTextContent('Missing label');});
  it('filters jobs by status',async()=>{mockServer();const user=userEvent.setup();render(<App/>);await connect(user);await user.selectOptions(screen.getByLabelText('Filter jobs'),'completed');expect(screen.queryByRole('heading',{name:'Baseline',exact:true})).not.toBeInTheDocument();expect(screen.getByText('No jobs in this view.')).toBeVisible();});
  it('requests cancellation and shows acknowledgement',async()=>{const server=mockServer();const user=userEvent.setup();render(<App/>);await connect(user);await user.click(screen.getByRole('button',{name:'Cancel Baseline'}));await screen.findByText('Cancellation requested.');expect(server).toHaveBeenCalledWith('/api/jobs/j1/cancel',expect.objectContaining({method:'POST'}));});
  it('shows logs, metrics and MLflow run IDs',async()=>{mockServer();const user=userEvent.setup();render(<App/>);await connect(user);await user.click(screen.getByRole('button',{name:'View logs & results for Baseline'}));expect(await screen.findByText('Epoch 1/2: training')).toBeVisible();expect(screen.getByText('0.5000')).toBeVisible();expect(screen.getByText(/MLflow run: run-123/)).toBeVisible();await user.click(screen.getByRole('button',{name:'Close details'}));expect(screen.queryByRole('region',{name:'Job details'})).not.toBeInTheDocument();});
  it('disconnect clears private job state',async()=>{mockServer();const user=userEvent.setup();render(<App/>);await connect(user);await user.click(screen.getByRole('button',{name:'Disconnect'}));expect(screen.queryByRole('heading',{name:'Baseline',exact:true})).not.toBeInTheDocument();expect(screen.getByRole('button',{name:/Queue training/})).toBeDisabled();});
  it('preview performs no network calls and disables mutations',async()=>{const server=vi.fn();vi.stubGlobal('fetch',server);const user=userEvent.setup();render(<App preview/>);expect(screen.getByText(/READ-ONLY PREVIEW/)).toBeVisible();expect(screen.getByRole('link',{name:/Vision Training Hub/})).toHaveAttribute('href','./?preview=1');expect(screen.getByRole('button',{name:'Training disabled in preview'})).toBeDisabled();await user.click(screen.getByRole('button',{name:/View logs & results for Object detection/}));expect(await screen.findByText(/No training was performed/)).toBeVisible();expect(server).not.toHaveBeenCalled();});
  it('downloads artifacts using authentication and encoded paths',async()=>{const server=mockServer();const blob=await createApi('secret',server).download('j1','weights/best.pt');expect(blob.size).toBeGreaterThan(0);expect(server).toHaveBeenCalledWith('/api/jobs/j1/artifacts/weights/best.pt',expect.objectContaining({headers:expect.objectContaining({Authorization:'Bearer secret'})}));});
  it('formats FastAPI field validation errors',async()=>{const fetcher=vi.fn(async()=>({ok:false,status:422,json:async()=>({detail:[{loc:['body','epochs'],msg:'Must be positive'}]})}));await expect(createApi('x',fetcher).post('/jobs',{})).rejects.toThrow('epochs: Must be positive');});
  it('aborts in-flight polling on unmount',async()=>{const server=mockServer();const user=userEvent.setup();const view=render(<App/>);await connect(user);const signal=server.mock.calls.find(([url])=>url==='/api/jobs')[1].signal;view.unmount();expect(signal.aborted).toBe(true);});
});

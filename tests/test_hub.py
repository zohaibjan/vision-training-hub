import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from hub.api import create_app
from hub.config import Config
from hub.datasets import validate
from hub.gpu import discover
from hub.store import Store
from hub.worker import Scheduler

@pytest.fixture
def config(tmp_path):
    c = Config(tmp_path / 'state', tmp_path / 'datasets', mode='demo', token='test-only-token')
    c.prepare()
    root = c.datasets / 'boxes'
    for split in ('train', 'val'):
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        Image.new('RGB', (32, 32), 'white').save(root / 'images' / split / 'a.png')
        (root / 'labels' / split / 'a.txt').write_text('0 0.5 0.5 0.5 0.5\n')
    (root / 'data.yaml').write_text('train: images/train\nval: images/val\nnames: [box]\n')
    return c

@pytest.fixture
def client(config):
    return TestClient(create_app(config), headers={'Authorization': 'Bearer test-only-token'})

def submit(client, **kw):
    r = client.post('/api/jobs', json={'dataset': 'boxes', 'epochs': 2, **kw})
    assert r.status_code == 201, r.text
    return r.json()

def test_auth(config):
    client = TestClient(create_app(config))
    assert client.get('/api/jobs').status_code == 401
    assert client.get('/').status_code == 200

@pytest.mark.parametrize('change', [{'epochs':0}, {'batch':-1}, {'imgsz':65}, {'model':'../../evil.pt'}, {'extra':1}])
def test_invalid_parameters(client, change):
    assert client.post('/api/jobs', json={'dataset':'boxes', **change}).status_code == 422

def test_dataset_listing_and_hash(client, config):
    assert client.get('/api/datasets').json()['datasets'] == ['boxes']
    a=validate(config.datasets,'boxes')
    assert a['counts']=={'train':1,'val':1}
    (config.datasets/'boxes/labels/train/a.txt').write_text('0 0.4 0.5 0.5 0.5')
    assert validate(config.datasets,'boxes')['sha256'] != a['sha256']

@pytest.mark.parametrize('selection', ['../', '/etc', 'missing'])
def test_dataset_escape(client, selection):
    assert client.post('/api/jobs',json={'dataset':selection}).status_code == 422

def test_symlink_escape(config):
    outside=config.state/'outside.yaml';outside.write_text('names: [secret]')
    path=config.datasets/'boxes/data.yaml';path.unlink();path.symlink_to(outside)
    with pytest.raises(ValueError):validate(config.datasets,'boxes')

@pytest.mark.parametrize('label', ['-1 0.5 0.5 0.5 0.5', '0 nan 0.5 0.5 0.5', '0 .5 .5 0 .5', '0 .5 .5', '0 2 .5 .5 .5'])
def test_bad_labels(config,label):
    (config.datasets/'boxes/labels/train/a.txt').write_text(label)
    with pytest.raises(ValueError):validate(config.datasets,'boxes')

def test_no_download_hook(config):
    path=config.datasets/'boxes/data.yaml';path.write_text(path.read_text()+'download: echo unsafe\n')
    with pytest.raises(ValueError):validate(config.datasets,'boxes')

def test_atomic_claims_and_fifo(config):
    store=Store(config.state);first=store.add({});second=store.add({})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        claimed=list(pool.map(lambda _:Store(config.state).claim('GPU-1'),range(4)))
    assert sum(j is not None for j in claimed)==1
    assert next(j for j in claimed if j)['id']==first['id']
    assert store.claim('GPU-2')['id']==second['id']

def test_cancel_queued(client,config):
    job=submit(client);client.post('/api/jobs/'+job['id']+'/cancel')
    assert Store(config.state).get(job['id'])['status']=='cancelled'
    assert Store(config.state).claim('demo-0') is None

def test_busy_gpu_keeps_queue(client,config):
    job=submit(client)
    scheduler=Scheduler(config,inventory=lambda c:([{'uuid':'GPU-busy','available':False}],None))
    scheduler.tick()
    assert scheduler.store.get(job['id'])['status']=='queued'

def test_gpu_probe_fails_closed(config,monkeypatch):
    from dataclasses import replace
    def fail(*a,**k):raise subprocess.CalledProcessError(1,'nvidia-smi')
    monkeypatch.setattr(subprocess,'check_output',fail)
    cards,error=discover(replace(config,mode='real'))
    assert not cards and error

def test_gpu_external_process_and_mig(config,monkeypatch):
    from dataclasses import replace
    def query(args,**kw):
        if '--query-compute-apps=gpu_uuid,pid' in args:return 'GPU-a, 123\n'
        return 'GPU-a, A, 9000, 0, Disabled\nGPU-b, B, 9000, 0, Disabled\nGPU-c, C, 9000, 0, Enabled\n'
    monkeypatch.setattr(subprocess,'check_output',query)
    cards,error=discover(replace(config,mode='real'))
    assert [c['available'] for c in cards]==[False,True,False] and error is None

def test_demo_queue_end_to_end(client,config):
    ids=[submit(client)['id'] for _ in range(3)]
    scheduler=Scheduler(config)
    try:
        scheduler.tick()
        states=[scheduler.store.get(i)['status'] for i in ids]
        assert states==['running','running','queued']
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            scheduler.tick()
            if all(scheduler.store.get(i)['status']=='completed' for i in ids):break
            time.sleep(.1)
        assert all(scheduler.store.get(i)['status']=='completed' for i in ids)
        detail=client.get('/api/jobs/'+ids[0]).json()
        assert detail['progress']['simulated'] and not detail['artifacts']
        assert 'SIMULATION ONLY' in client.get('/api/jobs/'+ids[0]+'/logs').json()['text']
    finally:scheduler.shutdown()

def test_cancel_running(client,config):
    job=submit(client,epochs=100)
    scheduler=Scheduler(config)
    try:
        scheduler.tick();client.post('/api/jobs/'+job['id']+'/cancel');scheduler.tick()
        assert scheduler.store.get(job['id'])['status']=='cancelled'
    finally:scheduler.shutdown()

def test_modified_dataset_fails(client,config):
    job=submit(client)
    (config.datasets/'boxes/labels/train/a.txt').write_text('0 .4 .5 .5 .5')
    scheduler=Scheduler(config);scheduler.tick()
    assert scheduler.store.get(job['id'])['status']=='failed'
    assert 'changed' in scheduler.store.get(job['id'])['error']

def test_restart_marks_interrupted(client,config):
    job=submit(client)
    store=Store(config.state);store.claim('demo-0')
    Scheduler(config).recover()
    assert store.get(job['id'])['status']=='interrupted'

def test_artifact_access(client,config):
    job=submit(client);base=config.state/'jobs'/job['id']/'training';base.mkdir(parents=True)
    (base/'results.csv').write_text('epoch,loss\n1,1\n')
    (base/'secret.env').write_text('not downloadable')
    assert client.get('/api/jobs/'+job['id']+'/artifacts/results.csv').status_code==200
    assert client.get('/api/jobs/'+job['id']+'/artifacts/secret.env').status_code==404
    assert client.get('/api/jobs/unknown/artifacts/results.csv').status_code==404

"""Contract tests for the training adapter; no CUDA or real YOLO training implied."""
import json
import sys
from types import SimpleNamespace
import pytest
from hub.runner import real_training

@pytest.mark.parametrize('fail',[False,True])
def test_training_adapter_contract(tmp_path,monkeypatch,fail):
    calls={}
    class Model:
        def __init__(self,name):calls['model']=name
        def add_callback(self,event,callback):calls['callback']=callback
        def train(self,**kwargs):
            calls['args']=kwargs
            calls['callback'](SimpleNamespace(epoch=0,metrics={'mAP':.25}))
            if fail:raise RuntimeError('training failure')
            (tmp_path/'training').mkdir()
    mlflow=SimpleNamespace(
        set_tracking_uri=lambda uri:calls.update(uri=uri),
        set_experiment=lambda name:None,
        start_run=lambda **kw:SimpleNamespace(info=SimpleNamespace(run_id='run-123',experiment_id='1')),
        set_tags=lambda tags:calls.update(tags=tags),
        set_tag=lambda *args:None,
        log_artifact=lambda *args:None,
        log_artifacts=lambda *args:calls.update(artifacts=True),
        end_run=lambda **kw:calls.update(ended=kw['status']))
    settings={}
    monkeypatch.setitem(sys.modules,'mlflow',mlflow)
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=Model,settings=settings))
    monkeypatch.setenv('MLFLOW_TRACKING_URI','http://127.0.0.1:5000')
    job={'id':'job-1','gpu':'GPU-uuid','dataset_config':{'names':['box']},'spec':{'model':'yolov8n.pt','dataset_sha256':'abc','epochs':2,'imgsz':64,'batch':1,'seed':42}}
    if fail:
        with pytest.raises(RuntimeError):real_training(tmp_path,job)
    else:real_training(tmp_path,job)
    assert settings['mlflow'] is True
    assert calls['args']['device']==0
    assert calls['ended']==('FAILED' if fail else 'FINISHED')
    assert json.loads((tmp_path/'tracking.json').read_text())['run_id']=='run-123'
    assert json.loads((tmp_path/'progress.json').read_text())['epoch']==1

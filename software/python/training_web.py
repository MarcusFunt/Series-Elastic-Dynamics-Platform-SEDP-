"""Persistent local training jobs for the browser; never execute user commands."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ID = re.compile(r'^[0-9a-f]{32}$')
ARTIFACT = re.compile(r'^[a-zA-Z0-9_]+\.(pt|json|npz|log)$')


class TrainingJobs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.worker = None
        for directory in self.root.iterdir():
            if directory.is_dir() and RUN_ID.fullmatch(directory.name):
                try:
                    job = self._read(directory.name)
                    if job['status'] in ('running', 'cancelling'):
                        job.update(status='interrupted', finished=time.time())
                        self._save(job)
                except (OSError, ValueError, KeyError):
                    pass

    def _directory(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError('Invalid run ID')
        directory = (self.root/run_id).resolve()
        if directory.parent != self.root:
            raise ValueError('Run path escapes training directory')
        return directory

    def _read(self, run_id):
        return json.loads((self._directory(run_id)/'job.json').read_text())

    def _save(self, job):
        target = self._directory(job['id'])/'job.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(job, indent=2, allow_nan=False))
        temporary.replace(target)

    def artifact(self, run_id, filename):
        if not isinstance(filename, str) or not ARTIFACT.fullmatch(filename):
            raise ValueError('Invalid artifact name')
        directory = self._directory(run_id)
        path = (directory/filename).resolve()
        if path.parent != directory:
            raise ValueError('Artifact path escapes run directory')
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def _settings(self, raw):
        defaults = dict(mode='ppo',steps=65536,envs=8,rollout=256,history=8,seed=173,
                        learning_rate=3e-5,anchor_kl=.005,samples=2400,episodes=8,
                        epochs=30,horizon=40,linear_encoder=False,evaluate=True,init=None)
        if not isinstance(raw, dict) or set(raw)-set(defaults):
            raise ValueError('Unknown training settings')
        cfg = {**defaults, **raw}
        if cfg['mode'] not in ('ppo','teacher','evaluate','mpc'):
            raise ValueError('Unknown training mode')
        bounds = dict(steps=(1,100000000),envs=(1,64),rollout=(2,4096),history=(1,32),
                      seed=(0,2147483647),samples=(16,1000000),episodes=(2,10000),
                      epochs=(1,10000),horizon=(2,100))
        for name,(low,high) in bounds.items():
            value = cfg[name]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer between {low} and {high}')
        for name,low,high in [('learning_rate',1e-8,.01),('anchor_kl',0.,1.)]:
            value = cfg[name]
            if type(value) not in (float,int) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Invalid {name}')
        for name in ('linear_encoder','evaluate'):
            if type(cfg[name]) is not bool:
                raise ValueError(f'{name} must be true or false')
        if cfg['samples'] < cfg['episodes']:
            raise ValueError('Samples must be at least the episode count')
        if cfg['init'] is not None:
            source = cfg['init']
            if not isinstance(source,dict) or set(source) != {'run','file'} or not str(source['file']).endswith('.pt'):
                raise ValueError('Select a checkpoint from a saved run')
            self.artifact(source['run'],source['file'])
        if cfg['mode']=='evaluate' and cfg['init'] is None:
            raise ValueError('Choose a checkpoint to evaluate')
        return cfg

    def start(self, settings):
        with self.lock:
            cfg = self._settings(settings)
            if self.active is not None:
                raise RuntimeError('A training/evaluation job is already running')
            run_id = uuid.uuid4().hex
            directory = self._directory(run_id)
            directory.mkdir()
            command = [sys.executable,'-u']
            if cfg['mode'] in ('evaluate','mpc'):
                command += [str(SCRIPT_DIR/'evaluate_v4.py'),'--json',str(directory/'evaluation.json'),
                            '--horizon',str(cfg['horizon'])]
                if cfg['mode']=='mpc': command += ['--mpc']
                else: command += ['--model',str(self.artifact(cfg['init']['run'],cfg['init']['file']))]
            else:
                command += [str(SCRIPT_DIR/'train_residual_v4.py'),cfg['mode'],'--outdir',str(directory)]
                for name in ('steps','envs','rollout','history','seed','learning_rate','anchor_kl',
                             'samples','episodes','epochs','horizon'):
                    command += ['--'+name.replace('_','-'),str(cfg[name])]
                if cfg['linear_encoder']: command += ['--linear-encoder']
                if not cfg['evaluate']: command += ['--skip-evaluation']
                if cfg['init'] and cfg['mode']=='ppo':
                    command += ['--init',str(self.artifact(cfg['init']['run'],cfg['init']['file']))]
            job = dict(id=run_id,status='running',settings=cfg,started=time.time(),finished=None,
                       returncode=None,command=command)
            self._save(job)
            try:
                log = (directory/'output.log').open('w')
                try:
                    self.process = subprocess.Popen(command,cwd=SCRIPT_DIR.parent.parent,
                        stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
                finally:
                    log.close()
            except OSError as exc:
                job.update(status='failed',finished=time.time(),error=str(exc))
                self._save(job)
                raise RuntimeError(f'Could not start training: {exc}') from exc
            self.active = run_id
            self.worker = threading.Thread(target=self._wait,args=(run_id,self.process),daemon=True)
            self.worker.start()
            return self.get(run_id)

    def _wait(self, run_id, process):
        code = process.wait()
        with self.lock:
            job = self._read(run_id)
            job.update(status='cancelled' if job['status']=='cancelling' else ('completed' if code==0 else 'failed'),
                       returncode=code,finished=time.time())
            self._save(job)
            self.active = None
            self.process = None

    def cancel(self, run_id):
        with self.lock:
            job = self._read(run_id)
            if self.active == run_id and self.process is not None and self.process.poll() is None:
                job['status'] = 'cancelling'
                self._save(job)
                self.process.terminate()
                process = self.process
                def kill_if_needed():
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill()
                threading.Thread(target=kill_if_needed,daemon=True).start()
            return self.get(run_id)

    def get(self, run_id):
        with self.lock:
            job = self._read(run_id)
            directory = self._directory(run_id)
            job['artifacts'] = sorted(p.name for p in directory.iterdir()
                if p.is_file() and not p.is_symlink() and ARTIFACT.fullmatch(p.name))
            path = directory/'output.log'
            # Bound browser payload even for long overnight runs.
            with path.open('rb') if path.exists() else open(os.devnull,'rb') as stream:
                stream.seek(0,2)
                stream.seek(max(0,stream.tell()-65536))
                job['logs'] = stream.read().decode('utf-8',errors='replace')
            job['metrics'] = []
            for line in job['logs'].splitlines():
                if line.startswith('EVENT '):
                    try: job['metrics'].append(json.loads(line[6:]))
                    except ValueError: pass
                elif line.startswith('PPO '):
                    try: job['metrics'].append({'phase':'ppo',**json.loads(line[4:])})
                    except ValueError: pass
            job['evaluation'] = None
            if (directory/'evaluation.json').is_file():
                try: job['evaluation'] = json.loads((directory/'evaluation.json').read_text())
                except (ValueError,OSError): pass
            return self._json_safe(job)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, dict): return {k:TrainingJobs._json_safe(v) for k,v in value.items()}
        if isinstance(value, list): return [TrainingJobs._json_safe(v) for v in value]
        return value

    def list(self):
        jobs = []
        with self.lock:
            for directory in self.root.iterdir():
                if directory.is_dir() and not directory.is_symlink() and RUN_ID.fullmatch(directory.name):
                    try:
                        job = self._read(directory.name)
                        summary = {k:job[k] for k in ('id','status','settings','started','finished')}
                        summary['artifacts'] = sorted(p.name for p in directory.iterdir()
                            if p.is_file() and not p.is_symlink() and ARTIFACT.fullmatch(p.name))
                        jobs.append(summary)
                    except (OSError,ValueError,KeyError): pass
        return sorted(jobs,key=lambda job:job['started'],reverse=True)

    def close(self):
        with self.lock:
            if self.active: self.cancel(self.active)
            worker = self.worker
        if worker: worker.join(timeout=7)

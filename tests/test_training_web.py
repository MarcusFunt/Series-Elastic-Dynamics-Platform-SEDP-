"""Exercise real training jobs: persistence, cancellation and input boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))
from training_web import TrainingJobs


class TrainingJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jobs = TrainingJobs(Path(self.tmp.name))

    def tearDown(self):
        self.jobs.close()
        self.tmp.cleanup()

    def wait_done(self, run):
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            result = self.jobs.get(run['id'])
            if result['status'] not in ('running', 'cancelling'):
                return result
            time.sleep(.05)
        self.fail('Training did not finish')

    def test_real_ppo_saves_candidate_and_progress_without_acceptance(self):
        run = self.jobs.start({'mode':'ppo', 'steps':16, 'envs':1, 'rollout':16, 'evaluate':False})
        done = self.wait_done(run)
        self.assertEqual(done['status'], 'completed', done['logs'])
        self.assertEqual(done['metrics'][-1]['steps'], 16)
        self.assertIn('policy_candidate.pt', done['artifacts'])
        self.assertNotIn('policy_accepted.pt', done['artifacts'])
        self.assertIsNone(done['evaluation'])
        restarted = TrainingJobs(Path(self.tmp.name))
        self.assertEqual(restarted.get(run['id'])['status'], 'completed')
        restarted.close()

    def test_single_job_and_cancellation(self):
        run = self.jobs.start({'mode':'ppo', 'steps':100000, 'evaluate':False})
        with self.assertRaises(RuntimeError):
            self.jobs.start({'mode':'ppo'})
        self.jobs.cancel(run['id'])
        self.assertEqual(self.wait_done(run)['status'], 'cancelled')

    def test_invalid_settings_and_paths_do_not_start_jobs(self):
        for settings in ({'mode':'shell'}, {'steps':0}, {'history':0}, {'envs':True},
                         {'rollout':2.5}, {'learning_rate':float('nan')}, {'init':'/tmp/model.pt'},
                         {'mode':'teacher','episodes':1}, {'mode':'teacher','samples':2,'episodes':8}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.jobs.start(settings)
        self.assertEqual(self.jobs.list(), [])
        with self.assertRaises(ValueError):
            self.jobs.get('../escape')

    def test_artifacts_cannot_escape_run_directory(self):
        run = self.jobs.start({'mode':'ppo','steps':16,'envs':1,'rollout':16,'evaluate':False})
        self.wait_done(run)
        with self.assertRaises(ValueError):
            self.jobs.artifact(run['id'], '../job.json')
        outside = Path(self.tmp.name)/'outside.pt'
        outside.write_text('private')
        link = Path(self.tmp.name)/run['id']/'policy_bad.pt'
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.jobs.artifact(run['id'], 'policy_bad.pt')

class RuntimePolicyTests(unittest.TestCase):
    def test_snapshots_do_not_advance_estimator(self):
        from active_vibration_rig_web import Runtime
        from active_vibration_rig_2d import Simulator, PlantParams, ControllerParams, MotionParams
        from rig_rl_env_v4 import RLEnvConfigV4
        from ppo_agent_v2 import ActorCriticV4
        from rig_rl_policy_v4 import EstimatedResidualPolicyV4
        cfg = RLEnvConfigV4()
        sim = Simulator(PlantParams(), ControllerParams(), MotionParams(), cfg.physics_dt, cfg.control_dt)
        policy = EstimatedResidualPolicyV4(ActorCriticV4(208,1,128),cfg,sim.p,sim.cp,sim.trajectory)
        sim.controller.set_rl_policy(policy)
        sim.controller.mode = 'ppo'
        runtime = Runtime(sim)
        try:
            with runtime.lock:
                runtime.playing = False
                count = policy.counter
                runtime.snapshot()
                runtime.snapshot()
                self.assertEqual(policy.counter, count)
        finally:
            runtime.running = False
            runtime.thread.join(timeout=2)

class TrainingAPITests(unittest.TestCase):
    def test_real_job_api_load_and_rejection_of_cross_origin_commands(self):
        from fastapi.testclient import TestClient
        from active_vibration_rig_web import Runtime, create_app
        from active_vibration_rig_2d import Simulator, PlantParams, ControllerParams, MotionParams
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(Simulator(PlantParams(),ControllerParams(),MotionParams()))
            try:
                app = create_app(runtime,Path(directory))
                with TestClient(app) as client:
                    r = client.post('/api/training/runs',json={'mode':'ppo','steps':16,'envs':1,'rollout':16,'evaluate':False})
                    self.assertEqual(r.status_code,200,r.text)
                    run_id = r.json()['id']
                    deadline = time.monotonic()+40
                    while time.monotonic()<deadline:
                        d = client.get('/api/training/runs/'+run_id).json()
                        if d['status'] not in ('running','cancelling'): break
                        time.sleep(.05)
                    self.assertEqual(d['status'],'completed',d['logs'])
                    load = client.post('/api/training/runs/'+run_id+'/load',json={'file':'policy_candidate.pt'})
                    self.assertEqual(load.status_code,200,load.text)
                    with runtime.lock:
                        self.assertEqual(runtime.sim.controller.mode,'ppo')
                        self.assertEqual(runtime.sim.control_dt,.01)
                    self.assertEqual(client.get('/api/training/runs/'+run_id+'/artifacts/policy_candidate.pt').status_code,200)
                    self.assertEqual(client.post('/api/training/runs',headers={'Origin':'https://other.example'},json={}).status_code,403)
                    self.assertEqual(client.post('/api/training/runs',json={'steps':-1}).status_code,400)
                    self.assertEqual(client.get('/api/training/runs/'+'a'*32).status_code,404)
                    self.assertEqual(client.post('/api/training/runs/'+run_id+'/load',json={'file':['bad']}).status_code,400)
                    # Rejected/invalid evaluations must still render as valid JSON.
                    (Path(directory)/run_id/'evaluation.json').write_text('{"promotion":{"accepted":false,"mean_energy_ratio":Infinity,"reasons":["invalid metrics"]},"rows":[]}')
                    self.assertEqual(client.get('/api/training/runs/'+run_id).json()['evaluation']['promotion']['mean_energy_ratio'],None)
            finally:
                runtime.running = False
                runtime.thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()

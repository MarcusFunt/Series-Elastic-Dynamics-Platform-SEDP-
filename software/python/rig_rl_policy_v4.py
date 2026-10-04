"""Runtime adapter using the same estimator, context and held-command loop."""
import numpy as np
from residual_control_v4 import ResidualControlLoop


class EstimatedResidualPolicyV4:
    def __init__(self,model,cfg,plant_params,controller_params,reference):
        if cfg.oracle_state:raise ValueError('Oracle-state checkpoints are diagnostic only')
        self.model=model;self.cfg=cfg;self.p=plant_params;self.cp=controller_params;self.reference=reference
        self.reset()

    def reset(self):
        self.loop=ResidualControlLoop(self.p,self.cp,self.cfg,seed=7919)
        self.counter=0;self.previous_torque=0.

    def command(self,true_state,ref):
        import torch
        time=self.counter*self.cfg.control_dt
        if self.counter:self.loop.predict(self.previous_torque)
        obs=self.loop.observe(true_state,time,self.reference,initial=self.counter==0)
        with torch.no_grad():action=self.model.deterministic(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
        self.previous_torque=self.loop.command(action,ref)
        self.counter+=1
        return self.previous_torque

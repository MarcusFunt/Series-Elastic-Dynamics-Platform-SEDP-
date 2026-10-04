"""Shared observable-state residual control and context/history encoding."""
from collections import deque
from dataclasses import replace
import numpy as np
from active_vibration_rig_2d import Controller, RigPlant
from state_estimator import MeasurementModel, StateEstimator

# 15 present features, 9 known-reference preview features, uncertainty and age.
FRAME_SIGNS = np.array([-1.]*26, dtype=np.float32)
FRAME_SIGNS[[11,24,25]] = 1.
PREVIEW_TIMES = (.05,.10,.20)


class ObservationHistory:
    def __init__(self, length=8):
        if length < 1: raise ValueError('history length must be positive')
        self.length=length; self.frames=deque(maxlen=length)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.length): self.frames.append(np.asarray(frame,np.float32).copy())
        return self.value

    def append(self, frame):
        if not self.frames: return self.reset(frame)
        self.frames.append(np.asarray(frame,np.float32).copy())
        return self.value

    @property
    def value(self): return np.concatenate(self.frames).astype(np.float32)


class ResidualControlLoop:
    def __init__(self, nominal_params, controller_params, cfg, seed=0, measurement_params=None):
        self.p=replace(nominal_params); self.cfg=cfg
        self.controller=Controller(RigPlant(self.p),controller_params,cfg.physics_dt)
        self.measurements=MeasurementModel(measurement_params or nominal_params,cfg.sensor_noise,cfg.linear_encoder,seed)
        self.estimator=StateEstimator(self.p,cfg.physics_dt,cfg.control_dt,cfg.linear_encoder)
        self.history=ObservationHistory(cfg.history_length)
        self.previous_effective_action=0.; self.state=np.zeros(7); self.last_measurement_time=0.
        self.last_command={}

    def observe(self, true_state, time, reference, initial=False):
        z=self.measurements.sample(true_state,time)
        self.estimator.update(z)
        self.state=true_state.copy() if self.cfg.oracle_state else self.estimator.state
        self.last_measurement_time=z['time']
        frame=self.frame(reference,time)
        return self.history.reset(frame) if initial else self.history.append(frame)

    def base_accel(self, ref): return self.controller.modern_accel(self.state,ref,'lqr')

    def frame(self, reference, time):
        c,p,y=self.cfg,self.p,self.state; xr,vr,ar=reference.sample(time)
        L=p.rail_half_travel; base=self.base_accel((xr,vr,ar))
        values=[(y[2]-xr)/c.pos_scale,(y[3]-vr)/c.vel_scale,
                y[4]/c.theta_scale,y[5]/c.theta_dot_scale,y[2]/L,y[3]/c.vel_scale,
                base/c.accel_scale,y[1]/c.motor_speed_scale,y[6]/p.motor_hold_torque,
                ar/c.accel_scale,self.previous_effective_action,1-abs(y[2])/L,
                (p.pulley_radius*y[0]-y[2])/.001,xr/L,vr/c.vel_scale]
        preview=getattr(reference,'preview',reference.sample)
        for offset in PREVIEW_TIMES:
            px,pv,pa=preview(time+offset)
            values.extend([(px-xr)/c.pos_scale,pv/c.vel_scale,pa/c.accel_scale])
        values.extend([np.sqrt(max(self.estimator.covariance[2,2],0))/.001,
                       max(0.,time-self.last_measurement_time)/c.control_dt])
        return np.clip(values,-8,8).astype(np.float32)

    def command(self, action, ref):
        raw=float(np.asarray(action).reshape(-1)[0])
        if not np.isfinite(raw): raise ValueError('Residual action must be finite')
        requested=float(np.clip(raw,-1,1))*self.cfg.residual_accel_limit
        base=self.base_accel(ref)
        total=self.controller.project_accel(self.state,base+requested)
        torque=self.controller.torque_from_accel(self.state,total)
        self.previous_effective_action=(total-base)/self.cfg.residual_accel_limit
        self.last_command={'base_accel':base,'requested_residual_accel':requested,
                           'effective_residual_accel':total-base,'total_accel':total,'tau_cmd':torque}
        return torque

    def predict(self, torque): self.estimator.predict(torque)

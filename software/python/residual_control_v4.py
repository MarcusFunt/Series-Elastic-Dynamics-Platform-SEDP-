"""Shared observable-state residual control and context/history encoding."""
from collections import deque
from dataclasses import replace
import numpy as np
from active_vibration_rig_2d import Controller, RigPlant
from state_estimator import MeasurementModel, StateEstimator
from timing_models import AsynchronousSensorSuite, has_synchronous_sensor_timing, normalize_sensor_timing

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
        self.sensor_timing=normalize_sensor_timing(
            getattr(cfg,'sensor_timing',{}),cfg.control_dt,cfg.linear_encoder,cfg.physics_dt)
        self.synchronous_sensors=has_synchronous_sensor_timing(
            self.sensor_timing,cfg.control_dt)
        self.sensor_suite=None if self.synchronous_sensors else AsynchronousSensorSuite(
            measurement_params or nominal_params,self.sensor_timing,cfg.sensor_noise,seed+32452843)
        history_horizon=(self.sensor_suite.max_latency+cfg.control_dt
                         if self.sensor_suite is not None else 0.0)
        self.estimator=StateEstimator(self.p,cfg.physics_dt,cfg.control_dt,cfg.linear_encoder,
                                      history_horizon=history_horizon)
        self.history=ObservationHistory(cfg.history_length)
        initial_measurement_time=(0.0 if self.synchronous_sensors else -cfg.control_dt)
        self.previous_effective_action=0.; self.state=np.zeros(7)
        self.last_measurement_time=initial_measurement_time
        self.last_command={}
        self.pending_sensor_packets=[]
        self.sensor_suite_started=False
        self.sensor_messages_received=0
        self.sensor_messages_processed=0
        self.delayed_measurements_processed=0
        self.last_measurement_packets=[]

    def observe(self, true_state, time, reference, initial=False):
        if self.synchronous_sensors:
            z=self.measurements.sample(true_state,time)
            self.estimator.update(z)
            self.last_measurement_time=z['time']
            self.last_measurement_packets=[{
                'acquisition_time':z['time'],'arrival_time':z['time'],
                'channels':tuple(key for key in z if key!='time')}]
            self.sensor_messages_received+=len(self.last_measurement_packets[0]['channels'])
            self.sensor_messages_processed+=len(self.last_measurement_packets[0]['channels'])
        else:
            if not self.sensor_suite_started:
                self.sensor_suite.reset()
                self.sensor_suite_started=True
            self.sensor_tick(true_state,time)
            due=sorted(self.pending_sensor_packets,
                       key=lambda packet:(packet['arrival_time'],packet['sequence']))
            self.pending_sensor_packets=[]
            self.last_measurement_packets=[dict(packet) for packet in due]
            grouped={}
            for packet in due:
                key=(round(packet['acquisition_time'],12),round(packet['arrival_time'],12))
                grouped.setdefault(key,{'time':packet['acquisition_time'],
                                        'acquisition_time':packet['acquisition_time'],
                                        'arrival_time':packet['arrival_time']})
                grouped[key][packet['channel']]=packet['value']
                self.last_measurement_time=max(self.last_measurement_time,
                                               packet['acquisition_time'])
            for measurement in grouped.values():
                self.estimator.update(measurement)
                if measurement['acquisition_time'] < time-1e-12:
                    self.delayed_measurements_processed += sum(
                        key in measurement for key in ('motor_angle','lever_angle','gyro','carriage_position'))
                self.sensor_messages_processed+=sum(
                    key in measurement for key in ('motor_angle','lever_angle','gyro','carriage_position'))
        self.state=true_state.copy() if self.cfg.oracle_state else self.estimator.state
        frame=self.frame(reference,time)
        return self.history.reset(frame) if initial else self.history.append(frame)

    def sensor_tick(self,true_state,time):
        if self.sensor_suite is None:
            return
        if not self.sensor_suite_started:
            self.sensor_suite.reset()
            self.sensor_suite_started=True
        packets=self.sensor_suite.tick(true_state,time)
        self.pending_sensor_packets.extend(packets)
        self.sensor_messages_received+=len(packets)

    def base_accel(self, ref): return self.controller.modern_accel(self.state,ref,'lqr')

    def frame(self, reference, time):
        c,p,y=self.cfg,self.p,self.state; xr,vr,ar=reference.sample(time)
        L=p.rail_half_travel; base=self.base_accel((xr,vr,ar))
        values=[(y[2]-xr)/c.pos_scale,(y[3]-vr)/c.vel_scale,
                y[4]/c.theta_scale,y[5]/c.theta_dot_scale,y[2]/L,y[3]/c.vel_scale,
                base/c.accel_scale,y[1]/c.motor_speed_scale,y[6]/p.motor_hold_torque,
                ar/c.accel_scale,self.previous_effective_action,1-abs(y[2])/L,
                (p.pulley_radius*y[0]-y[2])/.001,xr/L,vr/c.vel_scale]
        if c.preview_enabled:
            preview=getattr(reference,'preview',reference.sample)
            for offset in PREVIEW_TIMES:
                px,pv,pa=preview(time+offset)
                values.extend([(px-xr)/c.pos_scale,pv/c.vel_scale,pa/c.accel_scale])
        else:
            values.extend([0.0] * (3 * len(PREVIEW_TIMES)))
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

"""Encoder/gyro measurements and an eight-state extended Kalman filter.

The controller uses nominal parameters, never randomized ground-truth parameters.
Carriage position is inferred through the belt model unless a linear encoder is
explicitly enabled. This inference assumes a homed motor and calibrated pulley.
"""
from dataclasses import dataclass
import math
import numpy as np
from active_vibration_rig_2d import PlantParams, RigPlant, quantize_angle_12bit, wrap_pi


@dataclass
class MeasurementModel:
    params: PlantParams
    noise: bool = True
    linear_encoder: bool = False
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(self, true_state, time):
        p = self.params
        gyro_noise = self.rng.normal(0, p.gyro_noise_std) if self.noise else 0.
        z = {'motor_angle': quantize_angle_12bit(float(true_state[0])),
             'lever_angle': quantize_angle_12bit(float(true_state[4])),
             'gyro': float(true_state[5])+p.gyro_bias+gyro_noise, 'time': float(time)}
        if self.linear_encoder:
            z['carriage_position'] = float(true_state[2])+(self.rng.normal(0, 2e-5) if self.noise else 0.)
        return z


def held_transition(plant, state, torque, physics_dt, control_dt):
    y = np.asarray(state, dtype=float).copy()
    for _ in range(round(control_dt/physics_dt)):
        y = plant.rk4(y, torque, physics_dt)
    return y


def numerical_jacobian(function, point):
    point = np.asarray(point, dtype=float)
    columns = []
    for i in range(len(point)):
        eps = 1e-5*max(1., abs(point[i]))
        delta = np.zeros_like(point); delta[i] = eps
        columns.append((function(point+delta)-function(point-delta))/(2*eps))
    return np.stack(columns, axis=1)


class StateEstimator:
    """EKF on [phi, motor_rate, x, velocity, theta, rate, torque, gyro_bias]."""
    def __init__(self, params, physics_dt=.001, control_dt=.01, linear_encoder=False):
        if physics_dt <= 0 or control_dt < physics_dt or not math.isclose(control_dt/physics_dt, round(control_dt/physics_dt), abs_tol=1e-9):
            raise ValueError('Invalid estimator timing')
        self.params = params; self.plant = RigPlant(params)
        self.physics_dt = physics_dt; self.control_dt = control_dt
        self.linear_encoder = linear_encoder
        self.reset()

    def reset(self):
        self.mean = np.zeros(8)
        self.covariance = np.diag(np.square([.002, .5, .002, .08, .04, .2, .03, .03]))
        self.time = 0.; self.innovation_norm = 0.

    @property
    def state(self):
        return self.mean[:7].copy()

    def predict(self, torque):
        def transition(s):
            return np.r_[held_transition(self.plant,s[:7],torque,self.physics_dt,self.control_dt),s[7]]
        F = numerical_jacobian(transition, self.mean)
        self.mean = transition(self.mean)
        Q = np.diag(np.square([2e-5,.06,3e-5,.025,2e-4,.025,.008,1e-4]))
        self.covariance = F@self.covariance@F.T+Q
        self.time += self.control_dt

    def update(self, measurement):
        keys = ['motor_angle','lever_angle','gyro']
        H = np.zeros((3,8)); H[0,0]=1.; H[1,4]=1.; H[2,5]=1.; H[2,7]=1.
        variance = [(.001534/np.sqrt(12))**2, (.001534/np.sqrt(12))**2,
                    max(self.params.gyro_noise_std, .001)**2]
        if self.linear_encoder:
            keys.append('carriage_position'); H=np.vstack([H,np.eye(8)[2]])
            variance.append((2e-5)**2)
        innovation = np.array([measurement[k] for k in keys])-H@self.mean
        innovation[0]=wrap_pi(innovation[0]); innovation[1]=wrap_pi(innovation[1])
        R=np.diag(variance); P=self.covariance; S=H@P@H.T+R
        K=np.linalg.solve(S,H@P).T
        self.mean += K@innovation
        I=np.eye(8)-K@H
        self.covariance=I@P@I.T+K@R@K.T
        self.covariance=(self.covariance+self.covariance.T)*.5
        self.innovation_norm=float(np.sqrt(innovation@np.linalg.solve(S,innovation)))
        self.time=float(measurement['time'])
        return self.state

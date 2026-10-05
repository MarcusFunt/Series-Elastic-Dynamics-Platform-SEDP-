"""Locally linearized constrained MPC reference and residual teacher.

Predictions include all seven plant states and held torque commands. Rail,
velocity, conservative speed-dependent motor torque, and residual-authority
constraints are enforced in the QP. A nonlinear rollout checks the candidate;
solver/model-check failure returns zero residual to the shared LQR controller.
This is a simulation reference, not a certified hardware safety controller.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from active_vibration_rig_2d import Controller, RigPlant
from state_estimator import held_transition, numerical_jacobian


@dataclass
class LegacyMPCConfig:
    horizon: int = 40
    physics_dt: float = .001
    control_dt: float = .01
    residual_accel_limit: float = 2.5
    max_iterations: int = 60
    tracking_scale: float = .006
    angle_scale: float = np.radians(5.)
    action_weight: float = .03
    slew_weight: float = .06

    def __post_init__(self):
        if self.horizon<2 or self.residual_accel_limit<=0 or self.max_iterations<1:
            raise ValueError('Invalid MPC configuration')
        if self.physics_dt<=0 or self.control_dt<self.physics_dt or not np.isclose(self.control_dt/self.physics_dt,round(self.control_dt/self.physics_dt)):
            raise ValueError('MPC timing must have an integer substep ratio')


class LegacyConstrainedMPC:
    def __init__(self, params, controller_params, cfg=None):
        self.p=params; self.cp=controller_params;self.cfg=cfg or LegacyMPCConfig()
        self.plant=RigPlant(params);self.controller=Controller(self.plant,controller_params,self.cfg.physics_dt)
        self.previous_accel=0.;self.warm=np.zeros(self.cfg.horizon);self.diagnostics={}

    def _transition(self,y,a):
        torque=self.controller.torque_from_accel(y,float(a))
        return held_transition(self.plant,y,torque,self.cfg.physics_dt,self.cfg.control_dt)

    def action(self,state,references):
        c,p,cp=self.cfg,self.p,self.cp;n=c.horizon
        y=np.asarray(state,dtype=float)
        refs=np.asarray(references,dtype=float)
        if y.shape!=(7,) or refs.shape!=(n+1,3) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(refs)):
            raise ValueError('MPC requires finite seven-state input and horizon+1 references')
        started=time.perf_counter();base=self.controller.modern_accel(y,tuple(refs[0]),'lqr')
        F=numerical_jacobian(lambda s:self._transition(s,base),y)
        eps=1e-4;B=(self._transition(y,base+eps)-self._transition(y,base-eps))/(2*eps)
        d=self._transition(y,base)-F@y-B*base
        torque0=self.controller.torque_from_accel(y,base)
        torque_y=numerical_jacobian(lambda s:np.array([self.controller.torque_from_accel(s,base)]),y)[0]
        torque_a=(self.controller.torque_from_accel(y,base+eps)-self.controller.torque_from_accel(y,base-eps))/(2*eps)
        # Conservative lower torque envelope across the allowed motor-speed range.
        wm_limit=cp.max_speed/p.pulley_radius
        torque_limit=self.plant.motor_torque_limit(wm_limit)
        Q=np.diag([0.,0.,1.8/c.tracking_scale**2,.12/.35**2,
                   2.4/c.angle_scale**2,.55/np.radians(100.)**2,0.])
        H=np.zeros((n,n));g=np.zeros(n);rows=[];lower=[];upper=[]
        S=np.zeros((7,n));offset=y.copy();eye=np.eye(n)
        def constrain(row,lo,hi):
            rows.append(row);lower.append(lo);upper.append(hi)
        for k in range(n):
            ref=tuple(refs[k]);b=self.controller.modern_accel(y,ref,'lqr')
            bj=numerical_jacobian(lambda s:np.array([self.controller.modern_accel(s,ref,'lqr')]),y)[0]
            nominal=b+bj@(offset-y)
            residual_row=eye[k]-bj@S
            constrain(residual_row,nominal-c.residual_accel_limit,nominal+c.residual_accel_limit)
            tq_offset=torque0+torque_y@(offset-y)-torque_a*base
            tq_row=torque_y@S+torque_a*eye[k]
            constrain(tq_row,-torque_limit-tq_offset,torque_limit-tq_offset)
            H+=c.action_weight/c.residual_accel_limit**2*np.outer(residual_row,residual_row)
            g-=c.action_weight/c.residual_accel_limit**2*nominal*residual_row
            S=F@S+np.outer(B,eye[k]);offset=F@offset+d
            target=np.array([0.,0.,refs[k+1,0],refs[k+1,1],0.,0.,0.])
            weight=2. if k==n-1 else 1.
            H+=weight*S.T@Q@S;g+=weight*S.T@Q@(offset-target)
            constrain(S[2],-cp.rail_soft_fraction*p.rail_half_travel-offset[2],cp.rail_soft_fraction*p.rail_half_travel-offset[2])
            constrain(S[3],-cp.max_speed-offset[3],cp.max_speed-offset[3])
            constrain(S[1],-wm_limit-offset[1],wm_limit-offset[1])
        D=eye.copy();D[1:]-=eye[:-1]
        prev=np.zeros(n);prev[0]=self.previous_accel
        H+=c.slew_weight/c.residual_accel_limit**2*(D.T@D)
        g-=c.slew_weight/c.residual_accel_limit**2*D.T@prev
        lo,hi=self.controller.rail_accel_bounds(y)
        lb=np.full(n,-cp.max_accel);ub=np.full(n,cp.max_accel)
        lb[0]=max(lo,base-c.residual_accel_limit);ub[0]=min(hi,base+c.residual_accel_limit)
        if lb[0]>ub[0]:
            return self._fallback(started,'infeasible_initial_bounds','infeasible initial bounds')
        scale=max(float(np.max(np.diag(H))),1.)
        H/=scale;g/=scale
        result=minimize(lambda u:float(u@H@u+2*g@u),np.clip(self.warm,lb,ub),
                        jac=lambda u:2*(H@u+g),method='SLSQP',bounds=Bounds(lb,ub),
                        constraints=[LinearConstraint(np.asarray(rows),np.asarray(lower),np.asarray(upper))],
                        options={'maxiter':c.max_iterations,'ftol':1e-8})
        violation=max(float(np.max(np.asarray(lower)-np.asarray(rows)@result.x)),
                      float(np.max(np.asarray(rows)@result.x-np.asarray(upper))),0.)
        if not result.success or violation>1e-5:
            outcome=('iteration_limit' if int(getattr(result,'status',-1))==9 else
                     'infeasible_constraints' if int(getattr(result,'status',-1))==4 else
                     'numerical_solver_error' if int(getattr(result,'status',-1)) in (5,6,7) else
                     'solver_failure')
            return self._fallback(started,outcome,str(result.message))
        # Verify the model and actuator constraints against a nonlinear rollout.
        sy=y.copy()
        for k,a in enumerate(result.x):
            nonlinear_base=self.controller.modern_accel(sy,tuple(refs[k]),'lqr')
            torque=self.controller.torque_from_accel(sy,a)
            if abs(a-nonlinear_base)>c.residual_accel_limit+1e-3 or abs(torque)>torque_limit+1e-5:
                return self._fallback(started,'nonlinear_actuator_rejection','nonlinear actuator/authority check')
            sy=self._transition(sy,a)
            if abs(sy[2])>cp.rail_soft_fraction*p.rail_half_travel+1e-5 or abs(sy[3])>cp.max_speed+1e-4 or abs(sy[1])>wm_limit+1e-3:
                return self._fallback(started,'nonlinear_state_rejection','nonlinear rail/speed check')
        self.warm=np.r_[result.x[1:],result.x[-1]]
        self.previous_accel=float(result.x[0])
        self.diagnostics={'success':True,'outcome':'success','category':'success','reason':'',
                          'fallback_action':None,'fallback_reason':'','solve_seconds':time.perf_counter()-started,
                          'iterations':int(result.nit),'constraint_violation':violation}
        return float(np.clip((result.x[0]-base)/c.residual_accel_limit,-1,1))

    def _fallback(self,started,outcome,reason):
        self.warm[:]=0.
        self.diagnostics={'success':False,'outcome':outcome,'category':outcome,'reason':reason,
                          'fallback_action':0.,'fallback_reason':reason,
                          'solve_seconds':time.perf_counter()-started,'iterations':0}
        return 0.

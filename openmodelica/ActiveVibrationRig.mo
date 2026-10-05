within ;
package ActiveVibrationRig
  "OpenModelica model of the RP2350/stepper/GT2 active-vibration bench rig"
  import SI = Modelica.Units.SI;
  import Modelica.Constants.pi;

  type MotionMode = enumeration(
    Hold "Zero target",
    Sine "Single-frequency sinusoid",
    Aggressive "Two-tone high-acceleration trajectory");

  type ControllerMode = enumeration(
    Baseline "Carriage servo only; no resonator feedback",
    ActiveDamping "Carriage servo plus theta/thetaDot feedback");

  record RigParameters
    "Nominal physical parameters. Values are placeholders until hardware identification."
    parameter SI.Mass carriageMass = 0.45;
    parameter SI.Mass resonatorMass = 0.12;
    parameter SI.Length leverCOM = 0.095;
    parameter SI.Inertia leverInertiaCOM = 1.8e-4;
    parameter SI.Inertia resonatorInertiaPivot = leverInertiaCOM + resonatorMass*leverCOM^2;

    parameter SI.RotationalSpringConstant kTheta = 0.42;
    parameter SI.RotationalDampingConstant cTheta = 0.006;
    parameter Real kTheta3(unit="N.m/rad3") = 0;

    parameter Boolean useGeometricSprings = true;
    parameter SI.Length springAnchorHalfSpacing = 0.035;
    parameter SI.Length springAnchorY = -0.015;
    parameter SI.Length springShaftRadius = 0.040;
    parameter SI.TranslationalSpringConstant springK = 500;
    parameter SI.TranslationalDampingConstant springD = 1.0;
    parameter SI.Length springFreeLength = 0.010;

    parameter SI.TranslationalDampingConstant bX = 1.2;
    parameter SI.Force xCoulomb = 0.30;
    parameter SI.Velocity xFrictionEps = 0.015;

    parameter SI.Inertia motorInertia = 5e-5;
    parameter SI.RotationalDampingConstant motorViscous = 2e-4;
    parameter SI.Torque motorCoulomb = 0.007;
    parameter SI.AngularVelocity motorFrictionEps = 0.5;
    parameter SI.Torque motorHoldTorque = 0.48;
    parameter SI.AngularVelocity motorOmegaCorner = 55;
    parameter Real motorTorqueFloor(min=0,max=1) = 0.15;
    parameter SI.Time motorTorqueTimeConstant = 0.0015;

    parameter Integer pulleyTeeth = 20;
    parameter SI.Length beltPitch = 0.002;
    parameter SI.Length pulleyRadius = pulleyTeeth*beltPitch/(2*pi);
    parameter SI.TranslationalSpringConstant beltStiffness = 6500;
    parameter SI.TranslationalDampingConstant beltDamping = 18;
    parameter Real beltCubic(unit="N/m3") = 0;

    parameter SI.Length railHalfTravel = 0.065;
    parameter SI.TranslationalSpringConstant stopStiffness = 18000;
    parameter SI.TranslationalDampingConstant stopDamping = 120;

    parameter SI.Torque thetaCoulomb = 0.0015;
    parameter SI.AngularVelocity thetaFrictionEps = 0.05;
    parameter SI.Acceleration g = 9.81;
    parameter SI.Length imuRadius = 0.075;
  end RigParameters;

  function smoothSat
    input Real u;
    input Real limit(min=Modelica.Constants.small);
    output Real y;
  algorithm
    y := limit*tanh(u/limit);
  end smoothSat;

  model RigPlant
    parameter RigParameters p = RigParameters();
    parameter SI.Angle phiMotor_start = 0;
    parameter SI.AngularVelocity omegaMotor_start = 0;
    parameter SI.Position x_start = 0;
    parameter SI.Velocity v_start = 0;
    parameter SI.Angle theta_start = 0;
    parameter SI.AngularVelocity thetaDot_start = 0;

    Modelica.Blocks.Interfaces.RealInput tauCmd(unit="N.m");
    Modelica.Blocks.Interfaces.RealInput tauExt(unit="N.m");

    Modelica.Blocks.Interfaces.RealOutput phiMotor(unit="rad");
    Modelica.Blocks.Interfaces.RealOutput omegaMotor(unit="rad/s");
    Modelica.Blocks.Interfaces.RealOutput x(unit="m");
    Modelica.Blocks.Interfaces.RealOutput v(unit="m/s");
    Modelica.Blocks.Interfaces.RealOutput xDDot(unit="m/s2");
    Modelica.Blocks.Interfaces.RealOutput theta(unit="rad");
    Modelica.Blocks.Interfaces.RealOutput thetaDot(unit="rad/s");
    Modelica.Blocks.Interfaces.RealOutput thetaDDot(unit="rad/s2");
    Modelica.Blocks.Interfaces.RealOutput tauAct(unit="N.m");
    Modelica.Blocks.Interfaces.RealOutput beltForce(unit="N");
    Modelica.Blocks.Interfaces.RealOutput beltExtension(unit="m");
    Modelica.Blocks.Interfaces.RealOutput springTorque(unit="N.m");
    Modelica.Blocks.Interfaces.BooleanOutput leftLimit;
    Modelica.Blocks.Interfaces.BooleanOutput rightLimit;

  protected
    SI.AngularVelocity wM(start=omegaMotor_start);
    SI.Angle phiM(start=phiMotor_start);
    SI.Position xState(start=x_start);
    SI.Velocity vState(start=v_start);
    SI.Angle th(start=theta_start);
    SI.AngularVelocity wTh(start=thetaDot_start);
    SI.Torque tauState(start=0);

    SI.Force Fb;
    SI.Position deltaB;
    SI.Velocity deltaBDot;
    SI.Torque tauLimit;
    SI.Torque tauTarget;
    SI.Torque tauMotorLoss;
    SI.Force FxLoss;
    SI.Force FStop;

    Real A11(unit="kg");
    Real A12(unit="kg.m");
    Real A22(unit="kg.m2");
    Real detA(unit="kg2.m2");
    SI.Force rhsX;
    SI.Torque rhsTheta;
    SI.Acceleration ax;
    SI.AngularAcceleration ath;

    SI.Position xs;
    SI.Position ys;
    SI.Velocity vxs;
    SI.Velocity vys;
    SI.Length qLx;
    SI.Length qLy;
    SI.Length qRx;
    SI.Length qRy;
    SI.Length lL;
    SI.Length lR;
    SI.Velocity lDotL;
    SI.Velocity lDotR;
    SI.Force magL;
    SI.Force magR;
    SI.Force fLx;
    SI.Force fLy;
    SI.Force fRx;
    SI.Force fRy;
    SI.Torque tauSpringGeom;
    SI.Torque tauSpringEquivalent;
    SI.Torque tauSpring;

  equation
    phiMotor = phiM;
    omegaMotor = wM;
    x = xState;
    v = vState;
    theta = th;
    thetaDot = wTh;
    tauAct = tauState;
    xDDot = ax;
    thetaDDot = ath;
    beltForce = Fb;
    beltExtension = deltaB;
    springTorque = tauSpring;

    der(phiM) = wM;
    der(xState) = vState;
    der(th) = wTh;

    deltaB = p.pulleyRadius*phiM - xState;
    deltaBDot = p.pulleyRadius*wM - vState;
    Fb = p.beltStiffness*deltaB + p.beltDamping*deltaBDot + p.beltCubic*deltaB^3;

    tauLimit = p.motorHoldTorque*max(p.motorTorqueFloor, 1 - abs(wM)/p.motorOmegaCorner);
    tauTarget = smoothSat(tauCmd, tauLimit);
    der(tauState) = (tauTarget - tauState)/p.motorTorqueTimeConstant;

    tauMotorLoss = p.motorViscous*wM + p.motorCoulomb*tanh(wM/p.motorFrictionEps);
    p.motorInertia*der(wM) = tauState - tauMotorLoss - p.pulleyRadius*Fb;

    FxLoss = p.bX*vState + p.xCoulomb*tanh(vState/p.xFrictionEps);
    FStop = if noEvent(xState > p.railHalfTravel) then
        -p.stopStiffness*(xState - p.railHalfTravel) -p.stopDamping*max(vState,0)
      elseif noEvent(xState < -p.railHalfTravel) then
        -p.stopStiffness*(xState + p.railHalfTravel) -p.stopDamping*min(vState,0)
      else 0;

    xs = p.springShaftRadius*sin(th);
    ys = p.springShaftRadius*cos(th);
    vxs = p.springShaftRadius*cos(th)*wTh;
    vys = -p.springShaftRadius*sin(th)*wTh;

    qLx = xs + p.springAnchorHalfSpacing;
    qLy = ys - p.springAnchorY;
    qRx = xs - p.springAnchorHalfSpacing;
    qRy = ys - p.springAnchorY;
    lL = sqrt(qLx^2 + qLy^2);
    lR = sqrt(qRx^2 + qRy^2);
    lDotL = (qLx*vxs + qLy*vys)/max(lL,1e-9);
    lDotR = (qRx*vxs + qRy*vys)/max(lR,1e-9);

    magL = p.springK*(lL - p.springFreeLength) + p.springD*lDotL;
    magR = p.springK*(lR - p.springFreeLength) + p.springD*lDotR;
    fLx = -magL*qLx/max(lL,1e-9);
    fLy = -magL*qLy/max(lL,1e-9);
    fRx = -magR*qRx/max(lR,1e-9);
    fRy = -magR*qRy/max(lR,1e-9);
    // Generalized torque for x=r*sin(th), y=r*cos(th): F dot d(position)/d(th).
    tauSpringGeom = ys*(fLx + fRx) - xs*(fLy + fRy);

    tauSpringEquivalent = -p.kTheta*th - p.kTheta3*th^3 - p.cTheta*wTh;
    tauSpring = if p.useGeometricSprings then tauSpringGeom else tauSpringEquivalent;

    A11 = p.carriageMass + p.resonatorMass;
    A12 = p.resonatorMass*p.leverCOM*cos(th);
    A22 = p.resonatorInertiaPivot;
    detA = A11*A22 - A12^2;

    rhsX = Fb + FStop - FxLoss + p.resonatorMass*p.leverCOM*sin(th)*wTh^2;
    rhsTheta = tauSpring - p.thetaCoulomb*tanh(wTh/p.thetaFrictionEps)
      + tauExt + p.resonatorMass*p.g*p.leverCOM*sin(th);

    ax = (rhsX*A22 - A12*rhsTheta)/detA;
    ath = (A11*rhsTheta - A12*rhsX)/detA;
    der(vState) = ax;
    der(wTh) = ath;

    leftLimit = xState <= -p.railHalfTravel;
    rightLimit = xState >= p.railHalfTravel;
  end RigPlant;

  block AS5600
    parameter Integer bits = 12;
    parameter SI.Time samplePeriod = 0.001;
    parameter SI.Time lag = 0.00030;
    Modelica.Blocks.Interfaces.RealInput angle(unit="rad");
    Modelica.Blocks.Interfaces.RealOutput raw(unit="rad");
    Modelica.Blocks.Interfaces.RealOutput centered(unit="rad");
  protected
    Real filtered(start=0, unit="rad");
    discrete Real held(start=0, unit="rad");
    parameter Real levels = 2^bits;
    Real wrapped(unit="rad");
  equation
    der(filtered) = (angle - filtered)/lag;
    wrapped = mod(filtered, 2*pi);
    raw = held;
    centered = if held > pi then held - 2*pi else held;
    when sample(0, samplePeriod) then
      held = floor(wrapped/(2*pi)*levels + 0.5)/levels*(2*pi);
    end when;
  end AS5600;

  block LeverIMU
    parameter SI.Length radius = 0.075;
    parameter SI.Acceleration g = 9.81;
    Modelica.Blocks.Interfaces.RealInput theta(unit="rad");
    Modelica.Blocks.Interfaces.RealInput thetaDot(unit="rad/s");
    Modelica.Blocks.Interfaces.RealInput thetaDDot(unit="rad/s2");
    Modelica.Blocks.Interfaces.RealInput xDDot(unit="m/s2");
    Modelica.Blocks.Interfaces.RealOutput gyro(unit="rad/s");
    Modelica.Blocks.Interfaces.RealOutput accelTangential(unit="m/s2");
    Modelica.Blocks.Interfaces.RealOutput accelLongitudinal(unit="m/s2");
  protected
    SI.Acceleration axWorld;
    SI.Acceleration ayWorld;
    SI.Acceleration fxWorld;
    SI.Acceleration fyWorld;
  equation
    gyro = thetaDot;
    axWorld = xDDot + radius*cos(theta)*thetaDDot - radius*sin(theta)*thetaDot^2;
    ayWorld = -radius*sin(theta)*thetaDDot - radius*cos(theta)*thetaDot^2;
    fxWorld = axWorld;
    fyWorld = ayWorld + g;
    accelTangential = fxWorld*cos(theta) - fyWorld*sin(theta);
    accelLongitudinal = fxWorld*sin(theta) + fyWorld*cos(theta);
  end LeverIMU;

  block TrajectoryGenerator
    parameter MotionMode mode = MotionMode.Aggressive;
    parameter SI.Position amplitude = 0.035;
    parameter SI.Frequency f1 = 0.8;
    parameter SI.Frequency f2 = 1.7;
    Modelica.Blocks.Interfaces.RealOutput xRef(unit="m");
    Modelica.Blocks.Interfaces.RealOutput vRef(unit="m/s");
    Modelica.Blocks.Interfaces.RealOutput aRef(unit="m/s2");
  equation
    if mode == MotionMode.Hold then
      xRef = 0; vRef = 0; aRef = 0;
    elseif mode == MotionMode.Sine then
      xRef = amplitude*sin(2*pi*f1*time);
      vRef = amplitude*(2*pi*f1)*cos(2*pi*f1*time);
      aRef = -amplitude*(2*pi*f1)^2*sin(2*pi*f1*time);
    else
      xRef = amplitude*(0.65*sin(2*pi*f1*time) + 0.35*sin(2*pi*f2*time));
      vRef = amplitude*(0.65*(2*pi*f1)*cos(2*pi*f1*time)
        + 0.35*(2*pi*f2)*cos(2*pi*f2*time));
      aRef = -amplitude*(0.65*(2*pi*f1)^2*sin(2*pi*f1*time)
        + 0.35*(2*pi*f2)^2*sin(2*pi*f2*time));
    end if;
  end TrajectoryGenerator;

  block BaselineController
    parameter RigParameters p = RigParameters();
    parameter Real Kp(unit="N/m") = 220;
    parameter Real Kd(unit="N.s/m") = 18;
    Modelica.Blocks.Interfaces.RealInput x(unit="m");
    Modelica.Blocks.Interfaces.RealInput v(unit="m/s");
    Modelica.Blocks.Interfaces.RealInput xRef(unit="m");
    Modelica.Blocks.Interfaces.RealInput vRef(unit="m/s");
    Modelica.Blocks.Interfaces.RealInput aRef(unit="m/s2");
    Modelica.Blocks.Interfaces.RealOutput tauCmd(unit="N.m");
  protected
    SI.Force Fcmd;
  equation
    Fcmd = Kp*(xRef-x) + Kd*(vRef-v) + (p.carriageMass+p.resonatorMass)*aRef;
    tauCmd = p.pulleyRadius*Fcmd;
  end BaselineController;

  block ActiveDampingController
    parameter RigParameters p = RigParameters();
    parameter Real Kp(unit="N/m") = 220;
    parameter Real Kd(unit="N.s/m") = 18;
    parameter Real Ktheta(unit="N/rad") = 8.0;
    parameter Real Komega(unit="N.s/rad") = 1.1;
    Modelica.Blocks.Interfaces.RealInput x(unit="m");
    Modelica.Blocks.Interfaces.RealInput v(unit="m/s");
    Modelica.Blocks.Interfaces.RealInput theta(unit="rad");
    Modelica.Blocks.Interfaces.RealInput thetaDot(unit="rad/s");
    Modelica.Blocks.Interfaces.RealInput xRef(unit="m");
    Modelica.Blocks.Interfaces.RealInput vRef(unit="m/s");
    Modelica.Blocks.Interfaces.RealInput aRef(unit="m/s2");
    Modelica.Blocks.Interfaces.RealOutput tauCmd(unit="N.m");
  protected
    SI.Force Fcmd;
  equation
    Fcmd = Kp*(xRef-x) + Kd*(vRef-v)
      + (p.carriageMass+p.resonatorMass)*aRef
      - Ktheta*theta - Komega*thetaDot;
    tauCmd = p.pulleyRadius*Fcmd;
  end ActiveDampingController;

  model CompleteRig
    parameter RigParameters p = RigParameters();
    parameter MotionMode motionMode = MotionMode.Aggressive;
    parameter ControllerMode controllerMode = ControllerMode.Baseline;
    parameter SI.Position motionAmplitude = 0.035;
    parameter SI.Angle initialTheta = 0;

    RigPlant plant(p=p, theta_start=initialTheta);
    TrajectoryGenerator trajectory(mode=motionMode, amplitude=motionAmplitude);
    BaselineController baseline(p=p);
    ActiveDampingController active(p=p);
    AS5600 motorEncoder;
    AS5600 leverEncoder;
    LeverIMU imu(radius=p.imuRadius, g=p.g);

    Modelica.Blocks.Interfaces.RealOutput x(unit="m") = plant.x;
    Modelica.Blocks.Interfaces.RealOutput theta(unit="rad") = plant.theta;
    Modelica.Blocks.Interfaces.RealOutput motorEncoderRaw(unit="rad") = motorEncoder.raw;
    Modelica.Blocks.Interfaces.RealOutput leverEncoderCentered(unit="rad") = leverEncoder.centered;
    Modelica.Blocks.Interfaces.RealOutput gyro(unit="rad/s") = imu.gyro;
    Modelica.Blocks.Interfaces.RealOutput accelTangential(unit="m/s2") = imu.accelTangential;
    Modelica.Blocks.Interfaces.RealOutput accelLongitudinal(unit="m/s2") = imu.accelLongitudinal;
  equation
    connect(trajectory.xRef, baseline.xRef);
    connect(trajectory.vRef, baseline.vRef);
    connect(trajectory.aRef, baseline.aRef);
    connect(plant.x, baseline.x);
    connect(plant.v, baseline.v);

    connect(trajectory.xRef, active.xRef);
    connect(trajectory.vRef, active.vRef);
    connect(trajectory.aRef, active.aRef);
    connect(plant.x, active.x);
    connect(plant.v, active.v);
    connect(plant.theta, active.theta);
    connect(plant.thetaDot, active.thetaDot);

    plant.tauCmd = if controllerMode == ControllerMode.Baseline then baseline.tauCmd else active.tauCmd;
    plant.tauExt = 0;

    connect(plant.phiMotor, motorEncoder.angle);
    connect(plant.theta, leverEncoder.angle);
    connect(plant.theta, imu.theta);
    connect(plant.thetaDot, imu.thetaDot);
    connect(plant.thetaDDot, imu.thetaDDot);
    connect(plant.xDDot, imu.xDDot);

    annotation(experiment(StartTime=0, StopTime=8, Tolerance=1e-7, Interval=0.001));
  end CompleteRig;

  package Examples
    extends Modelica.Icons.ExamplesPackage;

    model BaselineAggressive
      extends ActiveVibrationRig.CompleteRig(
        controllerMode=ActiveVibrationRig.ControllerMode.Baseline,
        motionMode=ActiveVibrationRig.MotionMode.Aggressive,
        motionAmplitude=0.035);
    end BaselineAggressive;

    model ActiveDampingAggressive
      extends ActiveVibrationRig.CompleteRig(
        controllerMode=ActiveVibrationRig.ControllerMode.ActiveDamping,
        motionMode=ActiveVibrationRig.MotionMode.Aggressive,
        motionAmplitude=0.035);
    end ActiveDampingAggressive;

    model FreeDecay
      extends ActiveVibrationRig.CompleteRig(
        controllerMode=ActiveVibrationRig.ControllerMode.Baseline,
        motionMode=ActiveVibrationRig.MotionMode.Hold,
        initialTheta=0.16);
    end FreeDecay;

    model EquivalentTorsionBaseline
      parameter ActiveVibrationRig.RigParameters pEq(useGeometricSprings=false);
      extends ActiveVibrationRig.CompleteRig(
        p=pEq,
        controllerMode=ActiveVibrationRig.ControllerMode.Baseline,
        motionMode=ActiveVibrationRig.MotionMode.Aggressive);
    end EquivalentTorsionBaseline;
  end Examples;

  annotation(uses(Modelica(version="4.0.0")));
end ActiveVibrationRig;

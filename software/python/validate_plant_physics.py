#!/usr/bin/env python3
"""Report spring-law differences and RK4 timestep convergence for the SEDP plant."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math

import numpy as np

from active_vibration_rig_2d import PlantParams, RigPlant


def spring_model_comparison(params: PlantParams | None = None,
                            angles_deg=(-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)) -> dict:
    """Compare geometric and equivalent spring torque using the same plant parameters."""
    base = params or PlantParams()
    placeholder = PlantParams()
    placeholder_fields = ('spring_anchor_half_spacing', 'spring_anchor_y', 'spring_shaft_radius',
                          'spring_k', 'spring_d', 'spring_free_length')
    equivalent = RigPlant(replace(base, use_geometric_springs=False))
    geometric = RigPlant(replace(base, use_geometric_springs=True))
    rows = []
    for angle_deg in angles_deg:
        theta = math.radians(float(angle_deg))
        torque_geo = geometric.spring_torque(theta)
        torque_equiv = equivalent.spring_torque(theta)
        rows.append({
            'angle_deg': float(angle_deg),
            'geometric_torque_nm': torque_geo,
            'equivalent_torque_nm': torque_equiv,
            'difference_nm': torque_geo - torque_equiv,
        })
    return {
        'geometry_parameters': {
            'anchor_half_spacing_m': base.spring_anchor_half_spacing,
            'anchor_y_m': base.spring_anchor_y,
            'shaft_radius_m': base.spring_shaft_radius,
            'stiffness_n_per_m': base.spring_k,
            'damping_ns_per_m': base.spring_d,
            'free_length_m': base.spring_free_length,
            'values_are_openmodelica_placeholders': all(
                math.isclose(getattr(base, field), getattr(placeholder, field), rel_tol=0.0, abs_tol=1e-12)
                for field in placeholder_fields
            ),
        },
        'geometric_small_angle_stiffness_nm_per_rad': base.effective_small_angle_stiffness
        if base.use_geometric_springs else replace(base, use_geometric_springs=True).effective_small_angle_stiffness,
        'equivalent_small_angle_stiffness_nm_per_rad': base.effective_small_angle_stiffness
        if not base.use_geometric_springs else replace(base, use_geometric_springs=False).effective_small_angle_stiffness,
        'torque_comparison': rows,
    }


def _conservative_params(use_geometric_springs: bool) -> PlantParams:
    return replace(
        PlantParams(use_geometric_springs=use_geometric_springs),
        b_x=0.0, x_coulomb=0.0, motor_viscous=0.0,
        motor_coulomb=0.0, belt_damping=0.0, c_theta=0.0,
        spring_d=0.0, theta_coulomb=0.0,
    )


def timestep_convergence(dts=(0.004, 0.002, 0.001, 0.0005), reference_dt=0.000125,
                         duration=0.12, normalized_error_limit=1e-4,
                         energy_drift_limit=1e-4) -> dict:
    """Compare held-input RK4 trajectories against a fine-step reference.

    Error is the maximum across the interval of the RMS seven-state error
    normalized by characteristic scales [rad, rad/s, m, m/s, rad, rad/s, N m].
    The scenarios are frictionless free motion and the same initial condition
    with a constant 0.08 N m torque command. Energy drift is reported for free
    motion, where actuator input is zero.
    """
    dts = tuple(float(dt) for dt in dts)
    if not dts or reference_dt <= 0 or duration <= 0 or any(dt <= reference_dt for dt in dts):
        raise ValueError('Timesteps and duration must be positive and larger than the reference timestep')
    for dt in (*dts, reference_dt):
        if not math.isclose(duration / dt, round(duration / dt), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError('Duration must be an integer multiple of every timestep')
    initial = np.array([0.015, 0.08, 0.002, 0.025, 0.02, -0.04, 0.0], dtype=float)
    scales = np.array([0.05, 20.0, 0.01, 0.5, 0.1, 2.0, 0.48], dtype=float)
    models = {}
    state_names = ('motor_angle_rad', 'motor_speed_rad_s', 'carriage_x_m', 'carriage_speed_m_s',
                   'theta_rad', 'theta_speed_rad_s', 'actuator_torque_nm')
    for model_name, use_geometry in (('equivalent_torsion', False), ('two_spring_geometry', True)):
        params = _conservative_params(use_geometry)
        plant = RigPlant(params)
        scenarios = {}
        for scenario, command in (('free', 0.0), ('driven', 0.08)):
            ref_steps = int(round(duration / reference_dt))
            ref_history = [initial.copy()]
            y_ref = initial.copy()
            for _ in range(ref_steps):
                y_ref = plant.rk4(y_ref, command, reference_dt)
                ref_history.append(y_ref.copy())

            rows = []
            for dt in dts:
                stride = int(round(dt / reference_dt))
                if not math.isclose(dt / reference_dt, stride, rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError('Each candidate timestep must be an integer multiple of the reference timestep')
                y = initial.copy()
                max_normalized_error = 0.0
                max_energy_drift = 0.0
                initial_energy = plant.total_mechanical_energy(y)
                for index in range(int(round(duration / dt))):
                    y = plant.rk4(y, command, dt)
                    reference_state = ref_history[(index + 1) * stride]
                    normalized_error = float(np.sqrt(np.mean(((y - reference_state) / scales) ** 2)))
                    max_normalized_error = max(max_normalized_error, normalized_error)
                    if scenario == 'free':
                        relative_drift = abs(plant.total_mechanical_energy(y) - initial_energy) / max(abs(initial_energy), 1e-12)
                        max_energy_drift = max(max_energy_drift, float(relative_drift))
                rows.append({
                    'physics_dt_s': dt,
                    'max_normalized_state_error_rms': max_normalized_error,
                    'final_absolute_state_error': {
                        name: float(value) for name, value in zip(state_names, np.abs(y - y_ref))
                    },
                    'max_relative_energy_drift': max_energy_drift if scenario == 'free' else None,
                })
            scenarios[scenario] = {'steps': rows}

        supported = [dt for dt in dts if all(
            next(row for row in scenarios[name]['steps'] if row['physics_dt_s'] == dt)['max_normalized_state_error_rms'] <= normalized_error_limit
            and (name != 'free' or next(row for row in scenarios[name]['steps'] if row['physics_dt_s'] == dt)['max_relative_energy_drift'] <= energy_drift_limit)
            for name in scenarios
        )]
        models[model_name] = {
            'spring_mode': 'explicit_two_spring_geometry' if use_geometry else 'equivalent_torsion',
            'plant_parameters': asdict(params),
            'parameter_overrides': {
                'b_x': 0.0, 'x_coulomb': 0.0, 'motor_viscous': 0.0,
                'motor_coulomb': 0.0, 'belt_damping': 0.0, 'c_theta': 0.0,
                'spring_d': 0.0, 'theta_coulomb': 0.0,
            },
            'initial_state': {name: float(value) for name, value in zip(state_names, initial)},
            'inputs': {'free_actuator_torque_nm': 0.0, 'driven_actuator_torque_nm': 0.08},
            'validated_max_physics_dt_s': max(supported) if supported else None,
            'scenarios': scenarios,
        }
    return {
        'duration_s': float(duration),
        'reference_dt_s': float(reference_dt),
        'normalized_state_scales': {
            'motor_angle_rad': scales[0], 'motor_speed_rad_s': scales[1],
            'carriage_x_m': scales[2], 'carriage_speed_m_s': scales[3],
            'theta_rad': scales[4], 'theta_speed_rad_s': scales[5],
            'actuator_torque_nm': scales[6],
        },
        'acceptance_limits': {
            'max_normalized_state_error_rms': float(normalized_error_limit),
            'max_relative_energy_drift_free_case': float(energy_drift_limit),
        },
        'models': models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=0.12)
    parser.add_argument('--dts', type=float, nargs='+', default=[0.004, 0.002, 0.001, 0.0005])
    parser.add_argument('--reference-dt', type=float, default=0.000125)
    parser.add_argument('--out', type=str, help='Optional JSON output path')
    args = parser.parse_args()
    report = {
        'spring_model_comparison': spring_model_comparison(),
        'timestep_convergence': timestep_convergence(args.dts, args.reference_dt, args.duration),
    }
    encoded = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as output:
            output.write(encoded + '\n')
    else:
        print(encoded)


if __name__ == '__main__':
    main()

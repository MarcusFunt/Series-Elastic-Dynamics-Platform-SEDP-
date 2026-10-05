import sys
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class SpringGeometryTests(unittest.TestCase):
    @staticmethod
    def reference_geometry(p, theta, theta_dot):
        """Independent generalized-force calculation for the documented geometry."""
        angle = p.theta_neutral_world + theta
        shaft = np.array([p.spring_shaft_radius * np.sin(angle),
                          p.spring_shaft_radius * np.cos(angle)])
        velocity = np.array([p.spring_shaft_radius * np.cos(angle) * theta_dot,
                             -p.spring_shaft_radius * np.sin(angle) * theta_dot])
        lengths = []
        rates = []
        forces = []
        torque = 0.0
        potential = 0.0
        for anchor_x in (-p.spring_anchor_half_spacing, p.spring_anchor_half_spacing):
            displacement = shaft - np.array([anchor_x, p.spring_anchor_y])
            length = np.linalg.norm(displacement)
            rate = displacement @ velocity / length
            magnitude = p.spring_k * (length - p.spring_free_length) + p.spring_d * rate
            force = -magnitude * displacement / length
            lengths.append(length)
            rates.append(rate)
            forces.append(force)
            # theta grows from +y toward +x, so Q_theta = F dot d(position)/d(theta).
            torque += force @ np.array([p.spring_shaft_radius * np.cos(angle),
                                        -p.spring_shaft_radius * np.sin(angle)])
            potential += 0.5 * p.spring_k * (length - p.spring_free_length) ** 2
        return np.array(lengths), np.array(rates), np.array(forces), torque, potential

    def test_geometric_mode_matches_openmodelica_reference_equations(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        p = PlantParams(use_geometric_springs=True)
        plant = RigPlant(p)
        theta, theta_dot = 0.13, -0.42
        expected = self.reference_geometry(p, theta, theta_dot)
        actual = plant.spring_geometry(theta, theta_dot)

        np.testing.assert_allclose(actual['lengths'], expected[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual['length_rates'], expected[1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual['forces'], expected[2], rtol=0, atol=1e-12)
        self.assertAlmostEqual(actual['torque'], expected[3], places=12)
        self.assertAlmostEqual(actual['potential'], expected[4], places=12)

    def test_symmetric_geometry_has_neutral_pose_and_odd_restoring_torque(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        plant = RigPlant(PlantParams(use_geometric_springs=True))
        neutral = plant.spring_geometry(0.0)
        self.assertAlmostEqual(neutral['torque'], 0.0, places=12)
        self.assertAlmostEqual(neutral['lengths'][0], neutral['lengths'][1], places=14)
        self.assertAlmostEqual(neutral['forces'][0][0], -neutral['forces'][1][0], places=12)
        self.assertAlmostEqual(neutral['forces'][0][1], neutral['forces'][1][1], places=12)
        downward = plant.spring_geometry(np.pi)
        self.assertAlmostEqual(downward['torque'], 0.0, places=12)
        for theta in (0.04, 0.16, 0.31):
            self.assertAlmostEqual(plant.spring_torque(-theta), -plant.spring_torque(theta), places=11)
            self.assertAlmostEqual(plant.spring_torque(np.pi - theta),
                                   -plant.spring_torque(np.pi + theta), places=11)

    def test_geometry_uses_neutral_world_angle_offset(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        p = PlantParams(use_geometric_springs=True, theta_neutral_world=0.23)
        actual = RigPlant(p).spring_geometry(0.11, -0.2)
        expected = self.reference_geometry(p, 0.11, -0.2)
        np.testing.assert_allclose(actual['lengths'], expected[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual['length_rates'], expected[1], rtol=0, atol=1e-14)
        self.assertAlmostEqual(actual['torque'], expected[3], places=12)
        self.assertGreater(abs(RigPlant(p).spring_torque(0.0)), 1e-4)

    def test_geometric_torque_is_negative_potential_gradient(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        for p in (PlantParams(use_geometric_springs=True),
                  PlantParams(use_geometric_springs=True, theta_neutral_world=0.23)):
            plant = RigPlant(p)
            for theta in (-0.24, -0.07, 0.07, 0.24):
                step = 1e-6
                gradient = (plant.spring_potential(theta + step) - plant.spring_potential(theta - step)) / (2 * step)
                self.assertAlmostEqual(plant.spring_torque(theta), -gradient, delta=2e-7)

    def test_equivalent_torsion_remains_the_default_and_retains_its_law(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        p = PlantParams()
        plant = RigPlant(p)
        self.assertFalse(p.use_geometric_springs)
        theta, theta_dot = 0.21, -0.32
        self.assertAlmostEqual(plant.spring_torque(theta, theta_dot),
                               -p.k_theta * theta - p.k_theta3 * theta**3 - p.c_theta * theta_dot,
                               places=14)


class RestoringStiffnessTests(unittest.TestCase):
    def test_reported_effective_stiffness_matches_numerical_total_torque_slope(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        for p in (PlantParams(), PlantParams(use_geometric_springs=True),
                  PlantParams(use_geometric_springs=True, theta_neutral_world=0.23)):
            plant = RigPlant(p)

            def total_torque(theta):
                gravity = p.resonator_mass * p.gravity * p.lever_com_distance * (
                    np.sin(p.theta_neutral_world + theta) - np.sin(p.theta_neutral_world)
                )
                return plant.spring_torque(theta) + gravity

            step = 1e-5
            measured = -(total_torque(step) - total_torque(-step)) / (2 * step)
            self.assertAlmostEqual(p.effective_small_angle_stiffness, measured, delta=2e-6)


class MechanicalEnergyTests(unittest.TestCase):
    def test_total_energy_matches_mass_matrix_and_enabled_potentials(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        p = PlantParams(use_geometric_springs=True)
        plant = RigPlant(p)
        y = np.array([0.12, 1.3, p.rail_half_travel + 0.004, -0.08, 0.14, 0.7, 0.0])
        _, delta, _ = plant.belt_force(y)
        alpha = p.theta_neutral_world + y[4]
        total_mass = p.carriage_mass + p.resonator_mass
        coupling = p.resonator_mass * p.lever_com_distance * np.cos(alpha)
        kinetic = (0.5 * p.motor_inertia * y[1] ** 2
                   + 0.5 * total_mass * y[3] ** 2
                   + coupling * y[3] * y[5]
                   + 0.5 * p.resonator_inertia_pivot * y[5] ** 2)
        belt_potential = 0.5 * p.belt_stiffness * delta**2 + 0.25 * p.belt_cubic * delta**4
        spring_potential = plant.spring_geometry(y[4])['potential']
        gravity_potential = p.resonator_mass * p.gravity * p.lever_com_distance * (
            np.cos(alpha) - np.cos(p.theta_neutral_world) + y[4] * np.sin(p.theta_neutral_world)
        )
        stop_excess = abs(y[2]) - p.rail_half_travel
        stop_potential = 0.5 * p.stop_stiffness * stop_excess**2
        expected = kinetic + belt_potential + spring_potential + gravity_potential + stop_potential
        self.assertAlmostEqual(plant.total_mechanical_energy(y), expected, places=12)

    def test_undamped_unforced_plant_conserves_total_mechanical_energy(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        p = replace(PlantParams(use_geometric_springs=True),
                    b_x=0.0, x_coulomb=0.0, motor_viscous=0.0, motor_coulomb=0.0,
                    belt_damping=0.0, c_theta=0.0, spring_d=0.0, theta_coulomb=0.0)
        plant = RigPlant(p)
        y = np.array([0.015, 0.08, 0.002, 0.025, 0.07, -0.12, 0.0])
        initial = plant.total_mechanical_energy(y)
        max_relative_drift = 0.0
        for _ in range(800):
            y = plant.rk4(y, 0.0, 0.00025)
            energy = plant.total_mechanical_energy(y)
            max_relative_drift = max(max_relative_drift, abs(energy - initial) / abs(initial))
        self.assertLess(max_relative_drift, 2e-5)

    def test_passive_damping_reduces_total_mechanical_energy(self):
        from active_vibration_rig_2d import PlantParams, RigPlant

        plant = RigPlant(PlantParams())
        y = np.zeros(7)
        y[4] = 0.08
        y[5] = -0.25
        initial = plant.total_mechanical_energy(y)
        for _ in range(1000):
            y = plant.rk4(y, 0.0, 0.0005)
        self.assertLess(plant.total_mechanical_energy(y), initial)


class PhysicsDiagnosticTests(unittest.TestCase):
    def test_spring_comparison_reports_torque_on_both_sides_of_neutral(self):
        from validate_plant_physics import spring_model_comparison

        report = spring_model_comparison()
        rows = report['torque_comparison']
        self.assertEqual([row['angle_deg'] for row in rows], [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
        self.assertAlmostEqual(rows[0]['geometric_torque_nm'], -rows[-1]['geometric_torque_nm'], places=12)
        self.assertTrue(np.isfinite(report['geometric_small_angle_stiffness_nm_per_rad']))

    def test_timestep_diagnostic_shows_convergence_for_free_and_driven_cases(self):
        from validate_plant_physics import timestep_convergence

        report = timestep_convergence(dts=(0.002, 0.001, 0.0005), reference_dt=0.000125,
                                      duration=0.12)
        self.assertEqual(set(report['models']), {'equivalent_torsion', 'two_spring_geometry'})
        for name, model in report['models'].items():
            self.assertEqual(model['spring_mode'], name if name == 'equivalent_torsion' else 'explicit_two_spring_geometry')
            self.assertEqual(model['plant_parameters']['use_geometric_springs'], name == 'two_spring_geometry')
            self.assertEqual(model['plant_parameters']['spring_d'], 0.0)
            self.assertEqual(model['parameter_overrides']['spring_d'], 0.0)
            self.assertEqual(model['initial_state']['theta_rad'], 0.02)
            self.assertEqual(model['inputs']['free_actuator_torque_nm'], 0.0)
            self.assertEqual(model['inputs']['driven_actuator_torque_nm'], 0.08)
            self.assertEqual(model['validated_max_physics_dt_s'], 0.001)
            self.assertEqual(set(model['scenarios']), {'free', 'driven'})
            for scenario in model['scenarios'].values():
                errors = [row['max_normalized_state_error_rms'] for row in scenario['steps']]
                self.assertGreater(errors[0], errors[1])
                self.assertGreater(errors[1], errors[2])
            self.assertIn('max_relative_energy_drift', model['scenarios']['free']['steps'][0])


if __name__ == '__main__':
    unittest.main()

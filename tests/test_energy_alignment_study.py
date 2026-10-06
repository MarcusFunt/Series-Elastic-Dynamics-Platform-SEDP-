import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'software/python'))


class DevelopmentSelectionTests(unittest.TestCase):
    def test_prefers_gate_compatible_weight_before_energy_ratio(self):
        from run_energy_alignment_study import select_weight

        observations={
            .25:[{'mean_energy_ratio':.98,'reasons':['tracking gate']}],
            1.0:[{'mean_energy_ratio':.99,'reasons':['less than 5% mean energy improvement over paired LQR']}],
            4.0:[{'mean_energy_ratio':.90,'reasons':['safety gate']}],
        }
        self.assertEqual(select_weight(observations),1.0)


if __name__=='__main__':
    unittest.main()

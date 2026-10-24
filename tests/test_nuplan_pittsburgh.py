"""Guard session leakage and dataset-size contracts before long training jobs."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'scripts/prepare_nuplan_pittsburgh.py'
spec = importlib.util.spec_from_file_location('pittsburgh_preparation', path)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PittsburghDatasetTests(unittest.TestCase):
    def test_fragments_of_recording_cannot_cross_splits(self):
        logs = [Path(f'2021.08.{day:02d}.12.00.00_veh-08_{start:05d}_00999.db')
                for day in range(1, 13) for start in (0, 500)]
        assignments, splits = prep.partition_logs(logs, 20261022)
        self.assertEqual(sum(map(len, splits.values())), len(logs))
        for left in splits:
            self.assertTrue(splits[left])
            for right in splits:
                if left != right:
                    self.assertFalse(set(assignments[left]) & set(assignments[right]))
        self.assertEqual((assignments, splits), prep.partition_logs(list(reversed(logs)), 20261022))

    def test_too_few_sessions_rejected(self):
        with self.assertRaises(ValueError):
            prep.partition_logs([Path('2021.08.01.12.00.00_veh-08_00000_00500.db')], 0)

    def test_cross_split_recording_leakage_rejected_even_with_distinct_tokens(self):
        def summary(folder):
            split = Path(folder).name
            data = {'map':prep.MAP, 'dataset':'nuplan', 'scenario_token':split,
                    'log_name':'2021.08.01.12.00.00_veh-08_00000_00500'}
            return {split:data}, [split], {split:''}
        with patch('metadrive.scenario.utils.read_dataset_summary', side_effect=summary):
            with self.assertRaisesRegex(ValueError, 'Leakage'):
                prep.audit_dataset(Path('/unused'), dict(train=1,val=1,test=1))

    def test_missing_scenes_rejected_instead_of_silently_reducing_budget(self):
        with patch('metadrive.scenario.utils.read_dataset_summary', return_value=({}, [], {})):
            with self.assertRaisesRegex(ValueError, 'expected 8000'):
                prep.audit_dataset(Path('/unused'), dict(train=8000,val=100,test=100))


if __name__ == '__main__':
    unittest.main()

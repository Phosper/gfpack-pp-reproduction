"""Raw nominal maxima must remain distinct from feasible selected scores."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduce.diagnose import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_invalid_nominal_winner_is_separate_from_feasible_winner(self):
        square=[[0.,0.],[2.,0.],[2.,2.],[0.,2.]]
        raw=[[[0.,0.,1.,0.],[0.,0.,1.,0.]],[[0.,0.,1.,0.],[3.,0.,1.,0.]]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root=Path(temporary)
            for name,value in [('input.json',{'sample_index':0,'contours':[square,square]}),
                               ('raw.json',raw),('metadata.json',{'height':2.,'sampling_profile':'paper'})]:
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            for index,pose in enumerate(raw):
                (root/f'candidate_{index:04d}.json').write_text(json.dumps({'candidate':index,
                    'actions':pose,'result':{'trace':[]}}),encoding='utf-8')
            result=diagnose(root)
            self.assertTrue(result['post_complete'])
            self.assertEqual(result['best_raw_nominal_candidate'],0)
            self.assertFalse(result['best_raw_nominal_metrics']['strict_valid'])
            self.assertEqual(result['selected_strict_valid_candidate'],1)
            self.assertEqual(result['selected_strict_valid_metrics']['sum_pair_overlap_area'],0.)
            self.assertIsNone(result['records'][0]['repair_rounds_increasing_overlap'])


if __name__=='__main__':unittest.main()

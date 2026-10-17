import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from model.cgan import Generator
from comparison import summarize, sample_checkpoint

class ComparisonChecks(unittest.TestCase):
    def test_batched_sampling_is_repeatable_balanced_and_checkpoint_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);torch.manual_seed(3)
            torch.save({'model_state':Generator().state_dict(),'metadata':{'z_dim':100}},root/'generator.pt')
            a=sample_checkpoint('cgan',root/'generator.pt',20,10,7,root/'a','cpu')
            b=sample_checkpoint('cgan',root/'generator.pt',20,10,7,root/'b','cpu')
            with np.load(root/'a/samples.npz') as x,np.load(root/'b/samples.npz') as y:
                np.testing.assert_array_equal(x['images'],y['images'])
                self.assertEqual(x['images'].shape,(20,1,28,28))
            self.assertEqual(a['class_counts'],[2]*10)
            self.assertEqual(a['checkpoint_sha256'],b['checkpoint_sha256'])

    def test_missing_pair_and_mismatched_classifier_seed_rejected(self):
        p={'models':{'cgan':{},'diffusion':{}},'seeds':[1,2], 'classifier':{'seed_offset':20,'epochs':2},'sampling':{'count':10},'limits':'fixture'}
        with self.assertRaises(ValueError):summarize([],p)
        rows=[dict(model=m,seed=s,classifier_seed=s+20,synthetic_rows=10,test_rows=10000,epochs=2,accuracy=.7+.01*s) for m in p['models'] for s in p['seeds']]
        result=summarize(rows,p)
        self.assertAlmostEqual(result['models']['cgan']['mean_accuracy'],.715)
        self.assertEqual(result['paired_differences'][0]['diffusion_minus_cgan'],0.)
        rows[0]['classifier_seed']=99
        with self.assertRaisesRegex(ValueError,'Classifier seed'):summarize(rows,p)

if __name__=='__main__':unittest.main()

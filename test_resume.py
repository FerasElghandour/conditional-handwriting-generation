"""Generated fixtures verify exact continuation, not MNIST model quality."""
import os
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from training.session import run_session

class ResumeContract(unittest.TestCase):
    def loader(self):
        generator = torch.Generator().manual_seed(301)
        images = torch.rand(16, 1, 28, 28, generator=generator)*2-1
        labels = torch.arange(16)%10
        return DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=True,
                          generator=torch.Generator().manual_seed(42), drop_last=True)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first: self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for x, y in zip(first, second): self.assert_nested_equal(x, y)
        else:
            self.assertEqual(first, second)

    def test_exact_continuation_for_both_architectures(self):
        device = os.environ.get('TRAIN_TEST_DEVICE', 'cpu')
        config = {'seed':42, 'batch_size':8, 'fixture':'generated16; no benchmark claim'}
        for kind in ('cgan', 'diffusion'):
            with self.subTest(model=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                run_session(kind, self.loader(), config, 2, root/'full', device=device)
                run_session(kind, self.loader(), config, 1, root/'first', device=device)
                run_session(kind, self.loader(), config, 2, root/'resumed', device=device, resume=root/'first/training-state.pt')
                full = torch.load(root/'full/training-state.pt', weights_only=True, map_location='cpu')
                resumed = torch.load(root/'resumed/training-state.pt', weights_only=True, map_location='cpu')
                for key in ('models', 'optimizers', 'rng', 'contract', 'completed_epochs'):
                    self.assert_nested_equal(full[key], resumed[key])
                for record in (*full['history'], *resumed['history']): record.pop('seconds')
                self.assertEqual(full['history'], resumed['history'])
                # A modified contract must fail before producing a new result directory.
                with self.assertRaisesRegex(ValueError, 'identical'):
                    run_session(kind, self.loader(), {**config, 'seed':43}, 3,
                                root/'invalid', device=device, resume=root/'first/training-state.pt')
                self.assertFalse((root/'invalid').exists())

if __name__ == '__main__':
    unittest.main()

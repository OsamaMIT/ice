import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")
import unittest
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import jax.numpy as jnp
from dataclasses import replace
from a2rl_drone_training.hierarchical.environment import corner_geometry_bonus, residual_training_config
from a2rl_drone_training.config import TrainingConfig, ResidualConfig
from a2rl_drone_training.trainer import PPOTrainer

class CornerTests(unittest.TestCase):
    def reward(self, point=(-.1,.1,0.), velocity=(.1,.9,0.), clearance=.1, clean=True):
        vector=lambda v:jnp.asarray([v],dtype=float)
        return float(corner_geometry_bonus(vector(point),vector(velocity),vector((-1.,0.,0.)),vector((0.,1.,0.)),vector((0.,0.,0.)),vector((0.,1.,0.)),jnp.array([clearance]),jnp.array([clean]))[0])
    def test_shorter_smoother_off_center_crossing_gets_bonus(self):
        self.assertGreater(self.reward(),0.)
        self.assertLessEqual(self.reward(),8.)
        self.assertEqual(self.reward(point=(0.,0.,0.)),0.)
        self.assertEqual(self.reward(point=(.1,-.1,0.)),0.)
    def test_wrong_exit_or_stationary_gets_no_bonus(self):
        self.assertEqual(self.reward(velocity=(0.,-1.,0.)),0.)
        self.assertEqual(self.reward(velocity=(0.,0.,0.)),0.)
    def test_clearance_failure_nonfinite_and_no_crossing(self):
        self.assertGreater(self.reward(clearance=.001),0.)
        self.assertEqual(self.reward(clearance=-.001),0.)
        self.assertEqual(self.reward(clean=False),0.)
        self.assertEqual(self.reward(point=(float('nan'),0.,0.)),0.)
    def test_requires_racing_mode(self):
        cfg=TrainingConfig(controller='residual_mpc',residual=ResidualConfig(corner_geometry_reward=True))
        with self.assertRaisesRegex(ValueError,'requires racing'):
            residual_training_config(cfg)
    def test_checkpoint_reward_mode_mismatch_rejected(self):
        obj=object.__new__(PPOTrainer)
        obj.config=TrainingConfig(controller='residual_mpc',residual=ResidualConfig(racing_objective=True,corner_geometry_reward=True))
        obj.env=SimpleNamespace(reference=SimpleNamespace(fingerprint='test'))
        old=replace(obj.config,residual=replace(obj.config.residual,corner_geometry_reward=False))
        payload=dict(config=old,actor_params={},critic_params={},actor_opt_state={},critic_opt_state={})
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.pkl';p.write_bytes(pickle.dumps(payload))
            with patch('a2rl_drone_training.trainer.validate_checkpoint_actions'):
                with self.assertRaisesRegex(ValueError,'fresh policy'):obj.load_checkpoint(p)

if __name__=='__main__':unittest.main()

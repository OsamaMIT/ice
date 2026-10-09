import unittest

import jax
import jax.numpy as jnp
import numpy as np

from a2rl_drone_training.config import NetworkConfig, ObservationConfig, PPOConfig
from a2rl_drone_training.networks import (
    critic_apply,
    init_critic,
    init_gc_actor,
    sample_action,
)
from a2rl_drone_training.optim import adam_init
from a2rl_drone_training.ppo import (
    Rollout,
    _loss,
    compute_gae,
    constrain_actor_exploration,
    exploration_std_ceiling,
    ppo_update,
)


class GAETests(unittest.TestCase):
    def test_true_terminal_does_not_bootstrap(self):
        advantages, returns = compute_gae(
            rewards=jnp.array([[1.0]]),
            values=jnp.array([[0.0]]),
            next_values=jnp.array([[10.0]]),
            terminated=jnp.array([[True]]),
            truncated=jnp.array([[False]]),
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0)
        self.assertAlmostEqual(float(returns[0, 0]), 1.0)

    def test_artificial_truncation_bootstraps_final_observation(self):
        advantages, _ = compute_gae(
            rewards=jnp.array([[1.0]]),
            values=jnp.array([[0.0]]),
            next_values=jnp.array([[10.0]]),
            terminated=jnp.array([[False]]),
            truncated=jnp.array([[True]]),
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 10.0)

    def test_truncation_ends_recursive_trace(self):
        advantages, _ = compute_gae(
            rewards=jnp.array([[1.0], [100.0]]),
            values=jnp.zeros((2, 1)),
            next_values=jnp.array([[2.0], [0.0]]),
            terminated=jnp.array([[False], [False]]),
            truncated=jnp.array([[True], [False]]),
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(advantages[0, 0]), 2.8, places=5)


class PPOUpdateTests(unittest.TestCase):
    def setUp(self):
        self.obs_config = ObservationConfig(gate_context=1)
        self.net_config = NetworkConfig(
            state_hidden=(8,),
            state_latent_dim=8,
            gate_hidden=(8,),
            gate_latent_dim=8,
            fusion_hidden=(8,),
            critic_hidden=(8,),
        )
        key, actor_key, critic_key = jax.random.split(jax.random.key(11), 3)
        self.key = key
        self.actor = init_gc_actor(actor_key, self.obs_config, self.net_config)
        self.critic = init_critic(critic_key, 37, self.net_config)

    def _rollout(self):
        time, envs = 4, 2
        key_obs, key_critic, key_action = jax.random.split(self.key, 3)
        obs = jax.random.normal(key_obs, (time, envs, self.obs_config.dim))
        critic_obs = jax.random.normal(key_critic, (time, envs, 37))
        actions = []
        log_probs = []
        for step in range(time):
            key_action, action_key = jax.random.split(key_action)
            action, log_prob = sample_action(
                self.actor, obs[step], action_key, self.obs_config
            )
            actions.append(action)
            log_probs.append(log_prob)
        values = jax.vmap(lambda x: critic_apply(self.critic, x))(critic_obs)
        next_values = jnp.concatenate([values[1:], values[-1:]], axis=0)
        terminated = jnp.zeros((time, envs), dtype=jnp.bool_)
        truncated = jnp.zeros_like(terminated)
        return Rollout(
            obs=obs,
            critic_obs=critic_obs,
            actions=jnp.stack(actions),
            log_probs=jnp.stack(log_probs),
            values=values,
            rewards=jnp.ones((time, envs), dtype=jnp.float32),
            next_values=next_values,
            terminated=terminated,
            truncated=truncated,
        )

    def test_asymmetric_ppo_update_is_finite_and_scheduled(self):
        rollout = self._rollout()
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.truncated,
            gamma=0.999,
            gae_lambda=0.99,
        )
        config = PPOConfig(horizon=4, minibatches=2, update_epochs=2)
        result = ppo_update(
            actor_params=self.actor,
            critic_params=self.critic,
            actor_opt_state=adam_init(self.actor),
            critic_opt_state=adam_init(self.critic),
            rollout=rollout,
            advantages=advantages,
            returns=returns,
            key=self.key,
            obs_config=self.obs_config,
            config=config,
            schedule_progress=0.5,
        )
        metrics = result[-1]
        self.assertTrue(all(np.isfinite(float(value)) for value in metrics.values()))
        self.assertAlmostEqual(float(metrics["actor_lr"]), 1.65e-4, places=8)
        self.assertAlmostEqual(float(metrics["critic_lr"]), 1.65e-4, places=8)
        self.assertGreaterEqual(float(metrics["mean_action_alignment_loss"]), 0.0)
        expected_ceiling = exploration_std_ceiling(config, 0.5)
        self.assertAlmostEqual(
            float(metrics["exploration_std_ceiling"]),
            expected_ceiling,
            places=6,
        )
        returned_actor = result[0]
        self.assertTrue(
            np.all(np.exp(np.asarray(returned_actor["log_std"])) <= expected_ceiling + 1e-6)
        )

    def test_exploration_ceiling_cools_and_resets_clipped_adam_momentum(self):
        config = PPOConfig(
            exploration_std_start=0.6,
            exploration_std_end=0.2,
            exploration_std_floor=0.08,
            exploration_decay_fraction=0.75,
        )
        ceilings = [exploration_std_ceiling(config, progress) for progress in (0.0, 0.5, 1.0)]
        self.assertGreater(ceilings[0], ceilings[1])
        self.assertGreater(ceilings[1], ceilings[2])
        self.assertAlmostEqual(ceilings[-1], 0.2)

        actor = {**self.actor, "log_std": jnp.ones((4,), dtype=jnp.float32)}
        opt_state = adam_init(actor)
        opt_state = {
            **opt_state,
            "m": {**opt_state["m"], "log_std": jnp.ones((4,), dtype=jnp.float32)},
            "v": {**opt_state["v"], "log_std": jnp.ones((4,), dtype=jnp.float32)},
        }
        constrained, constrained_state, fraction = constrain_actor_exploration(
            actor,
            opt_state,
            config,
            0.5,
        )
        np.testing.assert_allclose(
            np.exp(np.asarray(constrained["log_std"])),
            ceilings[1],
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(constrained_state["m"]["log_std"]),
            np.zeros((4,), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(constrained_state["v"]["log_std"]),
            np.zeros((4,), dtype=np.float32),
        )
        self.assertEqual(float(fraction), 1.0)

    def test_kl_target_stops_remaining_epochs(self):
        rollout = self._rollout()
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.truncated,
            gamma=0.999,
            gae_lambda=0.99,
        )
        config = PPOConfig(
            horizon=4,
            minibatches=2,
            update_epochs=6,
            actor_lr=1.0e-2,
            target_kl=1.0e-12,
        )
        metrics = ppo_update(
            actor_params=self.actor,
            critic_params=self.critic,
            actor_opt_state=adam_init(self.actor),
            critic_opt_state=adam_init(self.critic),
            rollout=rollout,
            advantages=advantages,
            returns=returns,
            key=self.key,
            obs_config=self.obs_config,
            config=config,
        )[-1]
        self.assertLess(float(metrics["epochs_completed"]), config.update_epochs)

    def test_privileged_critic_features_do_not_change_actor_gradient(self):
        rollout = self._rollout()
        batch = {
            "obs": rollout.obs.reshape((-1, self.obs_config.dim)),
            "critic_obs": rollout.critic_obs.reshape((-1, 37)),
            "actions": rollout.actions.reshape((-1, 4)),
            "old_log_probs": rollout.log_probs.reshape((-1,)),
            "old_values": rollout.values.reshape((-1,)),
            "advantages": jnp.ones((8,), dtype=jnp.float32),
            "returns": jnp.ones((8,), dtype=jnp.float32),
        }
        config = PPOConfig(horizon=4, minibatches=2, update_epochs=1)

        def actor_gradient(critic_observation):
            local_batch = {**batch, "critic_obs": critic_observation}
            return jax.grad(
                lambda params: _loss(
                    params,
                    self.critic,
                    local_batch,
                    jnp.asarray(config.entropy_coef),
                    self.obs_config,
                    config,
                )[0]
            )(self.actor)

        first = actor_gradient(batch["critic_obs"])
        second = actor_gradient(batch["critic_obs"] + 1000.0)
        for a, b in zip(
            jax.tree_util.tree_leaves(first),
            jax.tree_util.tree_leaves(second),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()

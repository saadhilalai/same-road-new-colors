"""
Recipe "beta" (PROGRESS.md, "Research"): the AI's action randomness is a Beta distribution instead of a Gaussian
(Chou et al. 2017; Petrazzini & Antonelo 2021 used it on CarRacing). A Beta lives on a fixed range, so no action
is ever cut off at the edges, which with a Gaussian made braking rare.

Also Slaoui et al.'s penalty for Agent B's fallback (config.SLAOUI_PENALTY): the AI acts on the normal image, and is
penalized when its internal view of a recolored copy of the same moment differs from its view of the original.
"""

import math

import numpy as np
import torch as th
import torch.nn.functional as F
from functools import partial
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.preprocessing import get_action_dim, preprocess_obs
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import explained_variance
from torch import nn


class BetaDistribution(Distribution):
    """
    One Beta(alpha, beta) per action number on [0, 1], stretched to the action range [-1, 1].
    alpha, beta = 1 + softplus(network output), so both are above 1 and each distribution has a single peak
    (the paper: "1 is added to a softplus layer"). The best action (deterministic) is the Beta's mean, as in the paper.
    """

    EPS = 1e-6   # keeps actions off the exact edges, where the log-probability can be -infinity

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, params):
        alpha, beta = (1.0 + F.softplus(params)).chunk(2, dim=-1)
        self.distribution = th.distributions.Beta(alpha, beta)
        return self

    def _to_unit(self, actions):
        return ((actions + 1.0) / 2.0).clamp(self.EPS, 1.0 - self.EPS)

    def log_prob(self, actions):
        # Stretching [0, 1] to [-1, 1] halves the density: -log(2) per action number.
        return (self.distribution.log_prob(self._to_unit(actions)) - math.log(2.0)).sum(dim=-1)

    def entropy(self):
        return (self.distribution.entropy() + math.log(2.0)).sum(dim=-1)

    def sample(self):
        return 2.0 * self.distribution.sample().clamp(self.EPS, 1.0 - self.EPS) - 1.0

    def mode(self):
        return 2.0 * self.distribution.mean - 1.0

    def actions_from_params(self, params, deterministic=False):
        self.proba_distribution(params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, params):
        actions = self.actions_from_params(params)
        return actions, self.log_prob(actions)


class BetaCnnPolicy(ActorCriticCnnPolicy):
    """SB3's image policy with the Beta distribution. Everything else (networks, initial weights) is SB3's."""

    def _build(self, lr_schedule):
        # Same as ActorCriticPolicy._build (SB3 2.9.0), which only knows SB3's own distributions.
        self._build_mlp_extractor()
        self.action_dist = BetaDistribution(get_action_dim(self.action_space))
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi))

    def copy_feature_gap(self, obs):
        """Slaoui's penalty term: mean squared difference between the features of the images and of their copies."""
        x = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return F.mse_loss(self.features_extractor(x), self.features_extractor.copy_features(x))


class NormalViewCNN(NatureCNN):
    """
    NatureCNN for images that carry a recolored copy (envs.RecoloredCopy). After frame stacking the channels are
    [normal t-1 | copy t-1 | normal t | copy t] (3 each). Acting uses only the normal images, in the same layout
    as a plain agent's input; copy_features() runs the same network on the copies.
    """

    def __init__(self, observation_space, features_dim=512, channels_per_image=3):
        n_channels, h, w = observation_space.shape
        k = channels_per_image
        normal = [f * 2 * k + i for f in range(n_channels // (2 * k)) for i in range(k)]
        super().__init__(spaces.Box(0, 255, (len(normal), h, w), dtype=np.uint8), features_dim)
        self.register_buffer("normal_idx", th.tensor(normal), persistent=False)
        self.register_buffer("copy_idx", th.tensor([i + k for i in normal]), persistent=False)

    def forward(self, observations):
        return super().forward(observations.index_select(1, self.normal_idx))

    def copy_features(self, observations):
        return super().forward(observations.index_select(1, self.copy_idx))


class SlaouiPPO(PPO):
    """
    PPO plus Slaoui et al.'s penalty (config.SLAOUI_PENALTY times BetaCnnPolicy.copy_feature_gap), added to every
    minibatch's loss. train() is SB3 2.9.0's PPO.train with only the lines marked "Slaoui" added; with
    copy_penalty=0 it trains exactly like PPO (tested in scripts/test_beta_recipe.py).
    """

    def __init__(self, *args, copy_penalty=0.0, **kwargs):
        self.copy_penalty = copy_penalty
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        copy_gaps = []   # Slaoui

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if self.copy_penalty:   # Slaoui
                    copy_gap = self.policy.copy_feature_gap(rollout_data.observations)
                    copy_gaps.append(copy_gap.item())
                    loss = loss + self.copy_penalty * copy_gap

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if copy_gaps:   # Slaoui
            self.logger.record("train/copy_feature_gap", np.mean(copy_gaps))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

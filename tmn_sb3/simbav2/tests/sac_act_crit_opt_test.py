import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch as th
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor


from simbav2.optimizer import UnitAdam
from simbav2.sb3_simbav2_policies import SACSimbaV2Critic, SACSimbaV2Actor


def test_sb3_simba_normalization():
    obs_dim = 16
    action_dim = 3
    observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    features_extractor = FlattenExtractor(observation_space)
    features_dim = obs_dim

    print("--- Testing SACSimbaV2Actor ---")
    actor = SACSimbaV2Actor(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[256, 256],
        features_extractor=features_extractor,
        features_dim=features_dim,
        num_blocks=2,
        hidden_features=64,
    )

    # Initialize UnitAdam for Actor
    actor_optimizer = UnitAdam(actor.parameters(), lr=0.01)

    # Simulate step
    obs = th.randn(1, obs_dim)
    mean, log_std, _ = actor.get_action_dist_params(obs)
    loss_actor = mean.sum() + log_std.sum()
    loss_actor.backward()
    actor_optimizer.step()

    verify_params(actor, "Actor")

    print("\n--- Testing SACSimbaV2Critic ---")
    critic = SACSimbaV2Critic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[256, 256],
        features_extractor=features_extractor,
        features_dim=features_dim,
        n_critics=2,
        num_blocks=2,
        hidden_features=64,
    )

    critic_optimizer = UnitAdam(critic.parameters(), lr=0.01)

    actions = th.randn(1, action_dim)
    q_values = critic(obs, actions)
    loss_critic = sum(q.sum() for q in q_values)
    loss_critic.backward()
    critic_optimizer.step()

    verify_params(critic, "Critic")


def verify_params(model: th.nn.Module, model_name: str):
    """Checks all parameters in the model for the hyper_dense flag and verifies norm."""
    found_any = False
    success = True

    for name, param in model.named_parameters():
        if getattr(param, "_hyper_dense", False):
            found_any = True
            norms = th.linalg.norm(param, ord=2, dim=1)

            # Check if all rows are approximately 1.0
            is_unit = th.allclose(norms, th.ones_like(norms), atol=1e-5)
            if not is_unit:
                print(f"  [FAILED] {model_name} param: {name} | Norm: {norms[0].item():.4f}")
                success = False
            else:
                print(f"  -> Parameter '{name}': {is_unit} (Norm: {norms[0].item():.4f}...)")
                pass

    if found_any and success:
        print(f"  [PASSED] All HyperDense layers in {model_name} are correctly unit-normalized.")
    elif not found_any:
        print(f"  [WARNING] No parameters with '_hyper_dense' found in {model_name}!")


if __name__ == "__main__":
    test_sb3_simba_normalization()

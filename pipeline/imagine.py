"""
imagine.py — Imagination rollouts pour l'actor-critic
======================================================
L'agent "reve" des trajectoires en deroulant le RSSM avec
les actions de l'actor, SANS interagir avec l'env reel.

C'est le coeur de Dreamer : on entraine actor et critic
uniquement sur des trajectoires imaginees.
"""

import torch
from typing import Tuple

from models.rssm import RSSM
from models.actor_critic import Actor, Critic


def imagine_trajectories(
    rssm: RSSM,
    actor: Actor,
    initial_h: torch.Tensor,
    initial_z: torch.Tensor,
    horizon: int = 15,
) -> dict:
    """Deroule des trajectoires imaginaires avec l'actor.

    Args:
        rssm: world model (gele pendant l'imagination)
        actor: politique qui choisit les actions
        initial_h: (B, deter_dim) — etats de depart (depuis observe_sequence)
        initial_z: (B, stoch_dim)
        horizon: nombre de steps a imaginer

    Returns:
        dict avec h_seq, z_seq, action_seq, reward_seq, continue_seq
        Chaque tensor a shape (B, horizon, ...)
    """
    h, z = initial_h, initial_z

    hs, zs, actions, rewards, continues = [], [], [], [], []

    for t in range(horizon):
        # Actor choisit une action
        action = actor.get_action(h, z, sample=True)

        # World model imagine le step suivant
        h, z = rssm.imagine_step(h, z, action)

        # Predictions
        preds = rssm.predict(h, z)

        hs.append(h)
        zs.append(z)
        actions.append(action)
        rewards.append(preds["reward"])
        continues.append(preds["continue"].sigmoid())

    return {
        "h": torch.stack(hs, dim=1),           # (B, H, deter_dim)
        "z": torch.stack(zs, dim=1),           # (B, H, stoch_dim)
        "action": torch.stack(actions, dim=1),   # (B, H)
        "reward": torch.stack(rewards, dim=1),   # (B, H)
        "continue": torch.stack(continues, dim=1),  # (B, H)
    }


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Calcule les lambda-returns (Dreamer-style).

    V_lambda_t = r_t + gamma * c_t * ((1 - lambda) * V_{t+1} + lambda * V_lambda_{t+1})

    Args:
        rewards: (B, H)
        values: (B, H) — predictions du critic
        continues: (B, H) — probabilite de continuation [0, 1]
        gamma: discount factor
        lambda_: GAE lambda

    Returns:
        returns: (B, H) — targets pour le critic
    """
    B, H = rewards.shape
    device = rewards.device

    # Bootstrap avec la derniere valeur
    returns = torch.zeros(B, H, device=device)
    last_value = values[:, -1]

    for t in reversed(range(H)):
        if t == H - 1:
            next_value = last_value
        else:
            next_value = (1 - lambda_) * values[:, t + 1] + lambda_ * returns[:, t + 1]

        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * next_value

    return returns

"""
collector.py — Collecte d'episodes Crafter avec politique random
=================================================================
Joue dans l'env Crafter, stocke les trajectoires en .npz.
Chaque episode = dict de arrays numpy alignés temporellement.
"""

import os
import numpy as np
import crafter


# Les 16 clés de l'inventaire Crafter (vitals + items)
INVENTORY_KEYS = [
    "health", "food", "drink", "energy",
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]

# Les vitals ont un range [0, 9], les items n'ont pas de max strict
# On normalise les vitals par 9.0, les items on les clip à 10 et /10
VITAL_KEYS = {"health", "food", "drink", "energy"}


def extract_state(info: dict) -> np.ndarray:
    """Extrait un vecteur d'etat normalise [0, 1] depuis le dict info Crafter."""
    inventory = info["inventory"]
    state = np.zeros(len(INVENTORY_KEYS), dtype=np.float32)
    for i, key in enumerate(INVENTORY_KEYS):
        val = float(inventory[key])
        if key in VITAL_KEYS:
            state[i] = val / 9.0
        else:
            state[i] = min(val, 10.0) / 10.0
    return state


class CrafterCollector:
    """Collecte des episodes Crafter avec une politique random."""

    def __init__(self, save_dir: str, num_episodes: int = 500, seed: int = 0):
        self.save_dir = save_dir
        self.num_episodes = num_episodes
        self.seed = seed

    def collect(self):
        """Collecte et sauvegarde les episodes."""
        os.makedirs(self.save_dir, exist_ok=True)

        for ep in range(self.num_episodes):
            ep_seed = self.seed + ep
            env = crafter.Env(seed=ep_seed)
            obs = env.reset()
            rng = np.random.RandomState(ep_seed)

            observations = [obs]
            states = []
            actions = []
            rewards = []
            dones = []

            done = False
            while not done:
                action = rng.randint(env.action_space.n)
                next_obs, reward, done, info = env.step(action)

                states.append(extract_state(info))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                observations.append(next_obs)

            # observations a len T+1, les autres T
            # On aligne : obs[t], state[t], action[t], reward[t], done[t], next_obs[t]
            episode = {
                "observations": np.stack(observations[:-1], axis=0),  # (T, 64, 64, 3)
                "next_observations": np.stack(observations[1:], axis=0),  # (T, 64, 64, 3)
                "states": np.stack(states, axis=0),  # (T, 16)
                "actions": np.array(actions, dtype=np.int64),  # (T,)
                "rewards": np.array(rewards, dtype=np.float32),  # (T,)
                "dones": np.array(dones, dtype=np.bool_),  # (T,)
            }

            path = os.path.join(self.save_dir, f"episode_{ep:04d}.npz")
            np.savez_compressed(path, **episode)

            if (ep + 1) % 50 == 0 or ep == 0:
                T = len(actions)
                total_reward = sum(rewards)
                print(f"  Episode {ep+1}/{self.num_episodes} — length: {T}, reward: {total_reward:.1f}")

        print(f"Collecte terminee: {self.num_episodes} episodes dans {self.save_dir}")


if __name__ == "__main__":
    collector = CrafterCollector(save_dir="data/crafter_episodes", num_episodes=10, seed=0)
    collector.collect()

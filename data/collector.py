"""
collector.py — Collecte d'episodes Crafter
============================================
Joue dans l'env Crafter, stocke les trajectoires en .npz.
Chaque episode = dict de arrays numpy alignés temporellement.

Deux modes :
  - CrafterCollector : politique random (exploration large)
  - AgentCollector   : politique entraînée (transitions de meilleure qualité)

Chaque collecte est loggée dans manifest.json pour tracer l'origine des données.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
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


def _next_episode_index(save_dir: str) -> int:
    """Find the next available episode index in the save directory."""
    if not os.path.exists(save_dir):
        return 0
    existing = [f for f in os.listdir(save_dir) if f.startswith("episode_") and f.endswith(".npz")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.replace("episode_", "").replace(".npz", "")))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def _log_manifest(save_dir: str, source: str, start_idx: int, end_idx: int, mean_reward: float = None):
    """Log une collecte dans manifest.json."""
    manifest_path = os.path.join(save_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"collections": []}

    entry = {
        "source": source,
        "episodes": f"{start_idx:04d}-{end_idx:04d}",
        "count": end_idx - start_idx + 1,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    if mean_reward is not None:
        entry["mean_reward"] = round(mean_reward, 2)

    manifest["collections"].append(entry)

    # Update summary
    summary = {}
    for c in manifest["collections"]:
        src = c["source"]
        summary[src] = summary.get(src, 0) + c["count"]
    manifest["summary"] = summary
    manifest["total"] = sum(summary.values())

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


class CrafterCollector:
    """Collecte des episodes Crafter avec une politique random."""

    def __init__(self, save_dir: str, num_episodes: int = 500, seed: int = 0):
        self.save_dir = save_dir
        self.num_episodes = num_episodes
        self.seed = seed

    def collect(self):
        """Collecte et sauvegarde les episodes."""
        os.makedirs(self.save_dir, exist_ok=True)
        start_idx = _next_episode_index(self.save_dir)

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

            episode = {
                "observations": np.stack(observations[:-1], axis=0),
                "next_observations": np.stack(observations[1:], axis=0),
                "states": np.stack(states, axis=0),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "dones": np.array(dones, dtype=np.bool_),
            }

            file_idx = start_idx + ep
            path = os.path.join(self.save_dir, f"episode_{file_idx:04d}.npz")
            np.savez_compressed(path, **episode)

            if (ep + 1) % 50 == 0 or ep == 0:
                T = len(actions)
                total_reward = sum(rewards)
                print(f"  Episode {ep+1}/{self.num_episodes} — length: {T}, reward: {total_reward:.1f}")

        end_idx = start_idx + self.num_episodes - 1
        print(f"Collecte terminee: {self.num_episodes} episodes dans {self.save_dir}")
        print(f"  Fichiers: episode_{start_idx:04d}.npz → episode_{end_idx:04d}.npz")
        _log_manifest(self.save_dir, "random", start_idx, end_idx)


class AgentCollector:
    """Collecte des episodes Crafter avec un agent entraine (workspace + RSSM + actor)."""

    def __init__(self, save_dir: str, num_episodes: int = 200, seed: int = 5000):
        self.save_dir = save_dir
        self.num_episodes = num_episodes
        self.seed = seed

    @torch.no_grad()
    def collect(self, workspace, rssm, actor, device):
        """Collecte en jouant avec l'agent entraine."""
        os.makedirs(self.save_dir, exist_ok=True)
        start_idx = _next_episode_index(self.save_dir)

        workspace.eval()
        rssm.eval()

        total_rewards = []

        for ep in range(self.num_episodes):
            ep_seed = self.seed + ep
            env = crafter.Env(seed=ep_seed)
            obs = env.reset()

            h, z = rssm.initial_state(1, device)
            prev_action = torch.zeros(1, dtype=torch.long, device=device)

            observations = [obs]
            states = []
            actions = []
            rewards = []
            dones = []
            info = None

            done = False
            while not done:
                obs_t = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                if info is None:
                    state_t = torch.zeros(1, 16, device=device)
                    state_t[0, :4] = 1.0
                else:
                    state_t = torch.from_numpy(extract_state(info)).unsqueeze(0).to(device)

                w_t = workspace.encode_to_fused({"vision": obs_t, "state": state_t})
                out = rssm.observe_step(h, z, prev_action, w_t)
                h, z = out["h"], out["z"]

                action = actor.get_action(h, z, sample=True)
                action_int = action.item()

                next_obs, reward, done, info = env.step(action_int)

                states.append(extract_state(info))
                actions.append(action_int)
                rewards.append(reward)
                dones.append(done)
                observations.append(next_obs)

                obs = next_obs
                prev_action = action

            episode = {
                "observations": np.stack(observations[:-1], axis=0),
                "next_observations": np.stack(observations[1:], axis=0),
                "states": np.stack(states, axis=0),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "dones": np.array(dones, dtype=np.bool_),
            }

            file_idx = start_idx + ep
            path = os.path.join(self.save_dir, f"episode_{file_idx:04d}.npz")
            np.savez_compressed(path, **episode)

            ep_reward = sum(rewards)
            total_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0 or ep == 0:
                T = len(actions)
                print(f"  Episode {ep+1}/{self.num_episodes} — length: {T}, reward: {ep_reward:.1f}")

        end_idx = start_idx + self.num_episodes - 1
        mean_r = np.mean(total_rewards)
        print(f"Agent collecte terminee: {self.num_episodes} episodes (mean reward: {mean_r:.2f})")
        print(f"  Fichiers: episode_{start_idx:04d}.npz → episode_{end_idx:04d}.npz")
        _log_manifest(self.save_dir, "agent_v4", start_idx, end_idx, mean_reward=mean_r)


if __name__ == "__main__":
    collector = CrafterCollector(save_dir="data/crafter_episodes", num_episodes=10, seed=0)
    collector.collect()

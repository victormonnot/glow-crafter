# GLoW Crafter

Global Workspace + Dreamer-style World Model on Crafter (Minecraft 2D).

Inspired by [Multimodal Dreaming](https://arxiv.org/pdf/2502.21142) (ruflab, ANITI).

## Architecture

```
Crafter Env
    │
    ├── Vision (64x64 RGB)  ──► VisionEncoder ──► latent ──► projection ──┐
    ├── State (16 scalars)  ──► StateEncoder  ──► latent ──► projection ──┤──► Global Workspace (fused)
    └── Action (17 discrete) ─► ActionEncoder ──► latent ──► projection ──┘         │
                                                                                     ▼
                                                                              RSSM World Model
                                                                              (h=deterministic, z=stochastic)
                                                                                     │
                                                                              ┌──────┴──────┐
                                                                              Actor       Critic
                                                                          (policy π)    (value V)
```

## Training Pipeline — 4 Phases

| Phase | Description | What trains | What's frozen |
|-------|-------------|-------------|---------------|
| **0** | Data collection (random policy) | — | — |
| **1** | Pretrain autoencoders | Encoders + Decoders | — |
| **2** | Workspace alignment (contrastive + translation + cycle) | Projections + Workspace | — |
| **3** | World Model (RSSM) | RSSM | GW (frozen) |
| **4** | Actor-Critic (imagination) | Actor + Critic | GW + RSSM (frozen) |

## Commands

### Training

```bash
# Full pipeline (all 4 phases)
python main.py

# Skip data collection (if episodes already collected)
python main.py --skip-collect

# Run specific phase(s)
python main.py --phase 1           # Phase 1 only
python main.py --phase 2           # Phase 2 (loads phase 1 checkpoint)
python main.py --phase 3           # Phase 3 (loads phase 2 checkpoint)
python main.py --phase 4           # Phase 4 (loads phase 3 checkpoint)
python main.py --phase 3 4         # Phases 3 and 4
python main.py --phase 2 3 4       # Phases 2, 3, and 4

# Use a different config file
python main.py --config config/my_config.yaml
```

### Evaluation

```bash
# Eval only (loads latest checkpoints, runs agent in Crafter)
python main.py --eval

# Eval with a specific checkpoint version
python main.py --eval --load-version phase4_agent_v2
```

### Checkpoints

Each phase saves a versioned checkpoint with embedded metrics.

```bash
# List all checkpoints with their metrics and dates
python main.py --list-checkpoints

# Output example:
#   phase1_gw_v1.pt        2026-03-22T14:30:00  recon_loss=0.0005
#   phase2_gw_v1.pt        2026-03-22T14:45:00  total=15.07  contrastive=10.28  R@1=5.7%
#   phase2_gw_v2.pt        2026-03-22T19:00:00  total=12.40  contrastive=9.80  R@1=18.3%
#   phase3_rssm_v1.pt      2026-03-22T15:20:00  total=1.42  kl=1.33  ws_recon=0.089
#   phase4_agent_v1.pt     2026-03-22T16:00:00  mean_reward=-0.90  achievements=0

# Load a specific version (instead of latest)
python main.py --phase 4 --load-version phase2_gw_v1
```

**Naming convention:**
- `phase{N}_{name}_v{X}.pt` — versioned checkpoint (never overwritten)
- `phase{N}_{name}_latest.pt` — always points to the latest version

### Inference Notebook

```bash
jupyter notebook inference.ipynb
```

Contains:
1. Load checkpoints & play trained agent (animated)
2. Compare trained vs random agent (10 episodes, bar charts)
3. Workspace visualization (reconstruction, cross-modal translation)
4. RSSM imagination ("dreaming" — real vs imagined trajectories)
5. Value landscape & action distribution heatmap

## Project Structure

```
glow-crafter/
├── main.py                  # Entry point, CLI, orchestration
├── config/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── collector.py         # CrafterCollector (random policy episodes)
│   ├── dataset.py           # CrafterTransitionDataset, CrafterSequenceDataset (lazy-loaded)
│   └── transforms.py        # vision/state/action transforms
├── models/
│   ├── encoders.py          # VisionEncoder, StateEncoder, ActionEncoder
│   ├── decoders.py          # VisionDecoder, StateDecoder, ActionDecoder
│   ├── projections.py       # DomainProjection, InverseProjection
│   ├── domain_module.py     # DomainModule (encoder + decoder + projections)
│   ├── workspace.py         # GlobalWorkspace (fusion, translation)
│   ├── rssm.py              # RSSM world model (observe, imagine, predict)
│   └── actor_critic.py      # Actor (policy), Critic (value)
├── losses/
│   ├── contrastive.py       # InfoNCE contrastive loss
│   ├── translation.py       # TranslationLoss, ActionTranslationLoss
│   ├── cycle.py             # CycleConsistencyLoss
│   └── world_model.py       # WorldModelLoss (KL + recon + reward + continue)
├── pipeline/
│   ├── train.py             # train_phase1/2/3/4, epoch functions
│   ├── eval.py              # evaluate (GW metrics), evaluate_crafter_agent
│   └── imagine.py           # imagine_trajectories, compute_lambda_returns
├── checkpoints/             # Versioned checkpoints (auto-created)
├── inference.ipynb          # Visualization notebook
└── requirements.txt
```

## Key Hyperparameters (config.yaml)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `workspace_dim` | 128 | Shared workspace dimension |
| `align_epochs` | 100 | Phase 2 — was too low at 50 |
| `wm_epochs` | 75 | Phase 3 |
| `ac_epochs` | 100 | Phase 4 |
| `wm_max_batches` | 200 | Limits batches/epoch for speed |
| `ac_max_batches` | 200 | Same |
| `entropy_weight` | 1e-3 | Prevents policy collapse (was 3e-4) |
| `collect_episodes` | 200 | ~33K transitions |

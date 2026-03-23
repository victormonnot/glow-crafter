# GLoW Crafter — Journal de Developpement

Ce fichier trace toutes les etapes majeures du projet, les problemes rencontres, les solutions appliquees, et les resultats obtenus. Mis a jour a chaque changement significatif.

---

## Step 0 — Point de depart (copie de glow-minimal)

**Etat** : Le repo etait une copie de `glow-minimal` avec des domaines dummy (vision/text/audio sur des tenseurs random). Rien ne fonctionnait sur Crafter.

**Objectif** : Transformer ca en Global Workspace + World Model (Dreamer-style) sur Crafter.

---

## Step 1 — Environment Integration + Data Collection

**Ce qu'on a fait :**
- Cree `data/collector.py` : `CrafterCollector` joue avec une politique random dans Crafter
- Chaque episode sauvegarde en `.npz` : observations (64x64x3), state vector (16 scalars), actions, rewards, dones
- `extract_state(info)` : extrait les 16 valeurs du dict info Crafter (4 vitals normalises /9 + 12 items normalises /10)
- Cree `CrafterTransitionDataset` (transitions individuelles pour phases 1-2) et `CrafterSequenceDataset` (sequences pour phases 3-4) dans `data/dataset.py`
- Collecte de 200 episodes random (~40K transitions)

**Decisions :**
- Images en uint8 dans les .npz, converties en float32 [0,1] au chargement (economie disque)
- Sequences lazy-loaded pour eviter de tout charger en RAM

---

## Step 2 — Domain Modules Crafter + GW Training

**Ce qu'on a fait :**
- Remplace les encodeurs/decodeurs dummy par des vrais :
  - `VisionEncoder` : CNN 4 couches (64x64 -> latent 256)
  - `VisionDecoder` : 4 deconv layers (latent -> 64x64)
  - `StateEncoder` : MLP (16 scalars -> latent 128)
  - `StateDecoder` : MLP (latent -> 16 scalars)
  - `ActionEncoder` : Embedding(17) + MLP -> latent 64
  - `ActionDecoder` : MLP -> 17 logits
- 3 domaines dans le GW : vision, state, action
- Config adaptee : workspace_dim=128, domaine pairs vision-state, vision-action, state-action
- Phase 1 (pretrain autoencoders) + Phase 2 (alignement workspace) fonctionnels
- Pipeline d'eval avec R@K retrieval et translation metrics

---

## Step 3 — World Model (RSSM)

**Ce qu'on a fait :**
- Cree `models/rssm.py` : RSSM (Recurrent State-Space Model) style Dreamer
  - State = (h, z) : h = GRU deterministic (256d), z = stochastic latent (64d)
  - Prior p(z|h), Posterior q(z|h,w), predictions (workspace, reward, continue)
- Cree `losses/world_model.py` : KL(post||prior) + workspace recon + reward + continue
  - KL balance (Dreamer v2 trick) + free nats
- Le RSSM opere dans l'espace workspace (compact) — pas dans l'espace pixel
- Le GW est gele pendant cette phase

---

## Step 4 — Actor-Critic + Imagination

**Ce qu'on a fait :**
- Cree `models/actor_critic.py` : Actor (MLP -> Categorical sur 17 actions) + Critic (MLP -> valeur V)
- Cree `pipeline/imagine.py` : rollout RSSM avec actor pour H steps, lambda-returns
- Cree `pipeline/eval.py` : `evaluate_crafter_agent()` joue dans Crafter avec le policy appris
- Phase 4 entrainable

---

## Run 1 — Premier training complet

**Config** : 50 epochs par phase, batch_size=128, entropy_weight=3e-4
**Resultats** : Mean reward ~ -0.90, 0 achievements
**Probleme** : L'agent ne fait rien d'utile. Le GW alignment est faible.

---

## Changement — Augmentation des epochs + batch limiting

**Probleme** : Pas assez d'entrainement, les phases n'avaient pas converge.
**Solution** :
- Phase 2 : 50 -> 100 epochs
- Phase 3 : 50 -> 75 epochs
- Phase 4 : 50 -> 100 epochs
- Ajout de `wm_max_batches=200` et `ac_max_batches=200` pour limiter les batches par epoch (evite que les epochs soient trop longues avec beaucoup de data)
**Espoir** : Plus de training = meilleure convergence

---

## Changement — Systeme de checkpoints versiones

**Probleme** : Chaque retrain ecrasait les checkpoints precedents. Pas moyen de comparer les runs ou de revenir en arriere.
**Solution** :
- Checkpoints versiones : `phase2_gw_v1.pt`, `phase2_gw_v2.pt`, etc.
- Metriques embedees dans chaque checkpoint (losses, R@K, reward, etc.)
- `--list-checkpoints` pour voir tous les checkpoints avec leurs metriques
- `--load-version` pour charger une version specifique
- `_latest.pt` pointe toujours vers la derniere version

---

## Run 2 — Training avec plus d'epochs

**Resultats Phase 2** : R@1 vision->state = 6.6%, R@5 = 27%, translation MSE ~0.18
**Resultats Phase 4** : Mean reward = -0.75, entropy collapse (2.75 -> 0.23), actor_loss explose
**Probleme principal** : L'action comme domaine GW ne marche pas du tout. R@1 vision->action = 0%, translation accuracy = 5.9% (random = 1/17 = 5.9%). L'action est fondamentalement differente des observations — elle n'est pas semantiquement alignee avec la vision ou l'etat au meme timestep.

---

## Changement majeur — Suppression de l'action du GW

**Probleme** : L'action ne s'aligne pas avec vision/state dans le workspace. C'est normal — une action est une decision, pas une observation. Forcer l'alignement polluait le workspace.
**Analyse** : Dans le papier du lab "Multimodal Dreaming" et dans Dreamer, l'action n'est PAS un domaine du GW. Elle entre directement dans le RSSM et sort de l'Actor.
**Solution** :
- Retire l'action des domain modules du GW
- Retire les paires d'alignement contenant action
- domain_pairs = [["vision", "state"]] seulement
- L'action est geree uniquement par le RSSM (input) et l'Actor (output)
**Fichiers modifies** : config.yaml, main.py (build_workspace), pipeline/train.py, pipeline/eval.py

---

## Run 3 — Sans action domain, entropy_weight=1e-2

**Resultats Phase 2** : R@1 vision->state = 6.6% (pas de changement, normal vu qu'on a juste retire action)
**Resultats Phase 4** : Entropy collapse apres epoch 30 (2.5 -> 0.0001), actor_loss explose a -542
**Probleme** : Le poids d'entropy fixe ne s'adapte pas. Quand l'actor_loss change d'echelle, l'entropy_weight devient soit trop fort soit trop faible.

---

## Changement — 5 fixes pour Phase 4

**Probleme** : Entropy collapse + actor exploitation du world model
**Solutions appliquees (toutes ensemble) :**

1. **Auto-tune entropy (Dreamer v3 style)** : `log_alpha` learnable, optimise pour maintenir target_entropy=1.4. Alpha monte quand entropy baisse, descend quand entropy est haute. Plus besoin de fixer un poids manuellement.

2. **Symlog reward scaling** : `sign(r) * log(1 + |r|)` — amplifie les petits rewards (la plupart sont 0 ou 1), compresse les gros. Donne un signal de gradient plus uniforme.

3. **Gradient clipping** : actor_grad_clip=100, critic_grad_clip=100. Empeche les explosions de gradient quand l'actor trouve des "cheats" dans les trajectoires imaginees.

4. **Horizon reduit** : 15 -> 8 steps. Moins l'actor imagine loin, moins il peut exploiter les erreurs du world model.

5. **Early stopping** : Sauvegarde le meilleur modele (par imag_reward, seulement si entropy > 0.5). Restaure le best a la fin du training.

**Fichiers modifies** : pipeline/train.py (symlog, actor_critic_epoch, train_phase4), config.yaml

---

## Run 4 (actuel) — Avec les 5 fixes

**Resultats Phase 2** : R@1 vision->state = 6.6% (inchange)
**Resultats Phase 3** : KL et workspace_recon convergent normalement
**Resultats Phase 4** :
- Auto-tune entropy fonctionne : alpha s'adapte (0.91 -> 0.067 -> 2.27)
- Entropy reste stable jusqu'a epoch 82, puis collapse partiel, mais alpha remonte et compense
- Early stopping sauvegarde le best model a epoch 78 (imag_reward=0.0142)
- **Mean reward : 1.50** (+/- 0.86) — premier run au-dessus du random (~1.0)
- **Max reward : 3.10**
- **7 achievements** : wake_up (19/20), collect_sapling (14/20), collect_wood (9/20), collect_drink (3/20), place_table (3/20), place_plant (1/20), eat_cow (1/20)
- Checkpoint : `phase4_agent_v4.pt`

**Conclusion** : L'agent a un vrai comportement ! Il sait survivre, collecter des ressources, et crafter. Le bottleneck est maintenant le GW alignment (R@1 = 6.6%).

---

## Changement — Collecte de donnees elargie + boucle agent

**Probleme** : 200 episodes random = ~40K transitions. C'est peu. Plus de diversite dans les donnees devrait aider le GW et toutes les phases suivantes.
**Solution** :
- Ajout de `AgentCollector` dans `data/collector.py` : collecte des episodes avec l'agent entraine
- Commandes `--collect-random N` et `--collect-agent N` dans main.py
- Systeme de manifest (`manifest.json`) : trace l'origine de chaque collecte (random, agent_v4, ...), dates, nombre d'episodes, mean reward
- `--show-data` pour voir la composition du dataset
- Plan : collecter 600 random + 200 agent = 1000 episodes total, puis retrain

**Espoir** : Plus de data = meilleur GW alignment (plus de negatifs pour le contrastive) + meilleur world model (plus de variete de situations)

---

## Prochaines etapes (a venir)

- [ ] Collecter 600 episodes random + 200 episodes agent
- [ ] Retrain complet (Phases 1-4) sur 1000 episodes
- [ ] Evaluer si le R@1 monte avec plus de data
- [ ] Si non : ameliorer Phase 2 (temperature contrastive, reconstruction loss pendant alignement, projection plus profondes)
- [ ] Boucle : agent collecte -> retrain -> meilleur agent -> collecte -> ...

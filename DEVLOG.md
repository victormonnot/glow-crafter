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

## Ajout — Replay GIF (--play)

**Motivation** : On ne voyait que des chiffres et des images statiques dans le notebook. Pas de vrai gameplay visible.
**Solution** :
- `python main.py --play` : charge les checkpoints, joue un episode, sauvegarde un GIF (256x256, upscale depuis 64x64)
- `--seed 42` pour des replays reproductibles (meme seed = meme monde)
- `--seed 42 100 7` pour plusieurs GIFs d'un coup
- Nommage : `replays/phase4_v{N}_seed{S}.gif` — inclut la version du checkpoint pour comparer entre runs
- Meme seed + agents differents = comparaison visuelle directe de l'evolution

**Premier test** : phase4_v4_seed42.gif — 165 frames, reward 2.1, collect_wood + collect_sapling + wake_up

---

## Run 5 — Premiere boucle complete (1000 episodes : 800 random + 200 agent)

**Contexte** : Premiere iteration de la boucle collect -> train. 200 episodes existants + 600 random + 200 de l'agent v4.

**Resultats Phase 1** : recon_loss=0.0002 (tres bien, mieux qu'avant)
**Resultats Phase 2** :
- Contrastive loss : 1.14 (vs 1.20 avant) — mieux
- Translation MSE : 0.003 (vs 0.18 avant) — 60x mieux !
- R@1 vision->state : 3.5% — semble plus bas que le 6.6% d'avant MAIS le set d'eval est 5x plus grand (40K vs 8K samples), donc c'est en realite comparable ou mieux
**Resultats Phase 3** : ws_recon=0.050, KL=1.32 — convergence propre, comparable a avant
**Resultats Phase 4** :
- **Mean reward : 0.75** — en baisse vs 1.50 (run 4)
- **3 achievements** vs 7 avant
- Best epoch 55, entropy finale 0.84, alpha 0.002
- Le probleme : alpha est tombe trop bas (0.002), arretant de pousser l'exploration. L'entropy a collapse sans filet.

**Analyse** : Le GW et le world model sont meilleurs (translation MSE 60x plus bas), mais l'actor-critic a regresse. Le probleme est dans le controle de la policy, pas dans les representations. Alpha mourant = plus de regularisation entropy = policy qui s'effondre sur quelques actions.

**Checkpoint** : `phase4_agent_v5.pt`

---

## Changement — Fix controle de la policy (target_entropy + alpha clamp)

**Probleme** : Alpha tombe a 0.002 et meurt. L'entropy collapse suit inevitablement.
**Solutions** :

1. **Target entropy 1.4 -> 1.0** : 1.4 etait trop haut (max entropy pour 17 actions = ln(17) = 2.83). Ca forcait l'agent a rester trop uniforme au debut, puis alpha s'effondrait quand c'etait plus tenable. 1.0 est un compromis : assez d'exploration sans forcer la policy a etre random.

2. **Alpha clamp min=0.01** : Empeche alpha de descendre sous 0.01. Garantit un minimum permanent de regularisation entropy. Meme si l'optimizer veut le baisser plus, le clamp le retient.

**Fichiers modifies** : config.yaml (target_entropy, alpha_min), pipeline/train.py (alpha_min param + clamp apres optimizer step)

**Espoir** : Alpha reste au-dessus de 0.01, entropy ne collapse plus completement, l'agent garde un minimum d'exploration sur les 100 epochs.

---

## Run 6 — Phase 4 retrain avec target_entropy=1.0 + alpha_min=0.01

**Contexte** : Meme GW/RSSM que run 5 (1000 eps), seulement Phase 4 retrainee avec les fix entropy.
**Resultats** :
- **Mean reward : 0.65** — encore pire que run 5 (0.75)
- 4 achievements, best epoch 55
- Alpha a bien tenu a 0.01 (le clamp fonctionne)
- MAIS entropy est restee a ~2.0 pendant 90 epochs — l'agent est reste quasi-random

**Analyse** : Le diagnostic initial (world model dilue) etait faux. Le pic imag_reward a 0.022 (epoch 55) prouve que le world model SAT differencier les bonnes actions. Le vrai probleme : **le gradient reward est trop faible par rapport au bonus entropy**. Meme a alpha=0.01, le bonus entropy (0.01 * 2.0 = 0.02) domine l'imag_reward (~0.012). L'actor n'a pas assez de signal pour exploiter les bonnes trajectoires.

**Checkpoint** : `phase4_agent_v6.pt`

---

## Changement — Reward scaling + alpha_min plus bas

**Probleme** : Le signal reward est noye par le bonus entropy. L'actor voit des bonnes trajectoires mais ne les "lock" pas.
**Solutions** :

1. **reward_scale = 2.0** : Multiplie les returns par 2 avant le gradient actor. Le signal reward devient competitif face au bonus entropy (0.024 vs 0.02 au lieu de 0.012 vs 0.02). On commence conservateur a 2.0, on augmentera si besoin.

2. **alpha_min 0.01 -> 0.005** : Laisse alpha descendre plus bas. Reduit le bonus entropy minimum. Combined avec le reward scale, ca devrait permettre a l'actor d'exploiter.

**Fichiers modifies** : config.yaml (alpha_min, reward_scale), pipeline/train.py (reward_scale param, scaling des returns dans actor loss)

**Espoir** : L'entropy descend progressivement vers 1.0-1.5 (pas collapse, pas trop haute), l'actor apprend a exploiter les bonnes trajectoires.

---

## Run 7 — Phase 4 avec reward_scale=2.0 + alpha_min=0.005

**Contexte** : Meme GW/RSSM (1000 eps). Actor retrain avec reward_scale et alpha_min plus bas.
**Resultats** :
- **Mean reward : -0.10** — pire que random
- 3 achievements, best epoch 52 (imag_reward=0.0227)
- Entropy a bien baisse progressivement (2.8 -> 1.7 -> 0.9), pas de collapse brutal
- MAIS : l'actor a appris a "tricher" dans l'imagination. imag_reward haut (0.0227) mais reward reel nul (-0.10)
- Le world model imagine des trajectoires qui ne correspondent pas a la realite

**Analyse** : Le probleme n'est pas le controle de l'entropy. C'est que le RSSM (entraine sur 1000 episodes) predit des futurs trop lisses — les rewards imagines ne correspondent pas aux rewards reels. L'actor optimise une "fausse" reward. Horizon 8 aggrave le probleme (plus de steps = plus de drift). Les 3 tentatives sur 1000 episodes ont toutes echoue (0.75, 0.65, -0.10) alors que le run 4 (200 eps) donnait 1.50.

**Checkpoint** : `phase4_agent_v7.pt`

---

## Changement — Fix RSSM reward prediction + horizon plus court

**Probleme** : Le RSSM imagine des rewards qui ne correspondent pas a la realite. L'actor "triche" en exploitant les erreurs du world model.
**Solutions** :

1. **reward_weight 1.0 -> 3.0** (Phase 3) : Force le RSSM a mettre 3x plus d'importance sur la prediction de reward. Le world model differentiera mieux les etats avec/sans reward → trajectoires imaginees plus realistes.

2. **imagination_horizon 8 -> 6** (Phase 4) : Moins de steps imagines = predictions restent plus proches de la realite = moins de "triche" possible.

**Fichiers modifies** : config.yaml (reward_weight, imagination_horizon)
**Impact** : Retrain Phase 3 (RSSM) puis Phase 4 (actor-critic). Le GW (Phase 2) est inchange.

**Espoir** : imag_reward qui correle mieux avec le reward reel. L'actor apprend des vrais bons comportements, pas des exploits.

---

## Run 8 — Phase 3+4 avec reward_weight=3.0 + horizon=6

**Contexte** : RSSM retrain avec reward_weight x3, actor avec horizon 6.
**Resultats Phase 3** : reward loss 0.0025 (plus bas qu'avant a 0.0032). Convergence propre.
**Resultats Phase 4** :
- **Mean reward : 0.40** — toujours sous le random
- 4 achievements dont **defeat_zombie 5/20** (nouveau ! jamais vu avant)
- Best epoch : 6 — confirme que plus l'actor imagine, plus il diverge
- imag_reward plat (~0.01) pendant tout le training

**Analyse** : Le reward_weight x3 n'a pas suffi. Le probleme fondamental : ~99% des transitions ont reward=0, donc le RSSM apprend a predire ~0 partout (MSE basse mais aucune discrimination). C'est le **class imbalance problem** applique a la prediction de reward.

**Note** : defeat_zombie est interessant — l'agent a appris un comportement qu'on n'avait jamais vu, meme avec le meilleur run (1.50). Le horizon court (6) force peut-etre des strategies a court terme plus concretes.

**Checkpoint** : `phase4_agent_v8.pt`

---

## Etat actuel — Projet mis en pause

**Meilleur resultat** : Run 4 (200 episodes, phase4_v4) — mean_reward=1.50, 7 achievements
**Runs sur 1000 episodes** : 0.75, 0.65, -0.10, 0.40 — tous en regression

**Ce qui marche :**
- GW (vision + state) fonctionne, translation MSE tres bonne (0.003)
- RSSM converge bien (ws_recon, KL stables)
- Auto-tune entropy fonctionne (pas de collapse brutal)
- Pipeline complet fonctionnel (collect → train → eval → replay GIF)

**Ce qui ne marche pas encore :**
- Scaling a 1000 episodes degrade les performances de l'actor
- L'imag_reward ne correle pas bien avec le reward reel (actor "triche")
- Le RSSM predit reward ≈ 0 partout (class imbalance : 99% des transitions ont reward=0)

**Pistes pour quand on reprendra :**
- [ ] **Reward balancing dans le RSSM** : Ponderer les transitions avec reward != 0 beaucoup plus fort dans la loss (ex: x10-x50). Ou utiliser une loss focale / weighted BCE au lieu de MSE pour la reward prediction. C'est le fix le plus important — sans ca le RSSM ne differentie pas les bons etats.
- [ ] **Retour aux 200 episodes** comme baseline stable (run 4 = 1.50)
- [ ] **Ameliorer Phase 2** : temperature contrastive (0.07 → 0.5), reconstruction loss pendant alignement, projections plus profondes
- [ ] **Reward prediction categorielle** (comme Dreamer v3) : predire des bins de reward au lieu d'une valeur continue — resout le class imbalance naturellement
- [ ] **Boucle collect-train** : ne marche que si le RSSM est fiable. Fixer le RSSM d'abord.

**Priorite** : Passer a cocoBrain (priorite #1 pour le stage) et vmasHivemind.

# Evaluation & self-evolution harness

This directory holds the **evaluation-side** tooling for the `ninetoothed-operator-dev`
skill. None of it ships inside the skill at run time (the skill stays static, offline,
network-free); it exists to *measure* the skill and to *evolve* it against the proxy
task set. The one exception is `../scripts/strategy_selector.py`, a pure deterministic
function that reads the frozen `../references/rules.json` distilled here.

## Layout

```
evaluation/
  skill_eval/       robust_bench, reward_hacking_guard, failure_classifier   (pre-existing)
  proxy_tasks/      24-task proxy set (16 train / 8 holdout)                  (pre-existing)
  journal/          journal.py (append-only JSONL ledger) + curves.py         (L1)
  proxy_runner/     run_episode.py, rubric_scorer.py, reward.py, run_matrix.py (L0)
  evolve_moo/       version B: edit_ops, attribute, moo_loop                   (proposal route, self-evolving)
  evolve_full/      version A: strategies, bandit, memory, distill, loop       (full controller + memory)
  results/          baseline/, a-genN/, b-genN/, final/                        (all run artifacts)
```

## Two evolution routes (both write the same journal, share L0/L1)

* **Version B — `evolve_moo/moo_loop.py`.** The proposal/mid-term route made
  self-evolving: each generation evaluates the skill on train, attributes failures with
  the deterministic `failure_classifier`, and applies one `prune`/`substitute`/`reorder`
  edit to the skill text — promoted only if the pass-preservation guard sees no negative
  transfer. No bandit, no memory. Lowest risk; this alone satisfies the contest Stage 3.

* **Version A — `evolve_full/loop.py`.** The full controller: a contextual
  Thompson-sampling **bandit** picks a `(strategy, prompt-framing)` arm per episode;
  three-layer **memory** (episodic/semantic/procedural) accumulates; high-confidence
  **procedural rules** are distilled into `references/rules.json` + human-readable
  bullets each generation. The bandit's `prob_best` per context is logged so the report
  can show the RL criterion (action distribution shifts toward high-reward arms).

## Scoring

`rubric_scorer.py` implements the six-sub-score rubric from `../tests/verifier_spec.md`
(completion 4 / test 2 / perf 1 / minimality 1 / style 1 / compliance 1 = 10). It reuses
`../scripts/run_correctness_matrix.py` (MERE/MARE) for correctness and `skill_eval` for
the reward-hacking static check. The same scorer runs for `no_skill` / `v0` / `a` / `b`,
so deltas are meaningful.

`reward.py` computes the shaped scalar reward (latency-ratio + roofline − token/gpu/compile
cost) that drives the version-A bandit; it is distinct from the rubric total (the fitness).

## Running

Everything is offline-verifiable with a stub solver (`--fake`) — no GPU, no `claude`:

```bash
# offline dry-run of the whole pipeline (no GPU / no claude)
python evolve_moo/moo_loop.py --skill-root .. --journal results/b.jsonl --generations 3 --fake --no-git
python evolve_full/loop.py    --skill-root .. --journal results/a.jsonl --generations 2 --fake --no-git
```

On the **GPU host** (real `claude -p` episodes; needs torch + ninetoothed + CUDA):

```bash
# 0. baseline (learning-before): 24 tasks × {no_skill, v0}
python proxy_runner/run_matrix.py --skill-root .. --split all --modes no_skill v0 \
    --run-id baseline --journal results/journal.jsonl --csv results/baseline/matrix.csv

# 1. version B (train only), 5 generations, tag each gen
python evolve_moo/moo_loop.py --skill-root .. --journal results/journal.jsonl --generations 5

# 2. version A, 4 generations
python evolve_full/loop.py --skill-root .. --journal results/journal.jsonl --generations 4

# 3. learning-after: three-way comparison + curves
python journal/curves.py compare results/journal.jsonl --modes no_skill v0 a b
python journal/curves.py curve   results/journal.jsonl --run b --csv results/b_curve.csv
python journal/curves.py curve   results/journal.jsonl --run a --csv results/a_curve.csv
```

The episode harness is `claude -p --output-format json` (see `run_episode.py`), which is
the contest's fixed agent harness; token/cost per episode is parsed from its JSON and
recorded in the journal, so the whole learning process is auditable and reproducible.
```

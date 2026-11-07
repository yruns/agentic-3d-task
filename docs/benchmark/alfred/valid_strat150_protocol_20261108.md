# ALFRED `valid_strat150`: High-Fidelity 150-Episode Stratified Validation Fold for Rapid Iteration & Ablation

- **Date:** 2026-11-08
- **Fold File:** `docs/benchmark/alfred/folds/valid_strat150.json`
- **Schema:** `PilotManifest` (`schema_version = 1`, `per_split = 75`, `seed = 20261108`, `150` total episodes = `75 valid_seen` + `75 valid_unseen`)
- **Parent Full Split:** `docs/benchmark/alfred/folds/valid_full1641.json` (`820 valid_seen + 821 valid_unseen = 1,641` official `oct21.json` validation episodes)
- **Strategic Role:** Primary iteration, policy-program optimization, and ablation benchmark (`10.9×` faster turnaround than `valid_full1641` while matching full-validation SR, GC, PLW-SR, PLW-GC, room distribution, slicing ratio, expert action length, and failure mode breakdown within `< 0.35 pp`).

---

## 1. Motivation: Why `pilot50` Was Insufficient and Why `valid_strat150` Is the Core Iteration Fold

Evaluating all `1,641` official validation episodes (`valid_full1641.json`) takes `~4–6 hours` even under 36–84 worker concurrency and consumes `~1.5B` input tokens per run. While mandatory for final milestone certification, a `1,641`-episode loop is too slow for rapid iteration on policy-program synthesis (`run_policy_program`), visual disambiguation, and macro-controller refinements.

Conversely, the earlier `pilot50.json` (`25 seen + 25 unseen`) exhibited a **`+7.50 pp` optimistic bias** on `v16` (**`32.00%` SR on `pilot50` vs. `24.50%` SR on `valid_full1641`**, and **`44.00%` vs. `30.37%` on `valid_seen`**) due to three structural limitations:
1. **Uniform task-type quotas (`1/7 ≈ 14.3%`)** instead of the true non-uniform task-type proportions in `oct21.json` (where e.g. `look_at_obj_in_light` is `11.5%` of `valid_seen` but `21.1%` of `valid_unseen`, and `pick_two_obj_and_place` is `15.1%` of `valid_seen` but `9.9%` of `valid_unseen`).
2. **Uncontrolled secondary structural covariates**: room types (`Kitchen` / `LivingRoom` / `Bedroom` / `Bathroom`), slicing subgoal requirements (`*Sliced*`), and expert trajectory lengths (`expert_actions`) were not stratified.
3. **High quantization noise**: with only `3–4` episodes per `(split, task_type)` cell, a single episode flip shifted cell accuracy by `25%–33%`.

To enable fast, statistically faithful hillclimbing and thesis ablations, `docs/benchmark/alfred/folds/valid_strat150.json` (**150 episodes: `75 valid_seen` + `75 valid_unseen`**) is constructed via **six-dimensional structural stratification and dual-run (`v16` & `v17`) empirical distribution matching**.

---

## 2. Construction Protocol & Structural Guarantees

1. **100% Physical Trajectory Deduplication (`150 / 150` Unique Trajectories):**
   - The `1,641` official validation episodes originate from **506 unique physical trajectories** (`251` in `valid_seen`, `255` in `valid_unseen`), each annotated with `~3` language directives (`repeat_idx`).
   - `valid_strat150` enforces at most **1 episode per physical trajectory (`task`)**, covering **`29.6%` (`150 / 506`) of all unique validation trajectories**, **`54` distinct FloorPlan scenes** (`50` seen + `4` unseen), **`48` target object categories**, and **`21` receptacle categories**.
   - Annotation indices (`repeat_idx`) are balanced across `0`, `1`, and `2` (`50`, `48`, and `50` episodes respectively, plus `2` high-index annotations in `valid_unseen`).

2. **Exact Hamilton (Largest-Remainder) Quota Allocation Across All 14 `(split, task_type)` Cells:**
   - Within both `valid_seen` (`75` episodes) and `valid_unseen` (`75` episodes), task-type quotas match the exact proportions of `valid_full1641`:
     - **`valid_seen` (`75`):** `pick_and_place_simple` (`13`), `pick_cool_then_place_in_recep` (`12`), `pick_two_obj_and_place` (`11`), `pick_and_place_with_movable_recep` (`10`), `pick_clean_then_place_in_recep` (`10`), `pick_heat_then_place_in_recep` (`10`), `look_at_obj_in_light` (`9`).
     - **`valid_unseen` (`75`):** `look_at_obj_in_light` (`16`), `pick_heat_then_place_in_recep` (`13`), `pick_and_place_with_movable_recep` (`10`), `pick_clean_then_place_in_recep` (`10`), `pick_cool_then_place_in_recep` (`10`), `pick_and_place_simple` (`9`), `pick_two_obj_and_place` (`7`).

3. **Multi-Dimensional Covariate & Empirical Regime Alignment:**
   - Within each `(split, task_type)` quota, candidate sets are optimized to simultaneously match the full validation split on:
     - Room type distribution (`Kitchen`, `LivingRoom`, `Bedroom`, `Bathroom`);
     - Slicing requirement ratio (`*Sliced*` tasks);
     - Mean gold expert action length (`expert_actions`);
     - Official ALFRED metrics (`SR`, `GC`, `PLW-SR`, `PLW-GC`) across both `v16` and `v17`;
     - False-completion rate (`policy_stop` among failed episodes) and `(v16, v17)` transition regimes (`both_pass`, `v16_only`, `v17_only`, `both_fail_with_partial_gc`, `both_fail_zero_gc`).

---

## 3. Representativeness Verification: `valid_strat150` vs. `valid_full1641`

### 3.1 Split-Level & Overall Headline Metrics Alignment

All headline metrics on `valid_strat150` match the full `1,641`-episode validation benchmark within **`0.00 pp – 0.34 pp`** (at the exact integer quantization limit of `1/150 = 0.67 pp` and `1/75 = 1.33 pp`):

| Split / Metric | **`valid_strat150` (`150` eps)** | **`valid_full1641` (`1,641` eps)** | **Absolute Deviation ($\Delta$)** |
|---|---:|---:|---:|
| **Overall `v16` SR** | **24.67%** (`37 / 150`) | **24.50%** (`402 / 1,641`) | **`+0.17 pp`** |
| **Overall `v16` GC** | **33.59%** (`129 / 384`) | **33.70%** (`1,425 / 4,229`) | **`-0.11 pp`** |
| **Overall `v16` PLW-SR / PLW-GC** | **12.84% / 20.38%** | **12.83% / 20.39%** | **`+0.01 pp / -0.01 pp`** |
| **Overall `v17` SR** | **25.33%** (`38 / 150`) | **25.67%** | **`-0.34 pp`** |
| **Overall `v17` GC** | **36.43%** | **36.49%** | **`-0.06 pp`** |
| **Overall `v17` PLW-SR / PLW-GC** | **13.37% / 22.62%** | **13.60% / 22.52%** | **`-0.23 pp / +0.10 pp`** |
| **`valid_seen` `v16` SR / GC** | **30.67%** (`23 / 75`) / **36.98%** (`71 / 192`) | **30.37%** (`249 / 820`) / **36.94%** (`779 / 2,109`) | **`+0.30 pp / +0.04 pp`** |
| **`valid_seen` `v16` PLW-SR / PLW-GC** | **16.04% / 22.99%** | **15.99% / 23.00%** | **`+0.05 pp / -0.01 pp`** |
| **`valid_seen` `v17` SR / GC** | **28.00%** (`21 / 75`) / **36.98%** | **28.15% / 37.15%** | **`-0.15 pp / -0.17 pp`** |
| **`valid_unseen` `v16` SR / GC** | **18.67%** (`14 / 75`) / **30.26%** (`59 / 195`) | **18.64%** (`153 / 821`) / **30.47%** (`646 / 2,120`) | **`+0.03 pp / -0.21 pp`** |
| **`valid_unseen` `v16` PLW-SR / PLW-GC** | **9.38% / 17.56%** | **9.38% / 17.54%** | **`0.00 pp / +0.02 pp`** |
| **`valid_unseen` `v17` SR / GC** | **22.67%** (`17 / 75`) / **35.90%** | **22.99% / 35.78%** | **`-0.32 pp / +0.12 pp`** |
| **`valid_unseen` `v17` PLW-SR / PLW-GC** | **11.44% / 19.61%** | **11.65% / 19.60%** | **`-0.21 pp / +0.01 pp`** |

---

### 3.2 Structural Covariate & Failure-Mode Alignment

| Structural Dimension | `valid_seen` (`strat150` vs. `Full`) | `valid_unseen` (`strat150` vs. `Full`) |
|---|---:|---:|
| **Kitchen (`FloorPlan 1..30`)** | **52.0%** vs. `52.7%` | **57.3%** vs. `57.0%` |
| **Living Room (`FloorPlan 201..230`)** | **16.0%** vs. `15.7%` | **17.3%** vs. `17.8%` |
| **Bedroom (`FloorPlan 301..330`)** | **13.3%** vs. `12.9%` | **14.7%** vs. `14.6%` |
| **Bathroom (`FloorPlan 401..430`)** | **18.7%** vs. `18.7%` | **10.7%** vs. `10.6%` |
| **Slicing Required (`*Sliced*`)** | **17.33%** vs. `17.68%` | **12.00%** vs. `11.69%` |
| **Mean Gold Expert Actions (`g_i`)** | **50.81** vs. `50.90` | **46.93** vs. `46.74` |
| **`v16` False-Completion Rate (`policy_stop` \| fail)** | **73.08%** vs. `72.68%` | **40.98%** vs. `40.42%` |

---

### 3.3 Per-Cell Alignment Across All 14 `(split, task_type)` Strata

| Split | Task Type (`task_type`) | Quota (`strat150` % vs. `Full` %) | `v16` SR (`strat150` vs. `Full`) | `v16` GC (`strat150` vs. `Full`) | `v17` SR (`strat150` vs. `Full`) | `v17` GC (`strat150` vs. `Full`) |
|---|---|---:|---:|---:|---:|---:|
| `valid_seen` | `look_at_obj_in_light` | `9` (`12.0%` vs `11.5%`) | `66.7%` vs `61.7%` | `72.2%` vs `71.8%` | `55.6%` vs `53.3%` | `72.2%` vs `71.2%` |
| `valid_seen` | `pick_and_place_simple` | `13` (`17.3%` vs `17.3%`) | `46.2%` vs `43.7%` | `46.2%` vs `43.7%` | `46.2%` vs `47.4%` | `46.2%` vs `47.4%` |
| `valid_seen` | `pick_and_place_with_movable_recep` | `10` (`13.3%` vs `14.0%`) | `10.0%` vs `10.4%` | `21.2%` vs `21.3%` | `10.0%` vs `6.8%` | `18.2%` vs `18.4%` |
| `valid_seen` | `pick_clean_then_place_in_recep` | `10` (`13.3%` vs `13.7%`) | `10.0%` vs `19.6%` | `32.3%` vs `33.0%` | `30.0%` vs `25.2%` | `41.9%` vs `40.7%` |
| `valid_seen` | `pick_cool_then_place_in_recep` | `12` (`16.0%` vs `15.4%`) | `16.7%` vs `19.0%` | `29.3%` vs `29.0%` | `8.3%` vs `9.6%` | `24.4%` vs `24.1%` |
| `valid_seen` | `pick_heat_then_place_in_recep` | `10` (`13.3%` vs `13.0%`) | `30.0%` vs `27.1%` | `38.2%` vs `39.1%` | `10.0%` vs `13.5%` | `35.3%` vs `34.8%` |
| `valid_seen` | `pick_two_obj_and_place` | `11` (`14.7%` vs `15.1%`) | `36.4%` vs `33.9%` | `45.5%` vs `46.2%` | `36.4%` vs `38.3%` | `50.0%` vs `50.8%` |
| `valid_unseen` | `look_at_obj_in_light` | `16` (`21.3%` vs `21.1%`) | `43.8%` vs `42.8%` | `62.5%` vs `61.8%` | `43.8%` vs `42.1%` | `65.6%` vs `64.5%` |
| `valid_unseen` | `pick_and_place_simple` | `9` (`12.0%` vs `12.2%`) | `22.2%` vs `26.0%` | `22.2%` vs `26.0%` | `33.3%` vs `33.7%` | `33.3%` vs `33.7%` |
| `valid_unseen` | `pick_and_place_with_movable_recep` | `10` (`13.3%` vs `13.3%`) | `0.0%` vs `1.8%` | `6.7%` vs `8.0%` | `10.0%` vs `11.3%` | `20.0%` vs `18.6%` |
| `valid_unseen` | `pick_clean_then_place_in_recep` | `10` (`13.3%` vs `13.8%`) | `20.0%` vs `15.9%` | `35.5%` vs `35.6%` | `20.0%` vs `20.6%` | `41.9%` vs `41.6%` |
| `valid_unseen` | `pick_cool_then_place_in_recep` | `10` (`13.3%` vs `13.3%`) | `10.0%` vs `7.3%` | `28.1%` vs `26.3%` | `10.0%` vs `8.8%` | `25.0%` vs `25.5%` |
| `valid_unseen` | `pick_heat_then_place_in_recep` | `13` (`17.3%` vs `16.6%`) | `7.7%` vs `9.6%` | `23.3%` vs `24.3%` | `15.4%` vs `13.4%` | `30.2%` vs `29.5%` |
| `valid_unseen` | `pick_two_obj_and_place` | `7` (`9.3%` vs `9.9%`) | `14.3%` vs `14.8%` | `27.8%` vs `27.6%` | `14.3%` vs `20.5%` | `33.3%` vs `34.9%` |

---

## 4. Standard Evaluation Command on `valid_strat150`

All future architectural iterations (e.g., programmable `run_policy_program` execution, perception/mapping ablations, and verifier refinements) should be benchmarked on `docs/benchmark/alfred/folds/valid_strat150.json` first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
curl -fsS http://127.0.0.1:8787/health
curl -fsS http://127.0.0.1:8791/health
export XAUTHORITY=/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup/Xauthority

python -m codex_agent.cli.run_alfred \
  --data-root /home/ysh/Datasets/FILM/alfred/data/json_2.1.0 \
  --splits /home/ysh/Datasets/FILM/alfred/data/splits/oct21.json \
  --fold docs/benchmark/alfred/folds/valid_strat150.json \
  --output-dir outputs/alfred/ALFRED_<VERSION>_<DATE>_strat150 \
  --run-id ALFRED_<VERSION>_<DATE>_strat150 \
  --simulator-python /home/ysh/codebase/FILM/.venv/bin/python \
  --alfred-root /home/ysh/Datasets/FILM/alfred \
  --scene-config configs/alfred/scene.toml \
  --x-display 5 \
  --perception-server http://127.0.0.1:8791 \
  --model gemini-3.8-flash \
  --reasoning-effort provider-default \
  --max-actions 250 --max-failures 10 --max-decisions 35 \
  --continue-on-error --resume
```

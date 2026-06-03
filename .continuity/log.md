# Session log

## 2026-06-02 13:22

Onboarded Cascaide to Continuity: scaffolded .continuity/ (index, handoff, workboard, decisions, research-directions) and registered it in continuity.config.json (restarted server to load it). Documented both pipelines: image 2D-UNet DDPM (cascaide/) and sp_cas.py set/point DiT + count head. Confirmed 2 bugs: (1) TanhP99 inference broken (infer.py:244 build_encoder lacks dataset=); (2) energy_vector conditioner ImportError (EnergyVectorConditioner vs EnergyEmbedConditioner). Next: persist normalization params in checkpoints to fix TanhP99 inference. research-directions.md covers benchmarks/normalization/architectures/autoresearch.

## 2026-06-02 13:35

Installed ovito via mamba env "cascaide" (python 3.11, ovito 3.15, numpy 2.4) from conda.ovito.org — see runbook.md. Verified ovito reads vac+sia dumps; Frenkel conservation HOLDS (0/25), correcting a false-violation from a hand-rolled parser (SIA dumps have extra cols c_myKE c_myPE). Real-data facts: cascades are SMALL (~18/35/78 defects at 20/50/100 keV; max ~112) — synthetic is ~10x oversized, so H*W/2 cap is NOT binding for this corpus. Flipped research-directions to real-subset-first (cache 64-256 real cascades to .npz, zero-dep after). Added runbook.md. Next: torch not yet installed in env (pip install -e . to train).

## 2026-06-02 13:59

Built model-progress tooling (June 2 entry in journal.md): cascaide/eval/ (io/metrics/scorecard/registry/generators), RadialDensityAuxLoss (the missing radial loss, wired+config), viewer/dashboard.html (HPC-results-aware human viewer), scripts/{build_real_subset,run_benchmark}.py, tests/test_eval.py (6 pass), docs/benchmarks.md. Scorecard provably isolates the captures-shape-misses-substructure failure (blob passes count/radial, fails g(r)/cluster/NN). HPC results store: results/runs/<id>/{run.json,scorecard.json} -> registry.collect() -> index.json -> dashboard. Fixed core.py CosineSchedule import bug. Pre-existing test bug remains (test_diffusion_core t-range vs T=50). Tests need PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.

## 2026-06-02 14:46

Results now prominently viewable in Continuity: Cascaide human_entry=results/report.html (cover embeds it via iframe/mirror). Built report.py (self-contained, base64-inlined) + render.py (generated-vs-real cascade images per energy). Launched 300-epoch set-DiT training on 1200 real cascades (MPS); interim report at ~ep60 is live: 4 runs (3 baselines + real model). Set-DiT ~ep60 score 1.08 — radial_js 0.013 PASS (global shape good) but rdf/cluster/nn FAIL (substructure missed); visual confirms generated clouds are too compact/merged vs dispersed real cascades. Will run final benchmark when training hits 300 epochs.

## 2026-06-02 15:27

300-epoch set-DiT run done. RESULT: fully-trained sp_cas DiT is well-shaped (ep300 score 0.453, 6/7 pass; cluster_l1 0.049, nn_w1 0.60 both pass). Earlier "misses substructure" was UNDERTRAINING, not architecture — confirms researcher: sp_cas is current best, build on it. val-MSE selection is wrong: val-selected ep~50 scores 0.772 vs ep300 0.453 -> select by scorecard. Lone gap: rdf_l1 0.229 vs 0.20 (short-range pair corr), exactly what the substructure loss targets -> sharp hypothesis for Modal sweep. Modal launcher built (deploy/modal/, separate), NOT submitted. All runs viewable in Continuity cover (report). Items 1-4 + Modal built/tested.

## 2026-06-02 16:19

REAL Modal sweep LAUNCHED: 4 parallel T4 containers, full 3000 real cascades, 400 epochs, w_struct {0,0.3,0.6,1.0}. w_struct=0 = faithful sp_cas reproduction (control); question = does the substructure loss push rdf_l1 (the lone failing metric at ep300, 0.229 vs 0.20) under target without hurting the rest. Artifact collection FIXED + verified: each run dir is now self-contained (model.pt 28MB + scorecard + manifest + gen-vs-real images) and the results/runs/** downloader pulls weights back. Running ~12s/epoch -> ~1.5hr wall, est ~$3.5. On completion: auto-download + rebuild index/report -> dashboard/Continuity.

## 2026-06-02 17:54

Recipe changes shipped + EMA baseline submitted. (1) EMA added to train_set_v2 (on by default, decay 0.9999; used for select+sample+save). (2) scripts/rescore.py for fair 48/energy multi-seed scoring -> updates dashboard. (3) substructure loss demoted (default w_struct=0). (4) Min-SNR-gamma weighting WIRED but default OFF (ready, not confounding). Runs are now self-describing (label/config/tags include EMA/minSNR/w). Submitted 1 EMA baseline to Modal (w=0, EMA, 400ep, T4) = clean A/B vs the sweep no-EMA control. Plan when both land: rescore ALL run dirs at 48/energy x2 seeds for apples-to-apples vs the 0.453 baseline.

## 2026-06-02 18:05

Sweep landed. CONFIRMED: artifact fix works on real sweep (every run has model.pt 28MB) + Modal provenance tagged (scheduler=modal, job ids). BUG found+fixed: run_id collided (pre-EMA sweep used identical label "set-dit v2" for all 4 runs; 2 finished same UTC second -> same content-hash id -> w=0.3 overwritten; 3/4 dirs survived). Low science impact (loss-hurts trend already clear) but fixed registry.make_run_id to use a uuid4 suffix (robust under parallel same-second runs). Launched fair rescore of the 3 sweep runs at 48/energy (MPS). EMA baseline (w=0, EMA) still running on Modal. Next: fair numbers when rescore lands; full EMA-vs-control comparison when the EMA run lands.

## 2026-06-02 18:34

EMA BUG found+fixed. First EMA run gave scorecard 5.969 at ep100 because fixed decay=0.9999 with only ~9k steps left the shadow ~40% random-init (decay must match step count). Killed that Modal run (stopped app, no extra billing). Fix: EMAWarmup in train_set_v2 (effective decay = min(target,(1+step)/(10+step))) so shadow tracks model early. Smoke confirmed: warmup EMA = 1.18 at ep30 (vs 5.97 at ep100 broken). Also fixed run_id collision (uuid suffix). Resubmitted ONE longer EMA baseline (w=0, warmup EMA, 500ep, T4); monitoring first selection (ep100) to confirm not broken. Sweep rescore (3 runs @48/energy) still running on MPS.

## 2026-06-02 18:59

EMA fix CONFIRMED on Modal: warmup EMA run scores 0.492 at ep100 (vs 5.97 for the broken fixed-decay version; matches no-EMA control 12-sample 0.477). EMA now tracks the model -> not broken, no extra fix-loop needed. Letting the 500ep run finish (~80min), then will fair-rescore at 48/energy for the EMA-vs-control verdict. Reminder of fair sweep results: no-EMA control 0.472 (6/7); only failing metric now is vac_sia_sep (4.0 vs 3.0) since more data (3000) closed the g(r)/rdf gap (0.151, passes). Pair-relative encoding (built, unwired) is the data-justified next lever for vac-SIA.

## 2026-06-02 20:09

Wired pair-relative (Frenkel-pair) model for vac-SIA: cascaide/setdiff/paired.py (6-D PairedSetDenoiser DiT, PairNormalizer, dataset, sample_paired, save_paired) + setdiff/ema.py (shared warmup EMA) + generators.generate_paired_checkpoint + run_benchmark --generator paired-checkpoint + scripts/train_paired.py. Tested (22/22) + end-to-end smoke (self-contained run dir w/ model.pt). Caveat: no GT Frenkel labels -> Hungarian gives a min-cost matching; disp scale ~58A (vs vac ~140A), partial reduction, so vac-SIA improvement is empirical TBD. Added metrics: partial_rdf (vv/ss/vs), cross_nn_distances (Frenkel separation dist), radius_of_gyration, largest_cluster_frac, gyration_anisotropy. Captured top 3 as ADDITIVE scorecard diagnostics (vac_sia_nn_w1, rg_w1, largest_cluster_frac) - not weighted yet, so scores stay comparable. Verified discriminating (blob: vac_sia_nn_w1 2.8, lcf 0.37 vs real 0.23). EMA Modal run ~ep400, trending better than no-EMA control.

## 2026-06-02 20:14

Post-testing all models on the new metrics + surfacing in Continuity. report.py now renders a diagnostics table per run (vac->SIA NN dist W1, radius-of-gyration W1, largest-cluster fraction gen-vs-real). scripts/posttest.py re-evaluates every model on the full metric set and refreshes the report: drops stale diagnostic-less runs, re-runs baselines, re-benchmarks the local best/final checkpoints, re-scores the sweep model.pt runs in place (uniform n=32/energy for comparability). Running now (~22min MPS). The Continuity cover (results/report.html) will show the new per-model metrics on completion. EMA Modal run at ep420/500; will fold it in (re-run posttest) when it lands.

## 2026-06-02 20:52

DONE: post-tested all 8 models on the full metric set (incl. new diagnostics) and surfaced them on the Continuity cover (results/report.html, verified served via mirror). report.py renders a per-run diagnostics table (vac->SIA NN W1, Rg W1, largest-cluster frac). Final dashboard: copy 0.0 | jitter 0.434 | set-dit ep300 0.444 (7/7, best real model) | v2 control 0.518 | best-val 0.857 | v2 w=0.6 0.924 | v2 w=1.0 1.084 | blob 1.772. New vac_sia_nn_w1 confirms substructure loss worsens vac-SIA (3.5-4.0). EMA Modal run timed out (2h cap; fixed->5h) but EMA already verified working; user chose to STOP new runs and finalize. Paired model wired+tested (not trained at scale). scripts/posttest.py is idempotent to fold in future models.

## 2026-06-02 22:18

Added raw_data/ (gitignored, 254MB): 7100 minimized cascades across 9 CONTINUOUS energy ranges 0-300 keV (W, 1073K) — much richer than the old 3-point axis. Wrote scripts/process_raw_data.py (parallel pure-numpy parser, no ovito; robust to sia extra cols) -> data/cascaide_cascades.npz (17MB, loads via io.load_subset + extras: n_vac/n_sia/centroid/energy_range/supercell). 0 Frenkel violations, energy 0.01-299.8 keV, defects 0-1107. CRITICAL: supercell scales with energy (sc50->sc280) so absolute frames differ -> downstream MUST center per-cascade (global centroid is meaningless); centroid array stored. Big high-E cascades (max 1107 defects ~2200 tokens) -> O(N^2) attention heavy, needs train-time handling. .gitignore created (raw_data/, data/*.npz, results/, runs/, pycache, DS_Store). See runbook.md.

## 2026-06-02 22:34

Folded per-cascade centering into normalizers (cascaide/setdiff/data.py): PerCascadeNormalizer (center each cascade on its vac+SIA union centroid + global per-axis p99 scale; preserves vac-SIA relative geometry; serializes as G=0 CoordNormalizer so existing generate path works unchanged; isotropic s~86A) + SetDataset. Wired into train_set_v2 with --center/--energy_max/--energy_bin/--count_cap; scorecard.compute + generators._energy_counts now accept energy_bin for continuous energies. Tests 16/16. Smoke-verified full pipeline (train->select->save->finalize->results-store, per-cascade). LAUNCHED real run locally (MPS): <100keV subset of the new data (3290 cascades, continuous 0-100keV), 300 epochs, EMA, per-cascade centering, energy_bin=10 -> runs/setv2_lt100, writes to results/ dashboard for comparison vs old runs (ep300 0.444). ~2h. Ran local not Modal (avoid timeout/budget; comparable scale to old data).

## 2026-06-03 01:00

OVERNIGHT Modal run submitted (unattended): full 0-300 keV continuous data (7099 cascades after max_defects=1000 dropped 1), per-cascade centering, 250 epochs, EMA, energy_bin=25 (12 bins 0-300), A100. LIMIT: hard 3h Modal function timeout so it cannot run all night; sized to finish ~1.5h (ep1=20s/epoch, no OOM on big-cascade batches -> memory safe). On completion: auto-download + score + render + rebuild Continuity dashboard. Est cost ~$4-7 (A100 ~1.5h); budget ~$5.3 used. Risk: if it hits the 3h ceiling the run is lost (finalize wont download) but margin is large. This is the first training on the FULL continuous energy axis. modal_app reconfigured to a single bounded run + uploads cascaide_cascades.npz; added --max_defects safety cap to train_set_v2.

## 2026-06-03 01:06

Made the overnight run SLEEP-SAFE. Original modal run was NOT sleep-safe (ephemeral app tied to client + results downloaded locally -> sleep/disconnect would lose it). Stopped it (no orphan billing). Rebuilt modal_app: function writes results store to a persistent Modal Volume (cascaide-results) + commit(); launched with `modal run --detach ::main` so the cloud app survives laptop sleep/disconnect. Run healthy: full 0-300 keV (7099 cascades), per-cascade centering, 250ep, A100, ~22s/epoch (no OOM), hard 3h ceiling, ~<2h expected. Nothing depends on the laptop DURING the run. FETCH when back: `modal run deploy/modal/modal_app.py::fetch` -> pulls Volume to local results/ + rebuilds dashboard. Est cost ~$4-7.

## 2026-06-03 09:24

Built Polaris (ALCF) submit kit in deploy/polaris/: config.sh (edit-once: PROJECT/QUEUE/Eagle paths/Globus UUIDs), setup_env.sh (venv on Eagle over ALCF module load conda + pip install -e .), submit.sh (train=1 A100 / sweep=1 experiment per GPU across nodes, auto-sizes nodes), train.pbs, sweep.pbs+gpu_worker.sh (PALS rank->GPU), experiments.txt (design matrix), globus.sh (push npz->Eagle / pull results+ckpts->local), README.md (workflow + queue table). Workflow: code via git pull on Polaris, data/artifacts via Globus local<->Eagle, Polaris reads/writes /eagle directly. Added --no_report to train_set_v2 (parallel sweep ranks dont race on index.json; sweep rebuilds once). All bash -n clean + train_set_v2 compiles. NOT verifiable from here (module path/endpoint UUIDs/PROJECT need first-run confirmation on Polaris). NEXT: everything this session is uncommitted (eval/setdiff/deploy/docs/scripts/viewer/sp_cas + edits) -> must commit+push for the Polaris git pull.

## 2026-06-03 09:35

Pushed session work to FORK blaiszik/Cascaide branch tanhp99-global-centering (commit 8eabb005). origin (vigsam-coder/Cascaide) denies blaiszik write access -> use the fork; Polaris pulls from blaiszik/Cascaide. Polaris scripts updated to user conda pattern (module use /soft/modulefiles; module load conda; conda activate $ENV_PREFIX); setup_env clones ALCF base for CUDA torch. Git workflow going forward: commit to this branch + push to fork remote; git pull on Polaris. Remotes: origin=vigsam (upstream, RO), fork=blaiszik (push target).

"""n=24 head-to-head VERDICT: does the 500ep + --augment_rot run beat the prior best
per_cascade-250ep baseline? Scored identically to the centering A/B (full-range, n=24/energy,
dpmpp-20, --gen_energy sample, seed 0) so the numbers are directly comparable. Includes the two
best trace checkpoints (ep375/ep475) alongside final_model (ep500), since the n=12 trace suggested
an earlier checkpoint may edge out ep500."""
import numpy as np, torch, time
from cascaide.eval import io, scorecard as scm
from cascaide.eval.generators import generate_set_checkpoint

REPO = "/Users/blaiszik/Desktop/git/Cascaide"
CK = f"{REPO}/results/ckpts/percascade_500ep_aug"
models = [
    ("baseline per_cascade 250ep", f"{REPO}/results/runs/20260603T082541-set-dit-v2-87ec2c1b/model.pt"),
    ("500ep+aug ep375 (best trace)", f"{CK}/ckpt_0375.pt"),
    ("500ep+aug ep475",             f"{CK}/ckpt_0475.pt"),
    ("500ep+aug final (ep500)",     f"{CK}/final_model.pt"),
]
N_PER = 24
ref = io.load_subset(f"{REPO}/data/cascaide_cascades.npz")   # full range 0-300 keV


def reg(s, lo, hi): return [x for x in s if lo <= x["energy"] < hi]
def rdf_bulk(sc, ks):
    pe = sc["per_energy"]; v = [pe[k]["rdf"]["l1"] for k in ks if k in pe and "rdf" in pe[k]]
    return float(np.mean(v)) if v else float("nan")


print(f"full-range VERDICT | n={N_PER}/energy | dpmpp-20 | gen_energy=sample\n")
print(f"{'model':30} {'ep':>4} {'OVERALL':>8} {'p/t':>5} {'rdf':>6} {'rdf_hi':>7} {'low':>6} {'mid':>6} {'high':>6}")
out = {}
for label, d in models:
    ep = torch.load(d, map_location="cpu", weights_only=False).get("epoch")
    t0 = time.time()
    gen, _ = generate_set_checkpoint(d, ref, device="mps", n_per_energy=N_PER, energy_bin=25.0,
                                     sampler="dpmpp", steps=20, gen_energy="sample", seed=0)
    sc = scm.compute(gen, ref, label=label, energy_bin=25.0)
    regs = {}
    for nm, lo, hi in [("low", 0, 50), ("mid", 50, 150), ("high", 150, 1e9)]:
        r, g = reg(ref, lo, hi), reg(gen, lo, hi)
        regs[nm] = scm.compute(g, r, label=nm, energy_bin=25.0)["score"] if g and len(r) >= 8 else float("nan")
    rdf_all = rdf_bulk(sc, [str(k) for k in range(25, 301, 25)])
    rdf_hi = rdf_bulk(sc, ["175", "200", "225", "250", "275", "300"])
    out[label] = dict(overall=sc["score"], passed=sc["n_requirements_passed"],
                      total=sc["n_requirements"], rdf=rdf_all, rdf_hi=rdf_hi, **regs)
    print(f"{label:30} {str(ep):>4} {sc['score']:>8.3f} "
          f"{sc['n_requirements_passed']:>2}/{sc['n_requirements']:<2} {rdf_all:>6.3f} {rdf_hi:>7.3f} "
          f"{regs['low']:>6.3f} {regs['mid']:>6.3f} {regs['high']:>6.3f}  ({time.time()-t0:.0f}s)", flush=True)

base = out["baseline per_cascade 250ep"]
print("\nΔ vs baseline (negative = 500ep+aug BETTER):")
for label in [m[0] for m in models[1:]]:
    o = out[label]
    print(f"  {label:30} dOVERALL={o['overall']-base['overall']:+.3f}  drdf={o['rdf']-base['rdf']:+.3f}  "
          f"dlow={o['low']-base['low']:+.3f}  dmid={o['mid']-base['mid']:+.3f}  dhigh={o['high']-base['high']:+.3f}")

import json
json.dump(out, open("/tmp/verdict_results.json", "w"), indent=1)
print("\n-> /tmp/verdict_results.json")

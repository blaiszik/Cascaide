"""Build a single self-contained HTML report from the results store.

Unlike the interactive dashboard (which fetches JSON at runtime), this bakes everything —
requirement tables, gen-vs-real curve plots, and the generated-vs-real cascade images — into
ONE HTML file with images inlined as base64. That makes it trivially servable through
Continuity (no fetch, no relative-path issues) and shareable as a single artifact.

    from cascaide.eval.report import build_report
    build_report("results")            # -> results/report.html
"""
import os
import io as _io
import json
import glob
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_C, GEN_C = "#111827", "#2563eb"


def _b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _fig_to_b64(fig):
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _curve_png(sc):
    """Grid of metric curves (rows) × energies (cols): real solid vs generated dashed."""
    energies = [str(int(e)) for e in sc["energies_keV"]]
    rows = [("radial", "r", "radial pdf"), ("rdf", "r", "g(r)"),
            ("cluster_spectrum", "scales", "# clusters")]
    fig, axes = plt.subplots(len(rows), len(energies),
                             figsize=(3.2 * len(energies), 2.5 * len(rows)),
                             squeeze=False, facecolor="white")
    for ri, (key, xk, ylab) in enumerate(rows):
        for ci, e in enumerate(energies):
            ax = axes[ri][ci]
            pe = sc["per_energy"][e][key]
            ax.plot(pe[xk], pe["real"], color=REAL_C, lw=2, label="real")
            ax.plot(pe[xk], pe["gen"], color=GEN_C, lw=2, ls="--", label="generated")
            if ri == 0:
                ax.set_title(f"{e} keV", fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.tick_params(labelsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _req_table(sc):
    rows = ""
    for r in sc["requirements"]:
        c = "#16a34a" if r["pass"] else "#dc2626"
        rows += (f"<tr><td>{r['label']}</td>"
                 f"<td style='text-align:right'>{r['value']:.3f} {r['unit']}</td>"
                 f"<td style='text-align:right;color:#6b7280'>{r['target']}</td>"
                 f"<td style='text-align:center;color:{c};font-weight:700'>"
                 f"{'PASS' if r['pass'] else 'FAIL'}</td></tr>")
    return (f"<table class=req><tr><th>requirement</th><th>value</th>"
            f"<th>target</th><th></th></tr>{rows}</table>")


def _diag_table(sc):
    """Render the additive diagnostic metrics (averaged across energies), if present."""
    pes = [pe.get("diagnostics") for pe in sc.get("per_energy", {}).values()]
    pes = [d for d in pes if d]
    if not pes:
        return ""
    def avg(k):
        vals = [d[k] for d in pes if d.get(k) is not None and d.get(k) == d.get(k)]
        return sum(vals) / len(vals) if vals else float("nan")
    rows = [
        ("vac→SIA NN dist (W1, Å)", avg("vac_sia_nn_w1"),
         f"real {avg('vac_sia_nn_real_mean'):.1f} / gen {avg('vac_sia_nn_gen_mean'):.1f}"),
        ("radius of gyration (W1, Å)", avg("rg_w1"),
         f"real {avg('rg_real_mean'):.1f} / gen {avg('rg_gen_mean'):.1f}"),
        ("largest-cluster frac", abs(avg("largest_cluster_frac_real") - avg("largest_cluster_frac_gen")),
         f"real {avg('largest_cluster_frac_real'):.2f} / gen {avg('largest_cluster_frac_gen'):.2f}"),
    ]
    body = "".join(
        f"<tr><td>{n}</td><td style='text-align:right'>{v:.3f}</td>"
        f"<td style='text-align:right;color:#6b7280'>{ctx}</td></tr>"
        for n, v, ctx in rows)
    return ("<table class=req style='margin-top:8px'><tr><th>diagnostic (not weighted)</th>"
            f"<th>gen–real</th><th>mean</th></tr>{body}</table>")


def build_report(results_dir, out_html=None):
    out_html = out_html or os.path.join(results_dir, "report.html")
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "runs", "*"))):
        mp = os.path.join(run_dir, "run.json")
        sp = os.path.join(run_dir, "scorecard.json")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        manifest = json.load(open(mp))
        sc = json.load(open(sp))
        imgs = [_b64(p) for p in sorted(glob.glob(os.path.join(run_dir, "samples", "*.png")))]
        runs.append((manifest, sc, imgs))
    runs.sort(key=lambda x: x[1].get("score", 9e9))

    cards = ""
    for manifest, sc, imgs in runs:
        passed = sc["n_requirements_passed"]; total = sc["n_requirements"]
        sccol = "#16a34a" if sc["score"] <= 0.6 else "#d97706" if sc["score"] <= 1.1 else "#dc2626"
        img_html = "".join(f"<img src='{d}' class=cmp>" for d in imgs) or \
            "<div class=note>No generated-vs-real images for this run " \
            "(baselines have none; checkpoint runs render them).</div>"
        cards += f"""
        <section class=card>
          <h2>{manifest.get('label','?')}
            <span class=score style="background:{sccol}1a;color:{sccol}">score {sc['score']:.3f}</span>
            <span class=meta>{manifest.get('model',{}).get('architecture','?')} ·
              {passed}/{total} requirements · {manifest.get('created','')[:19].replace('T',' ')}
              {('· git '+(manifest.get('provenance',{}).get('git') or {}).get('commit','')[:7]) if (manifest.get('provenance',{}).get('git') or {}).get('commit') else ''}</span>
          </h2>
          <div class=cols>
            <div>{_req_table(sc)}{_diag_table(sc)}</div>
            <div><img src="{_curve_png(sc)}" class=curves>
              <div class=note>solid = real · dashed = generated · per energy</div></div>
          </div>
          <h3>Generated vs real cascades</h3>
          <div class=imgs>{img_html}</div>
        </section>"""

    best = runs[0][1]["score"] if runs else None
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Cascaide — Results Report</title><style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fc;color:#0f172a}}
 header{{padding:18px 26px;background:#0f1830;color:#fff}}
 header h1{{margin:0;font-size:18px}} header .s{{color:#9fb3d8;font-size:13px}}
 main{{max-width:1180px;margin:0 auto;padding:22px}}
 .card{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;padding:18px 20px;margin-bottom:22px;box-shadow:0 1px 3px #0f172a0d}}
 h2{{font-size:16px;margin:0 0 12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
 h3{{font-size:13px;color:#475569;margin:16px 0 8px;text-transform:uppercase;letter-spacing:.4px}}
 .score{{font-size:12px;font-weight:700;border-radius:7px;padding:2px 9px}}
 .meta{{font-size:12px;color:#64748b;font-weight:400}}
 .cols{{display:grid;grid-template-columns:380px 1fr;gap:18px;align-items:start}}
 table.req{{width:100%;border-collapse:collapse}}
 table.req th{{text-align:left;color:#64748b;font-size:12px;border-bottom:1px solid #e3e8f0;padding:5px 6px}}
 table.req td{{padding:5px 6px;border-bottom:1px solid #f1f5f9}}
 img.curves{{width:100%;border:1px solid #eef2f7;border-radius:8px}}
 .imgs{{display:flex;flex-wrap:wrap;gap:12px}}
 img.cmp{{width:100%;max-width:760px;border:1px solid #eef2f7;border-radius:10px}}
 .note{{color:#94a3b8;font-size:12px;margin-top:4px}}
</style></head><body>
<header><h1>Cascaide — Model Progress Report</h1>
<div class=s>{len(runs)} runs · best score {best:.3f} · sorted best→worst · generated (not encoded) vs real</div></header>
<main>{cards or '<div class=card>No runs yet. Run scripts/run_benchmark.py.</div>'}</main>
</body></html>"""
    with open(out_html, "w") as f:
        f.write(html)
    return out_html

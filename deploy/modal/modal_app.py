"""Modal launcher for ONE configurable Cascaide set-DiT v2 training run — SLEEP-SAFE.

Thin wrapper around `scripts/train_set_v2.py` (HPC stays primary; nothing in the `cascaide`
package imports modal). Uploads the processed `data/cascaide_cascades.npz` (no ovito, no raw
dumps).

SLEEP-SAFE design (so it survives the laptop sleeping / disconnecting):
  - The function writes the results store to a persistent **Modal Volume** in the cloud and
    `commit()`s it — NOT returned to the local client. So nothing depends on the laptop.
  - Launch DETACHED so the app survives client disconnect:
        CASCAIDE_GPU=A100 modal run --detach deploy/modal/modal_app.py
  - When you're back, pull the results locally + rebuild the dashboard:
        modal run deploy/modal/modal_app.py::fetch
  - `timeout` is a HARD 3h ceiling so it can't run all night; work is sized to ~1-2h.
    `--max-defects` caps the O(N^2) attention memory from the high-energy tail.
"""
import os
import modal

GPU = os.environ.get("CASCAIDE_GPU", "A100")          # A100 (fast, 40GB) | L40S | A10G | T4
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOL = modal.Volume.from_name("cascaide-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "matplotlib")
    .add_local_dir(os.path.join(REPO, "cascaide"), "/root/cascaide",
                   ignore=["**/dataset/**", "dataset"], copy=True)
    .add_local_file(os.path.join(REPO, "sp_cas.py"), "/root/sp_cas.py", copy=True)
    .add_local_file(os.path.join(REPO, "pyproject.toml"), "/root/pyproject.toml", copy=True)
    .add_local_dir(os.path.join(REPO, "scripts"), "/root/scripts", copy=True)
    .add_local_file(os.path.join(REPO, "data", "cascaide_cascades.npz"),
                    "/root/data/cascaide_cascades.npz", copy=True)
    .run_commands("cd /root && pip install -e . --no-deps")
)

app = modal.App("cascaide-setv2")


@app.function(image=image, gpu=GPU, timeout=60 * 60 * 5, volumes={"/root/results": VOL})
def train_one(epochs, center, energy_max, energy_bin, max_defects, count_cap,
              batch_size, select_every, select_n, finalize_n,
              augment_rot=False, save_every=0, no_results=False, run_name=None):
    """Train on a Modal GPU. Checkpoints are written DIRECTLY onto the Volume (out_dir under
    /root/results) so they survive the container — including the --save_every convergence
    trace — and persist on VOL.commit()."""
    import subprocess
    os.chdir("/root")
    tag = run_name or f"{center}_{epochs}ep" + ("_aug" if augment_rot else "")
    out_dir = f"/root/results/ckpts/{tag}"
    cmd = ["python", "scripts/train_set_v2.py",
           "--subset", "data/cascaide_cascades.npz",
           "--output_dir", out_dir,
           "--center", center, "--epochs", str(epochs),
           "--energy_bin", str(energy_bin), "--max_defects", str(max_defects),
           "--count_cap", str(count_cap), "--batch_size", str(batch_size),
           "--select_every", str(select_every), "--select_n", str(select_n),
           "--finalize_n", str(finalize_n), "--save_every", str(save_every),
           "--results", "/root/results", "--device", "cuda"]
    if energy_max and energy_max > 0:
        cmd += ["--energy_max", str(energy_max)]
    if augment_rot:
        cmd += ["--augment_rot"]
    if no_results:
        cmd += ["--no_results"]
    print(f"[modal] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    VOL.commit()                       # persist checkpoints + results in the cloud
    print(f"[modal] committed to Volume 'cascaide-results' -> {out_dir}")


@app.local_entrypoint()
def main(epochs: int = 250, center: str = "per_cascade", energy_max: float = 0.0,
         energy_bin: float = 25.0, max_defects: int = 1000, count_cap: int = 1200,
         batch_size: int = 16, select_every: int = 125, select_n: int = 8,
         finalize_n: int = 12, augment_rot: bool = False, save_every: int = 0,
         no_results: bool = False, run_name: str = ""):
    print(f"Modal run on {GPU}: full 0-300 keV, center={center}, epochs={epochs}, "
          f"augment_rot={augment_rot}, save_every={save_every}, no_results={no_results} "
          f"(5h ceiling; checkpoints -> Volume 'cascaide-results').")
    print("Launch with --detach so it survives sleep; fetch later with "
          "`modal run deploy/modal/modal_app.py::fetch`")
    train_one.remote(epochs, center, energy_max, energy_bin, max_defects,
                     count_cap, batch_size, select_every, select_n, finalize_n,
                     augment_rot=augment_rot, save_every=save_every,
                     no_results=no_results, run_name=run_name or None)
    print("Training submitted. Results are in the Volume when done.")


@app.local_entrypoint()
def fetch():
    """Pull the results store from the Volume to local ./results and rebuild the dashboard.
    Run this when you're back at the machine (laptop need not have been awake)."""
    from modal.volume import FileEntryType
    out_root = os.path.join(REPO, "results")
    n = 0
    for e in VOL.iterdir("/", recursive=True):
        if e.type == FileEntryType.FILE:
            dst = os.path.join(out_root, e.path.lstrip("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in VOL.read_file(e.path):
                    f.write(chunk)
            n += 1
    print(f"fetched {n} files -> {out_root}")
    from cascaide.eval import registry, report
    registry.collect(out_root)
    report.build_report(out_root)
    print("dashboard rebuilt — Continuity cover will show the run on reload.")

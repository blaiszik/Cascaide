"""Results store + registry — the bridge between HPC runs and the human dashboard.

Production (many async cluster jobs) is decoupled from viewing (one static dashboard) by a
file-based, versioned layout. Each job writes its OWN run directory (no write contention);
results are rsync'd off scratch; `collect()` rescans and rebuilds a single `index.json`
that the dashboard filters on. No database, no live server.

    results/
      index.json              # collect() output: flat list of run records (dashboard reads this)
      runs/<run_id>/
        run.json              # manifest: identity, provenance, model, HPC job info, status, tags
        scorecard.json        # metrics (from eval.scorecard)
        curves.json           # optional, large arrays split out
        samples/              # optional artifacts

`run_id` = "<UTCstamp>-<arch>-<6hash>" — sortable and unique. Manifest schema is versioned
so old cluster results stay loadable.
"""
import os
import json
import glob
import uuid
import hashlib
import subprocess
from datetime import datetime, timezone

MANIFEST_SCHEMA_VERSION = 1


# ------------------------------------------------------------------------- provenance
def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def capture_git(repo="."):
    cwd = os.getcwd()
    try:
        os.chdir(repo)
        commit = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = _run(["git", "status", "--porcelain"])
        return {"commit": commit, "branch": branch,
                "dirty": bool(dirty) if dirty is not None else None}
    finally:
        os.chdir(cwd)


def capture_hpc():
    """Identify the execution environment (SLURM, PBS/Cobalt, Modal, else local)."""
    e = os.environ
    if e.get("MODAL_TASK_ID") or e.get("MODAL_IMAGE_ID"):
        return {"scheduler": "modal", "job_id": e.get("MODAL_TASK_ID"),
                "host": e.get("MODAL_IMAGE_ID")}
    if e.get("SLURM_JOB_ID"):
        return {"scheduler": "slurm", "job_id": e.get("SLURM_JOB_ID"),
                "job_name": e.get("SLURM_JOB_NAME"), "host": e.get("SLURMD_NODENAME"),
                "nodes": e.get("SLURM_JOB_NUM_NODES"),
                "gpus": e.get("SLURM_GPUS_ON_NODE") or e.get("SLURM_GPUS"),
                "partition": e.get("SLURM_JOB_PARTITION")}
    if e.get("PBS_JOBID") or e.get("COBALT_JOBID"):
        return {"scheduler": "pbs", "job_id": e.get("PBS_JOBID") or e.get("COBALT_JOBID"),
                "host": e.get("HOSTNAME"), "nodes": e.get("PBS_NUM_NODES")}
    return {"scheduler": "local", "host": e.get("HOSTNAME") or _run(["hostname"])}


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def make_run_id(arch="model", label=""):
    # uuid4 suffix (not a content hash) so PARALLEL runs that finish in the same second with
    # the same label still get unique ids — a content hash collided in the first sweep.
    suffix = uuid.uuid4().hex[:8]
    safe = "".join(c if c.isalnum() else "-" for c in (arch or "model"))[:16].strip("-")
    return f"{_stamp()}-{safe}-{suffix}"


# ------------------------------------------------------------------------- manifest
def build_manifest(*, label, arch, encoder=None, normalization=None, params=None,
                   checkpoint=None, epoch=None, energies_keV=None, config=None,
                   config_path=None, tags=None, status="done", repo="."):
    """Assemble a run manifest (run.json contents). Captures git + HPC env automatically."""
    config_hash = None
    if config is not None:
        config_hash = hashlib.sha1(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "label": label,
        "status": status,
        "created": _now_iso(),
        "finished": _now_iso() if status in ("done", "failed") else None,
        "provenance": {"git": capture_git(repo), "config_path": config_path,
                       "config_hash": config_hash, "config": config},
        "model": {"architecture": arch, "encoder": encoder,
                  "normalization": normalization, "params": params,
                  "checkpoint": checkpoint, "epoch": epoch},
        "hpc": capture_hpc(),
        "energies_keV": energies_keV,
        "tags": tags or [],
    }


def write_run(results_dir, manifest, scorecard=None, curves=None, run_id=None):
    """Create results/runs/<run_id>/ and write run.json (+ scorecard/curves). Returns dir."""
    run_id = run_id or make_run_id(manifest.get("model", {}).get("architecture", "model"),
                                   manifest.get("label", ""))
    run_dir = os.path.join(results_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    manifest = {**manifest, "run_id": run_id,
                "artifacts": {"scorecard": "scorecard.json" if scorecard else None,
                              "curves": "curves.json" if curves else None}}
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if scorecard is not None:
        with open(os.path.join(run_dir, "scorecard.json"), "w") as f:
            json.dump(scorecard, f, indent=2)
    if curves is not None:
        with open(os.path.join(run_dir, "curves.json"), "w") as f:
            json.dump(curves, f, indent=2)
    return run_dir


# ------------------------------------------------------------------------- collect/index
def _flatten_record(manifest, scorecard):
    m, model = manifest, manifest.get("model", {})
    rec = {
        "run_id": m.get("run_id"),
        "label": m.get("label"),
        "status": m.get("status"),
        "created": m.get("created"),
        "finished": m.get("finished"),
        "arch": model.get("architecture"),
        "encoder": model.get("encoder"),
        "normalization": model.get("normalization"),
        "params": model.get("params"),
        "epoch": model.get("epoch"),
        "git_commit": (m.get("provenance", {}).get("git") or {}).get("commit"),
        "git_branch": (m.get("provenance", {}).get("git") or {}).get("branch"),
        "config_hash": m.get("provenance", {}).get("config_hash"),
        "hpc_job": (m.get("hpc") or {}).get("job_id"),
        "scheduler": (m.get("hpc") or {}).get("scheduler"),
        "tags": m.get("tags", []),
        "energies": m.get("energies_keV"),
        "path": f"runs/{m.get('run_id')}/",
    }
    if scorecard:
        rec["score"] = scorecard.get("score")
        rec["n_passed"] = scorecard.get("n_requirements_passed")
        rec["n_requirements"] = scorecard.get("n_requirements")
        # headline substructure metrics for at-a-glance filtering/sorting
        rec["key_metrics"] = {
            r["key"]: r["value"] for r in scorecard.get("requirements", [])
        }
    return rec


def collect(results_dir, write=True):
    """Rescan results/runs/*/ and (re)build index.json. Idempotent.

    This is the "filter results back into the dashboard" step: after new run dirs land
    (e.g. rsync'd off the cluster), call collect() and the dashboard sees them on reload.
    """
    records = []
    for run_dir in sorted(glob.glob(os.path.join(results_dir, "runs", "*"))):
        mpath = os.path.join(run_dir, "run.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            manifest = json.load(f)
        scorecard = None
        spath = os.path.join(run_dir, "scorecard.json")
        if os.path.exists(spath):
            with open(spath) as f:
                scorecard = json.load(f)
        records.append(_flatten_record(manifest, scorecard))

    # newest first
    records.sort(key=lambda r: r.get("created") or "", reverse=True)
    index = {"schema_version": MANIFEST_SCHEMA_VERSION,
             "generated": _now_iso(), "n_runs": len(records), "runs": records}
    if write:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
    return index

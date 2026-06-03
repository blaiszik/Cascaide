"""Scorecard-based model selection.

`best_model.pt` froze at ~ep50 because val coord-MSE plateaus early and is decoupled from
substructure quality. So select/early-stop on the SCORECARD instead. These helpers score a
set of generated samples (or a checkpoint) and return the single fitness scalar — call
periodically in training and keep the best.
"""
from cascaide.eval import scorecard as _scorecard


def score_generated(generated, reference, label="select", energy_bin=None):
    """Return (score, scorecard_dict) for already-generated samples vs reference.
    Lower score = better. ``energy_bin`` bins continuous energies for grouping."""
    sc = _scorecard.compute(generated, reference, label=label, energy_bin=energy_bin)
    return sc["score"], sc


def score_set_checkpoint(checkpoint, reference, n_per_energy=16, device=None,
                         label="ckpt"):
    """Load a set checkpoint, generate, and score it. Convenience for offline selection."""
    from cascaide.eval.generators import generate_set_checkpoint
    gen, _ = generate_set_checkpoint(checkpoint, reference, device=device,
                                     n_per_energy=n_per_energy)
    return score_generated(gen, reference, label=label)

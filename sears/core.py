from math import isfinite
from statistics import median

import torch
import torch.nn.functional as F


RANK = 16
SCALE = 2.0
THRESHOLD = 0.95
TEMPERATURE = 0.05
EPS = 1e-12


def allocate(prototype, prototypes, assignment, roots):
    if not assignment:
        return 0
    similarities = torch.stack(prototypes) @ prototype
    best = int(similarities.argmax())
    return assignment[best] if float(similarities[best]) >= THRESHOLD else roots


def root_probabilities(features, prototypes, assignment):
    similarities = F.normalize(features.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
    probabilities = (similarities / TEMPERATURE).softmax(dim=-1)
    index = torch.tensor(assignment, device=features.device, dtype=torch.long)
    roots = probabilities.new_zeros((probabilities.shape[0], max(assignment) + 1))
    return roots.index_add_(1, index, probabilities)


def effective_weight(factors, device):
    a, b = factors
    return SCALE * (b.to(device=device, dtype=torch.float32) @ a.to(device=device, dtype=torch.float32))


def change_profile(delta):
    energy = delta.square()
    center = float(energy.median())
    mean = float(energy.mean())
    if not isfinite(center) or not isfinite(mean):
        raise FloatingPointError("Non-finite squared LoRA changes")
    return center, center / mean if center > 0 and mean > 0 else None


@torch.no_grad()
def adaptive_retention_synthesis(stored, trained, device):
    profiles = {}
    for key in stored:
        delta = effective_weight(trained[key], device) - effective_weight(stored[key], device)
        profiles[key] = change_profile(delta)
    ratios = [ratio for _, ratio in profiles.values() if ratio is not None]
    if not ratios:
        return stored
    kappa = median(ratios)
    restored = {}
    for key, (old_a, old_b) in stored.items():
        old = effective_weight((old_a, old_b), device)
        new = effective_weight(trained[key], device)
        delta = new - old
        energy = delta.square()
        c = max(profiles[key][0] / kappa, EPS) + EPS
        confidence = energy / (energy + c)
        new_amplitude = (new.abs() + EPS).square()
        old_amplitude = (old.abs() + EPS).square()
        amplitude_share = new_amplitude / (old_amplitude + new_amplitude + EPS)
        target = old + confidence * amplitude_share * delta
        del old, new, delta, energy, confidence, new_amplitude, old_amplitude, amplitude_share
        u, s, vh = torch.linalg.svd(target, full_matrices=False)
        rank = old_a.shape[0]
        sigma = (s[:rank] / SCALE).sqrt()
        restored[key] = ((sigma[:, None] * vh[:rank]).cpu(), (u[:, :rank] * sigma[None, :]).cpu())
        del target, u, s, vh, sigma
    return restored

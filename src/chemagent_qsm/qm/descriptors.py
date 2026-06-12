from __future__ import annotations

import numpy as np

HARTREE_TO_EV = 27.211386245988


def orbital_gap_ev(mo_energy, mo_occ) -> float | None:
    eps = np.asarray(mo_energy, dtype=float).reshape(-1)
    occ = np.asarray(mo_occ, dtype=float).reshape(-1)
    occ_idx = np.where(occ > 1e-8)[0]
    virt_idx = np.where(occ <= 1e-8)[0]
    if len(occ_idx) == 0 or len(virt_idx) == 0:
        return None
    return float((eps[virt_idx[0]] - eps[occ_idx[-1]]) * HARTREE_TO_EV)


def summarize_orbitals(mo_energy, mo_occ) -> dict:
    eps = np.asarray(mo_energy, dtype=float).reshape(-1)
    occ = np.asarray(mo_occ, dtype=float).reshape(-1)
    occ_idx = np.where(occ > 1e-8)[0]
    virt_idx = np.where(occ <= 1e-8)[0]
    homo = float(eps[occ_idx[-1]] * HARTREE_TO_EV) if len(occ_idx) else None
    lumo = float(eps[virt_idx[0]] * HARTREE_TO_EV) if len(virt_idx) else None
    gap = None if homo is None or lumo is None else float(lumo - homo)
    out = {"homo_ev": homo, "lumo_ev": lumo, "gap_ev": gap}
    out["electronic_descriptors"] = frontier_orbital_reactivity(homo, lumo)
    return out


def frontier_orbital_reactivity(homo_ev: float | None, lumo_ev: float | None) -> dict[str, float | None]:
    """Koopmans-style conceptual-DFT descriptors from frontier orbital energies.

    These are screening descriptors, not substitutes for vertical IP/EA calculations. They are
    useful in drug-discovery triage because they summarize electron donation/acceptance and
    reactivity from the electronic-structure calculation in a compact, auditable form.
    """
    if homo_ev is None or lumo_ev is None:
        return {
            "koopmans_ip_ev": None,
            "koopmans_ea_ev": None,
            "electronegativity_ev": None,
            "chemical_potential_ev": None,
            "hardness_ev": None,
            "softness_ev-1": None,
            "electrophilicity_ev": None,
        }
    ip = -float(homo_ev)
    ea = -float(lumo_ev)
    electronegativity = 0.5 * (ip + ea)
    chemical_potential = -electronegativity
    hardness = 0.5 * (ip - ea)
    softness = None if abs(hardness) < 1e-12 else 1.0 / (2.0 * hardness)
    electrophilicity = None if abs(hardness) < 1e-12 else chemical_potential * chemical_potential / (2.0 * hardness)
    return {
        "koopmans_ip_ev": ip,
        "koopmans_ea_ev": ea,
        "electronegativity_ev": electronegativity,
        "chemical_potential_ev": chemical_potential,
        "hardness_ev": hardness,
        "softness_ev-1": softness,
        "electrophilicity_ev": electrophilicity,
    }

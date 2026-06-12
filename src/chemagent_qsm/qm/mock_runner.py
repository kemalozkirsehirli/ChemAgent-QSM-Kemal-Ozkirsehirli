from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from chemagent_qsm.io.molecule_io import parse_atom_block
from chemagent_qsm.qm.descriptors import summarize_orbitals
from chemagent_qsm.qm.spectra import gaussian_ir_spectrum
from chemagent_qsm.schema import MoleculeSpec, QMSettings

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


@dataclass
class MockQMRunner:
    settings: QMSettings

    def _atoms(self, molecule: MoleculeSpec):
        if molecule.atom_block:
            return parse_atom_block(molecule.atom_block)
        if molecule.smiles:
            # A deterministic approximate pseudo-atomization for offline tests.
            counts = []
            for ch, sym in [("C", "C"), ("N", "N"), ("O", "O"), ("S", "S"), ("P", "P")]:
                counts.extend([(sym, 0.0, 0.0, 0.0)] * molecule.smiles.count(ch))
            if not counts:
                counts = [("C", 0.0, 0.0, 0.0)]
            return counts
        raise ValueError("Mock runner requires atom_block or smiles.")

    def _seed(self, molecule: MoleculeSpec) -> int:
        text = molecule.model_dump_json() + self.settings.model_dump_json()
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def single_point(self, molecule: MoleculeSpec) -> dict[str, Any]:
        atoms = self._atoms(molecule)
        ztot = sum(ATOMIC_NUMBERS.get(sym, 6) for sym, *_ in atoms) - molecule.charge
        seed = self._seed(molecule)
        correction = (seed % 10000) / 100000.0
        energy = -0.48 * ztot - 0.015 * len(atoms) - correction
        n_orb = max(4, len(atoms) * 4)
        mo_energy = np.linspace(-0.65, 0.35, n_orb) + (seed % 37) * 1e-4
        n_occ = max(1, min(n_orb - 1, ztot // 2))
        mo_occ = np.zeros(n_orb)
        mo_occ[:n_occ] = 2
        return {
            "backend": "mock",
            "method": str(self.settings.method),
            "basis": self.settings.basis,
            "xc": self.settings.xc,
            "converged": True,
            "total_energy_hartree": float(energy),
            "num_atoms": len(atoms),
            "num_electrons_approx": int(ztot),
            "mo_energy_hartree": mo_energy.tolist(),
            "mo_occ": mo_occ.tolist(),
        }

    def optimize(self, molecule: MoleculeSpec) -> tuple[MoleculeSpec, dict[str, Any]]:
        atoms = self._atoms(molecule)
        if molecule.atom_block:
            optimized = molecule.atom_block
        else:
            optimized = "\n".join(f"{sym} {i*0.7:.6f} 0.000000 0.000000" for i, (sym, *_xyz) in enumerate(atoms))
        new_mol = molecule.model_copy(update={"atom_block": optimized})
        return new_mol, {
            "backend": "mock",
            "optimized": True,
            "num_steps": min(12, max(3, len(atoms))),
            "final_gradient_norm": 1.0e-4,
            "atom_block": optimized,
        }

    def descriptors(self, molecule: MoleculeSpec, single_point: dict[str, Any] | None = None) -> dict[str, Any]:
        sp = single_point or self.single_point(molecule)
        mo_e = np.asarray(sp["mo_energy_hartree"])
        mo_o = np.asarray(sp["mo_occ"])
        occ = np.where(mo_o > 1e-8)[0]
        virt = np.where(mo_o <= 1e-8)[0]
        hartree_to_ev = 27.211386245988
        homo = float(mo_e[occ[-1]] * hartree_to_ev) if len(occ) else None
        lumo = float(mo_e[virt[0]] * hartree_to_ev) if len(virt) else None
        atoms = self._atoms(molecule)
        charges = [float((ATOMIC_NUMBERS.get(sym, 6) % 5) * 0.05 - 0.1) for sym, *_ in atoms]
        orbital_summary = summarize_orbitals(mo_e, mo_o)
        return {
            "backend": "mock",
            "total_energy_hartree": sp["total_energy_hartree"],
            **orbital_summary,
            "dipole_debye": [float(len(atoms) * 0.03), 0.0, float(len(atoms) * 0.01)],
            "mulliken_charges": charges,
            "drug_discovery_bridge": {
                "descriptor_family": "frontier-orbital conceptual DFT + partial charge screen",
                "intended_use": "rank or triage drug-like molecules before higher-cost binding/free-energy workflows",
                "limitations": "Koopmans-style descriptors are approximate and should be calibrated against assay or higher-level QM data.",
            },
        }

    def frequencies(self, molecule: MoleculeSpec) -> dict[str, Any]:
        atoms = self._atoms(molecule)
        n_modes = max(1, 3 * len(atoms) - 6)
        base = np.linspace(550.0, 3450.0, n_modes)
        seed = self._seed(molecule) % 31
        freqs = base + seed
        intensities = 0.2 + np.abs(np.sin(freqs / 333.0))
        return {
            "backend": "mock",
            "frequencies_cm-1": freqs.tolist(),
            "ir_intensities": intensities.tolist(),
            "has_imaginary_mode": False,
        }

    def ir_spectrum(self, molecule: MoleculeSpec, frequencies: dict[str, Any] | None = None) -> dict[str, Any]:
        freq_data = frequencies or self.frequencies(molecule)
        x, y = gaussian_ir_spectrum(
            freq_data["frequencies_cm-1"],
            freq_data.get("ir_intensities"),
            self.settings.ir_min_cm,
            self.settings.ir_max_cm,
            self.settings.ir_points,
            self.settings.ir_width_cm,
        )
        return {"x_cm-1": x.tolist(), "intensity": y.tolist()}

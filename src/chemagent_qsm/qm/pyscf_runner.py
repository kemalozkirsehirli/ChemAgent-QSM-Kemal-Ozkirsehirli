from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chemagent_qsm.exceptions import OptionalDependencyError
from chemagent_qsm.io.molecule_io import atom_block_from_spec
from chemagent_qsm.qm.descriptors import summarize_orbitals
from chemagent_qsm.qm.spectra import gaussian_ir_spectrum
from chemagent_qsm.schema import MoleculeSpec, QMMethod, QMSettings


@dataclass
class PySCFRunner:
    settings: QMSettings

    def _imports(self):
        try:
            from pyscf import dft, gto, lib, scf
        except Exception as exc:
            raise OptionalDependencyError("PySCF is not installed. Install chemagent-qsm[qm].") from exc
        return gto, scf, dft, lib

    def build_mol(self, molecule: MoleculeSpec):
        gto, _scf, _dft, lib = self._imports()
        if self.settings.num_threads:
            lib.num_threads(self.settings.num_threads)
        atom_block = atom_block_from_spec(molecule)
        mol = gto.Mole()
        mol.atom = atom_block
        mol.basis = self.settings.basis
        mol.charge = molecule.charge
        mol.spin = molecule.spin
        mol.unit = molecule.unit
        mol.max_memory = self.settings.memory_mb
        mol.build()
        return mol

    def make_mf(self, mol):
        _gto, scf, dft, _lib = self._imports()
        if self.settings.method == QMMethod.HF:
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = self.settings.xc
        mf.max_cycle = self.settings.max_scf_cycles
        mf.conv_tol = self.settings.conv_tol
        if self.settings.density_fit and hasattr(mf, "density_fit"):
            mf = mf.density_fit()
        if self.settings.newton and hasattr(mf, "newton"):
            mf = mf.newton()
        if self.settings.use_gpu:
            try:
                mf = mf.to_gpu()
            except Exception:
                pass
        return mf

    def single_point(self, molecule: MoleculeSpec) -> dict[str, Any]:
        mol = self.build_mol(molecule)
        mf = self.make_mf(mol)
        energy = mf.kernel()
        mo_energy = np.asarray(mf.mo_energy).reshape(-1)
        mo_occ = np.asarray(mf.mo_occ).reshape(-1)
        return {
            "backend": "pyscf",
            "method": str(self.settings.method),
            "basis": self.settings.basis,
            "xc": self.settings.xc if self.settings.method == QMMethod.DFT else None,
            "converged": bool(getattr(mf, "converged", False)),
            "total_energy_hartree": float(energy),
            "num_atoms": int(mol.natm),
            "num_electrons": int(mol.nelectron),
            "mo_energy_hartree": mo_energy.tolist(),
            "mo_occ": mo_occ.tolist(),
        }

    def optimize(self, molecule: MoleculeSpec) -> tuple[MoleculeSpec, dict[str, Any]]:
        mol = self.build_mol(molecule)
        mf = self.make_mf(mol)
        if self.settings.geomopt_backend == "berny":
            from pyscf.geomopt.berny_solver import optimize
        else:
            from pyscf.geomopt.geometric_solver import optimize
        mol_eq = optimize(mf, maxsteps=self.settings.geomopt_maxsteps)
        atom_block = "\n".join(
            f"{mol_eq.atom_symbol(i)} {x:.10f} {y:.10f} {z:.10f}"
            for i, (x, y, z) in enumerate(mol_eq.atom_coords(unit=molecule.unit))
        )
        new_mol = molecule.model_copy(update={"atom_block": atom_block})
        return new_mol, {
            "backend": "pyscf",
            "optimized": True,
            "geomopt_backend": self.settings.geomopt_backend,
            "atom_block": atom_block,
        }

    def descriptors(self, molecule: MoleculeSpec, single_point: dict[str, Any] | None = None) -> dict[str, Any]:
        mol = self.build_mol(molecule)
        mf = self.make_mf(mol)
        energy = mf.kernel()
        orbital_summary = summarize_orbitals(mf.mo_energy, mf.mo_occ)
        try:
            dipole = [float(x) for x in mf.dip_moment(unit="Debye")]
        except Exception:
            dipole = None
        charges = None
        try:
            _pop, chg = mf.mulliken_pop(verbose=0)
            charges = [float(x) for x in np.asarray(chg).reshape(-1)]
        except Exception:
            pass
        return {
            "backend": "pyscf",
            "method": str(self.settings.method),
            "basis": self.settings.basis,
            "xc": self.settings.xc if self.settings.method == QMMethod.DFT else None,
            "density_fit": self.settings.density_fit,
            "total_energy_hartree": float(energy),
            **orbital_summary,
            "dipole_debye": dipole,
            "mulliken_charges": charges,
            "drug_discovery_bridge": {
                "descriptor_family": "frontier-orbital conceptual DFT + dipole + Mulliken population screen",
                "intended_use": "rank or triage drug-like molecules before higher-cost docking, binding-affinity, or free-energy workflows",
                "limitations": "Use calibrated benchmarks before interpreting these descriptors as potency, selectivity, or ADMET predictors.",
            },
        }

    def frequencies(self, molecule: MoleculeSpec) -> dict[str, Any]:
        mol = self.build_mol(molecule)
        mf = self.make_mf(mol)
        mf.kernel()
        hess = mf.Hessian().kernel()
        try:
            from pyscf.hessian import thermo
            results = thermo.harmonic_analysis(mol, hess)
            freqs = np.asarray(results["freq_wavenumber"], dtype=float)
            intens = np.ones_like(freqs)
        except Exception:
            flat = np.asarray(hess, dtype=float).reshape(3 * mol.natm, 3 * mol.natm)
            vals = np.linalg.eigvalsh((flat + flat.T) / 2)
            freqs = np.sign(vals) * np.sqrt(np.abs(vals)) * 5140.48
            intens = np.ones_like(freqs)
        return {
            "backend": "pyscf",
            "frequencies_cm-1": freqs.tolist(),
            "ir_intensities": intens.tolist(),
            "has_imaginary_mode": bool(np.any(freqs < -20.0)),
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

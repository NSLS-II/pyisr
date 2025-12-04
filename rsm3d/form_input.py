#!/usr/bin/env python3
"""
spec_parser.py

Defines ExperimentSetup, Crystal, ScanAngles, and SpecParser classes to parse
SPEC-format files and return scan data as a pandas or Dask DataFrame.
"""

import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import yaml
from pathlib import Path


class ExperimentSetup:
    """
    Load experiment parameters from a YAML file. Wavelength is optional:
      • if provided and >1e-3 Å, used directly
      • if provided in meters (<1e-3), converted to Å
      • if omitted or non‐positive, computed from energy [Å] = 12.398419843320026 / E[keV]

    Required keys (either top-level or inside `ExperimentSetup:`):
      distance, pitch, ycenter, xcenter, xpixels, ypixels, energy

    Optional key:
      wavelength
    """
    REQUIRED_KEYS = (
        "distance", "pitch", "ycenter", "xcenter",
        "xpixels", "ypixels", "energy",
    )

    def __init__(
        self,
        distance: float,
        pitch: float,
        ycenter: int,
        xcenter: int,
        xpixels: int,
        ypixels: int,
        energy: float,
        wavelength: float | None = None,
    ):
        self.distance = float(distance)
        self.pitch = float(pitch)
        self.ycenter = int(ycenter)
        self.xcenter = int(xcenter)
        self.xpixels = int(xpixels)
        self.ypixels = int(ypixels)
        self.energy = float(energy)
        self.energy_keV = float(energy)
        if self.distance <= 0:
            raise ValueError("ExperimentSetup: 'distance' must be > 0")
        if self.pitch <= 0:
            raise ValueError("ExperimentSetup: 'pitch' must be > 0")
        if self.xpixels <= 0 or self.ypixels <= 0:
            raise ValueError("ExperimentSetup: 'xpixels' and 'ypixels' must be > 0")
        if self.energy_keV <= 0:
            raise ValueError("ExperimentSetup: 'energy' (keV) must be > 0")

        lam_A: float | None = None
        if wavelength is not None:
            try:
                lam_A = float(wavelength)
            except (TypeError, ValueError):
                lam_A = None
        if lam_A is not None and 0.0 < lam_A < 1e-3:
            lam_A *= 1e10
        if lam_A is None or lam_A <= 0.0:
            lam_A = self._energy_keV_to_lambda_A(self.energy_keV)
        if lam_A <= 0.0:
            raise ValueError("ExperimentSetup: computed wavelength is non-positive")
        self.wavelength = lam_A

    @staticmethod
    def _energy_keV_to_lambda_A(E_keV: float) -> float:
        return 12.398419843320026 / float(E_keV)

    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(str(v).replace("_", "").strip())
            except Exception:
                raise ValueError(f"Expected float-compatible value, got {v!r}")

    @staticmethod
    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            try:
                return int(float(str(v).replace("_", "").strip()))
            except Exception:
                raise ValueError(f"Expected int-compatible value, got {v!r}")

    @classmethod
    def _extract_section(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping of keys to values.")
        for key in ("ExperimentSetup", "experiment", "experiment_setup"):
            sec = data.get(key)
            if isinstance(sec, dict):
                return sec
        if any(k in data for k in cls.REQUIRED_KEYS):
            return data
        for v in data.values():
            if isinstance(v, dict) and any(k in v for k in cls.REQUIRED_KEYS):
                return v
        raise ValueError(
            "Could not find experiment setup in YAML. "
            "Expected an 'ExperimentSetup' section or flat keys."
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Experiment YAML not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sec = cls._extract_section(doc)
        merged = {}
        for k in cls.REQUIRED_KEYS + ("wavelength",):
            if k in sec:
                merged[k] = sec[k]
            elif k in doc:
                merged[k] = doc[k]
        missing = [k for k in cls.REQUIRED_KEYS if merged.get(k) in (None, "", "None", "null")]
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")
        params = {
            "distance":  cls._to_float(merged["distance"]),
            "pitch":     cls._to_float(merged["pitch"]),
            "ycenter":   cls._to_int(merged["ycenter"]),
            "xcenter":   cls._to_int(merged["xcenter"]),
            "xpixels":   cls._to_int(merged["xpixels"]),
            "ypixels":   cls._to_int(merged["ypixels"]),
            "energy":    cls._to_float(merged["energy"]),
            "wavelength": merged.get("wavelength", None),
        }
        return cls(**params)

    def __repr__(self):
        return (
            f"<ExperimentSetup: distance={self.distance} m, pitch={self.pitch} m, "
            f"xcenter={self.xcenter}, ycenter={self.ycenter}, "
            f"xpixels={self.xpixels}, ypixels={self.ypixels}, "
            f"energy={self.energy} keV, wavelength={self.wavelength} Å>"
        )

class Crystal:
    pass

class ScanAngles:
    ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
    HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

    # Modified __init__ to also accept a Crystal object (if needed later)
    def __init__(self, filename, crystal=None, npartitions=1):
        self.filename    = filename
        self.npartitions = npartitions
        self.crystal     = crystal

    def parse_all_scans(self):
        """
        Read the SPEC file and return a list of dicts with scan data.
        Each record will include:
          - scan_number: zero-padded scan number
          - data_number: zero-based, zero-padded data row index
          - type: scan type ('ascan' or 'hklscan')
          - tth, th, chi, phi: goniometer angles
          - h, k, l: reciprocal-lattice coordinates
          - ub: 3×3 UB matrix read from the "#G3" line within that scan (if present), otherwise None.
        """
        # Read global #O0 ordering from the SPEC file.
        o0_names = []
        with open(self.filename) as f:
            for line in f:
                if line.startswith('#O0 '):
                    o0_names = line[4:].split()
                    break
        if not o0_names:
            raise RuntimeError("Missing global #O0 line in SPEC file")

       
        results = []
        cur_scan = None
        cur_type = None
        skip_current = False      # ignore 'ascan' scans
        p0_map   = {}
        data_idx = {}
        in_data  = False
        counter  = 0
        current_ub = None         # per-scan UB

        with open(self.filename) as f:
            for raw in f:
                line = raw.strip()
                if line.startswith('#S '):
                    parts = line.split()
                    cur_scan = int(parts[1])
                    cur_type = parts[2] if len(parts) > 2 else ''
                    skip_current = (cur_type.lower() == 'ascan')
                    p0_map.clear()
                    data_idx.clear()
                    in_data = False
                    counter = 0
                    current_ub = None
                    continue

                # Skip all lines in ascans
                if skip_current:
                    continue

                # Look for UB update within a scan: "#G3" line.
                if cur_scan is not None and line.startswith('#G3 '):
                    # Parse UB: assume nine numbers follow "#G3"
                    ub_vals = [float(x) for x in line.split()[1:]]
                    if len(ub_vals) != 9:
                        raise RuntimeError(f"Scan {cur_scan}: UB line does not have 9 values.")
                    current_ub = np.array(ub_vals).reshape((3, 3))
                    continue

                # Grab fixed motors from "#P0" line (for ascan).
                if cur_scan is not None and line.startswith('#P0 '):
                    vals = [float(x) for x in line.split()[1:]]
                    p0_map = {name: vals[i] for i, name in enumerate(o0_names)}
                    continue

                # Data header (#L): only parse hklscan
                if cur_scan is not None and line.startswith('#L '):
                    cols = line.split()[1:]
                    if cur_type.lower() != 'hklscan':
                        in_data = False
                        continue
                    # hklscan columns
                    for ax in self.HKL_AXES:
                        data_idx[ax] = cols.index(ax)
                    for hk in ('H', 'K', 'L'):
                        data_idx[hk] = cols.index(hk)
                    in_data = True
                    continue
    
                if in_data:
                    if not line or (line.startswith('#') and not line[1].isdigit()):
                       in_data = False
                       continue
                    parts = line.split()
                    if len(parts) < max(data_idx.values()) + 1:
                       continue
                    rec = {
                       'scan_number': f"{cur_scan:03d}",
                       'data_number': f"{counter:03d}",
                       'type': cur_type,
                       'ub': current_ub.copy() if current_ub is not None else None
                    }
                    # parse hklscan fields
                    rec.update({
                        'tth': float(parts[data_idx['VTTH']]),
                        'th':  float(parts[data_idx['VTH']]),
                        'chi': float(parts[data_idx['Chi']]),
                        'phi': float(parts[data_idx['Phi']]),
                        'h':   float(parts[data_idx['H']]),
                        'k':   float(parts[data_idx['K']]),
                        'l':   float(parts[data_idx['L']]),
                    })
                    results.append(rec)
                    counter += 1
        return results

    def to_pandas(self):
        data = self.parse_all_scans()
        return pd.DataFrame(data)

    def to_dask(self):
        return dd.from_pandas(self.to_pandas(), npartitions=self.npartitions)

class FormInput:
    """
    Aggregate ExperimentSetup details with parsed scan metadata for a SPEC file.
    """
    def __init__(self, filename: str, setup_yaml: str, npartitions: int = 1):
        self.filename = filename
        self.npartitions = npartitions
        self.setup = ExperimentSetup.from_yaml(setup_yaml)
        self.scans = ScanAngles(filename, npartitions=npartitions)

    def to_pandas(self):
        return self.scans.to_pandas()

    def to_dask(self):
        return self.scans.to_dask()

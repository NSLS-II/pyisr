# # #!/usr/bin/env python3
# # """
# # spec_parser.py

# # Defines SpecParser class to parse SPEC-format files and return data as a pandas or Dask DataFrame.
# # """


# # import sys
# # import numpy as np
# # import pandas as pd
# # import dask.dataframe as dd

# # class ExperimentSetup:
# #     """
# #     Encapsulates the experimental geometry and detector parameters.

# #     Attributes:
# #         distance  (float): Sample-to-detector distance (mm)
# #         pitch     (float): Detector pixel size (µm)
# #         ycenter   (int)  : Vertical center pixel index
# #         xcenter   (int)  : Horizontal center pixel index
# #         xpixels   (int)  : Number of horizontal pixels
# #         ypixels   (int)  : Number of vertical pixels
# #         wavelength(float): Incident beam wavelength (Å)
# #         phi       (float): Detector horizontal two-theta (°)
# #         theta     (float): Starting theta position (°)
# #         dtheta    (float): Two-theta step increment (°)
# #         energy    (float): Beam energy (eV)
# #     """
# #     def __init__(self,
# #                  distance=781.05,
# #                  pitch=0.75,
# #                  ycenter=257,
# #                  xcenter=515,
# #                  xpixels=1030,
# #                  ypixels=514,
# #                  wavelength=0.283383,
# #                  phi=0.0,
# #                  theta=15.3069,
# #                  dtheta=0.04,
# #                  energy=11470.0):
# #         self.distance   = distance
# #         self.pitch      = pitch
# #         self.ycenter    = ycenter
# #         self.xcenter    = xcenter
# #         self.xpixels    = xpixels
# #         self.ypixels    = ypixels
# #         self.wavelength = wavelength
# #         self.phi        = phi
# #         self.theta      = theta
# #         self.dtheta     = dtheta
# #         self.energy     = energy

# #     def __repr__(self):
# #         return (
# #             f"ExperimentSetup(distance={self.distance} mm, pitch={self.pitch} µm, "
# #             f"ycenter={self.ycenter}, xcenter={self.xcenter}, xpixels={self.xpixels}, "
# #             f"ypixels={self.ypixels}, wavelength={self.wavelength} Å, phi={self.phi}°, "
# #             f"theta={self.theta}°, dtheta={self.dtheta}°, energy={self.energy} eV)"
# #         )

# # class Crystal:
# #     """
# #     Holds crystal lattice parameters and orientation matrix (UB).

# #     Attributes (from #G1 terms):
# #         a, b, c             : lattice constants
# #         alpha, beta, gamma  : lattice angles
# #         a_hkl, b_hkl, c_hkl : reflection directions
# #         alpha_hkl, beta_hkl, gamma_hkl : reflection angles
# #         H_or0, K_or0, L_or0 : original hkl
# #         H_or1, K_or1, L_or1 : transformed hkl
# #         u00..u05, u10..u15, u06, u16 : orientation components
# #         lambda0, lambda1    : wavelength multipliers
# #         UB                  : 3×3 orientation matrix (numpy array)
# #     """
# #     def __init__(self, params, ub_matrix):
# #         # params: list or array of 34 float terms from #G1
# #         # ub_matrix: list or array of 9 float terms from #G3
# #         self.a, self.b, self.c = params[0:3]
# #         self.alpha, self.beta, self.gamma = params[3:6]
# #         self.a_hkl, self.b_hkl, self.c_hkl = params[6:9]
# #         self.alpha_hkl, self.beta_hkl, self.gamma_hkl = params[9:12]
# #         self.H_or0, self.K_or0, self.L_or0 = params[12:15]
# #         self.H_or1, self.K_or1, self.L_or1 = params[15:18]
# #         (self.u00, self.u01, self.u02,
# #          self.u03, self.u04, self.u05) = params[18:24]
# #         (self.u10, self.u11, self.u12,
# #          self.u13, self.u14, self.u15) = params[24:30]
# #         self.lambda0, self.lambda1 = params[30:32]
# #         self.u06, self.u16 = params[32:34]
# #         # build UB matrix
# #         ub = np.array(ub_matrix, dtype=float)
# #         self.UB = ub.reshape((3,3))

# #     @classmethod
# #     def from_spec(cls, filename):
# #         """
# #         Parse #G1 and #G3 lines from the SPEC file to construct a Crystal.
# #         """
# #         g1_vals, g3_vals = None, None
# #         with open(filename) as f:
# #             for line in f:
# #                 if line.startswith('#G1 '):
# #                     g1_vals = [float(x) for x in line.split()[1:]]
# #                 elif line.startswith('#G3 '):
# #                     g3_vals = [float(x) for x in line.split()[1:]]
# #                 if g1_vals is not None and g3_vals is not None:
# #                     break
# #         if g1_vals is None or g3_vals is None:
# #             raise RuntimeError("SPEC file missing #G1 or #G3 for crystal parameters")
# #         return cls(g1_vals, g3_vals)

# #     def __repr__(self):
# #         return (
# #             f"Crystal(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, beta={self.beta}, "
# #             f"gamma={self.gamma}, UB=\n{self.UB})"
# #         )


# # class SpecParser:
# #     """
# #     Parse SPEC-format files and convert scan data into a pandas or Dask DataFrame.

# #     Attributes:
# #         filename (str): Path to the SPEC file.
# #         npartitions (int): Number of partitions for the Dask DataFrame.
# #     """

# #     ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
# #     HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

# #     def __init__(self, filename, npartitions=1):
# #         self.filename = filename
# #         self.npartitions = npartitions

# #     def parse_all_scans(self):
# #         """
# #         Read the SPEC file and return a list of dicts with scan data:
# #           - scan_number: zero-padded scan number
# #           - data_number: zero-based, zero-padded data row index
# #           - type: scan type ('ascan' or 'hklscan')
# #           - tth, th, chi, phi: goniometer angles
# #           - h, k, l: reciprocal-lattice coordinates
# #         """
# #         # Read global #O0 ordering
# #         o0_names = []
# #         with open(self.filename) as f:
# #             for line in f:
# #                 if line.startswith('#O0 '):
# #                     o0_names = line[4:].split()
# #                     break
# #         if not o0_names:
# #             raise RuntimeError("Missing global #O0 line in SPEC file")

# #         results = []
# #         current_scan = None
# #         current_type = None
# #         p0_map       = {}
# #         data_idx     = {}
# #         in_data      = False
# #         data_counter = 0

# #         with open(self.filename) as f:
# #             for raw in f:
# #                 line = raw.strip()

# #                 # New scan
# #                 if line.startswith('#S '):
# #                     parts = line.split()
# #                     current_scan = int(parts[1])
# #                     current_type = parts[2] if len(parts) > 2 else ''
# #                     p0_map       = {}
# #                     data_idx.clear()
# #                     in_data      = False
# #                     data_counter = 0
# #                     continue

# #                 # P0 for ascan
# #                 if current_scan is not None and line.startswith('#P0 '):
# #                     vals = [float(x) for x in line[4:].split()]
# #                     p0_map = { name: vals[i] for i,name in enumerate(o0_names) }
# #                     continue

# #                 # Data header
# #                 if current_scan is not None and line.startswith('#L '):
# #                     cols = line[3:].split()
# #                     if current_type.lower() == 'ascan':
# #                         axes = [c for c in self.ASCAN_AXES if c in cols]
# #                         if len(axes) != 1:
# #                             raise RuntimeError(f"Scan {current_scan} (ascan) #L must contain one of {self.ASCAN_AXES}")
# #                         scan_col = axes[0]
# #                         data_idx['scan_col'] = cols.index(scan_col)
# #                         for hk in ('H','K','L'):
# #                             data_idx[hk] = cols.index(hk)
# #                     elif current_type.lower() == 'hklscan':
# #                         for ax in self.HKL_AXES:
# #                             data_idx[ax] = cols.index(ax)
# #                         for hk in ('H','K','L'):
# #                             data_idx[hk] = cols.index(hk)
# #                     else:
# #                         in_data = False
# #                         continue

# #                     in_data = True
# #                     continue

# #                 # Data rows
# #                 if in_data:
# #                     if not line or (line.startswith('#') and not line[1].isdigit()):
# #                         in_data = False
# #                         continue
# #                     parts = line.split()
# #                     if len(parts) < max(data_idx.values()) + 1:
# #                         continue

# #                     rec = {
# #                         'scan_number': f"{current_scan:03d}",
# #                         'data_number': f"{data_counter:03d}",
# #                         'type':        current_type
# #                     }

# #                     if current_type.lower() == 'ascan':
# #                         rec.update({
# #                             'tth': p0_map.get('VTTH'),
# #                             'th':  p0_map.get('VTH'),
# #                             'chi': p0_map.get('Chi'),
# #                             'phi': p0_map.get('Phi'),
# #                             'h':   float(parts[data_idx['H']]),
# #                             'k':   float(parts[data_idx['K']]),
# #                             'l':   float(parts[data_idx['L']]),
# #                         })
# #                         val = float(parts[data_idx['scan_col']])
# #                         if scan_col == 'VTTH': rec['tth'] = val
# #                         elif scan_col == 'VTH': rec['th'] = val
# #                         elif scan_col == 'Phi': rec['phi'] = val
# #                         elif scan_col == 'Chi': rec['chi'] = val
# #                     else:
# #                         rec.update({
# #                             'tth': float(parts[data_idx['VTTH']]),
# #                             'th':  float(parts[data_idx['VTH']]),
# #                             'chi': float(parts[data_idx['Chi']]),
# #                             'phi': float(parts[data_idx['Phi']]),
# #                             'h':   float(parts[data_idx['H']]),
# #                             'k':   float(parts[data_idx['K']]),
# #                             'l':   float(parts[data_idx['L']]),
# #                         })

# #                     results.append(rec)
# #                     data_counter += 1

# #         return results

# #     def to_pandas(self):
# #         """
# #         Convert parsed scan data to a pandas DataFrame.
# #         """
# #         data = self.parse_all_scans()
# #         return pd.DataFrame(data)

# #     def to_dask(self):
# #         """
# #         Convert parsed scan data to a Dask DataFrame.
# #         """
# #         df = self.to_pandas()
# #         return dd.from_pandas(df, npartitions=self.npartitions)


# #!/usr/bin/env python3
# """
# spec_pipeline.py

# Defines ExperimentSetup, Crystal, ScanAngles, and ExperimentData classes to manage experimental
# setup, crystal orientation, scan angles, and provide a unified interface for SPEC-format files.
# """

# import sys
# import numpy as np
# import pandas as pd
# import dask.dataframe as dd

# class ExperimentSetup:
#     """
#     Encapsulates the experimental geometry and detector parameters.
#     """
#     def __init__(self,
#                  distance=781.05,
#                  pitch=0.75,
#                  ycenter=257,
#                  xcenter=515,
#                  xpixels=1030,
#                  ypixels=514,
#                  wavelength=0.283383,
#                  phi=0.0,
#                  theta=15.3069,
#                  dtheta=0.04,
#                  energy=11470.0):
#         self.distance   = distance
#         self.pitch      = pitch
#         self.ycenter    = ycenter
#         self.xcenter    = xcenter
#         self.xpixels    = xpixels
#         self.ypixels    = ypixels
#         self.wavelength = wavelength
#         self.phi        = phi
#         self.theta      = theta
#         self.dtheta     = dtheta
#         self.energy     = energy

#     def __repr__(self):
#         return (
#             f"ExperimentSetup(distance={self.distance} mm, pitch={self.pitch} µm, "
#             f"ycenter={self.ycenter}, xcenter={self.xcenter}, xpixels={self.xpixels}, "
#             f"ypixels={self.ypixels}, wavelength={self.wavelength} Å, phi={self.phi}°, "
#             f"theta={self.theta}°, dtheta={self.dtheta}°, energy={self.energy} eV)"
#         )

# class Crystal:
#     """
#     Holds crystal lattice parameters and orientation matrix (UB).
#     """
#     def __init__(self, params, ub_matrix):
#         self.a, self.b, self.c = params[0:3]
#         self.alpha, self.beta, self.gamma = params[3:6]
#         self.a_hkl, self.b_hkl, self.c_hkl = params[6:9]
#         self.alpha_hkl, self.beta_hkl, self.gamma_hkl = params[9:12]
#         self.H_or0, self.K_or0, self.L_or0 = params[12:15]
#         self.H_or1, self.K_or1, self.L_or1 = params[15:18]
#         (self.u00, self.u01, self.u02,
#          self.u03, self.u04, self.u05) = params[18:24]
#         (self.u10, self.u11, self.u12,
#          self.u13, self.u14, self.u15) = params[24:30]
#         self.lambda0, self.lambda1 = params[30:32]
#         self.u06, self.u16 = params[32:34]
#         ub = np.array(ub_matrix, dtype=float)
#         self.UB = ub.reshape((3,3))

#     @classmethod
#     def from_spec(cls, filename):
#         g1_vals = None
#         g3_vals = None
#         with open(filename) as f:
#             for line in f:
#                 if line.startswith('#G1 '):
#                     g1_vals = [float(x) for x in line.split()[1:]]
#                 elif line.startswith('#G3 '):
#                     g3_vals = [float(x) for x in line.split()[1:]]
#                 if g1_vals is not None and g3_vals is not None:
#                     break
#         if g1_vals is None or g3_vals is None:
#             raise RuntimeError("Missing #G1 or #G3 in SPEC file for Crystal.")
#         return cls(g1_vals, g3_vals)

#     def __repr__(self):
#         return (
#             f"Crystal(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, beta={self.beta}, "
#             f"gamma={self.gamma}, UB=\n{self.UB})"
#         )


# # ...existing code...
# class ScanAngles:
#     ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
#     HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

#     # Modify __init__ to accept a Crystal object
#     def __init__(self, filename, crystal, npartitions=1):
#         self.filename    = filename
#         self.npartitions = npartitions
#         self.crystal     = crystal   # store the Crystal object
#     def parse_all_scans(self):
#     """
#     Read the SPEC file and return a list of dicts with scan data.
#     Each record will include a 'ub' key containing the 3x3 UB matrix
#     read from the "#G3" line within that scan (if present), otherwise None.
#     """
#     o0_names = []
#     with open(self.filename) as f:
#         for line in f:
#             if line.startswith('#O0 '):
#                 o0_names = line[4:].split()
#                 break
#     if not o0_names:
#         raise RuntimeError("Missing global #O0 line in SPEC file")

#     results = []
#     cur_scan = None
#     cur_type = None
#     p0_map   = {}
#     data_idx = {}
#     in_data  = False
#     counter  = 0
#     current_ub = None          # initialize per-scan UB

#     with open(self.filename) as f:
#         for raw in f:
#             line = raw.strip()

#             # New scan: reset state variables.
#             if line.startswith('#S '):
#                 parts = line.split()
#                 cur_scan = int(parts[1])
#                 cur_type = parts[2] if len(parts) > 2 else ''
#                 p0_map.clear()
#                 data_idx.clear()
#                 in_data = False
#                 counter = 0
#                 current_ub = None   # reset per-scan UB for new scan
#                 continue

#             # Look for UB updates within a scan from "#G3" line.
#             if cur_scan is not None and line.startswith('#G3 '):
#                 # Parse UB: assume 9 numbers follow "#G3"
#                 ub_vals = [float(x) for x in line.split()[1:]]
#                 if len(ub_vals) != 9:
#                     raise RuntimeError(f"Scan {cur_scan}: UB line does not have 9 values.")
#                 current_ub = np.array(ub_vals).reshape((3, 3))
#                 continue

#             # Grab fixed motors from #P0 (for ascan)
#             if cur_scan is not None and line.startswith('#P0 '):
#                 vals = [float(x) for x in line.split()[1:]]
#                 p0_map = {name: vals[i] for i, name in enumerate(o0_names)}
#                 continue

#             # Data header: pick columns based on scan type.
#             if cur_scan is not None and line.startswith('#L '):
#                 cols = line.split()[1:]
#                 if cur_type.lower() == 'ascan':
#                     axes = [c for c in self.ASCAN_AXES if c in cols]
#                     if len(axes) != 1:
#                         raise RuntimeError(f"Scan {cur_scan} (ascan): expected one of {self.ASCAN_AXES} in header")
#                     scan_col = axes[0]
#                     data_idx['scan_col'] = cols.index(scan_col)
#                     for hk in ('H', 'K', 'L'):
#                         data_idx[hk] = cols.index(hk)
#                 elif cur_type.lower() == 'hklscan':
#                     for ax in self.HKL_AXES:
#                         data_idx[ax] = cols.index(ax)
#                     for hk in ('H', 'K', 'L'):
#                         data_idx[hk] = cols.index(hk)
#                 else:
#                     in_data = False
#                     continue
#                 in_data = True
#                 continue

#             # Data rows.
#             if in_data:
#                 if not line or (line.startswith('#') and not line[1].isdigit()):
#                     in_data = False
#                     continue
#                 parts = line.split()
#                 if len(parts) < max(data_idx.values()) + 1:
#                     continue
#                 rec = {
#                     'scan_number': f"{cur_scan:03d}",
#                     'data_number': f"{counter:03d}",
#                     'type': cur_type,
#                     # Attach this scan's UB read from "#G3"
#                     'ub': current_ub.copy() if current_ub is not None else None
#                 }
#                 if cur_type.lower() == 'ascan':
#                     rec.update({
#                         'tth': p0_map.get('VTTH'),
#                         'th':  p0_map.get('VTH'),
#                         'chi': p0_map.get('Chi'),
#                         'phi': p0_map.get('Phi'),
#                         'h':   float(parts[data_idx['H']]),
#                         'k':   float(parts[data_idx['K']]),
#                         'l':   float(parts[data_idx['L']])
#                     })
#                     val = float(parts[data_idx['scan_col']])
#                     if scan_col == 'VTTH':
#                         rec['tth'] = val
#                     elif scan_col == 'VTH':
#                         rec['th'] = val
#                     elif scan_col == 'Phi':
#                         rec['phi'] = val
#                     elif scan_col == 'Chi':
#                         rec['chi'] = val
#                 else:  # For hklscan.
#                     rec.update({
#                         'tth': float(parts[data_idx['VTTH']]),
#                         'th':  float(parts[data_idx['VTH']]),
#                         'chi': float(parts[data_idx['Chi']]),
#                         'phi': float(parts[data_idx['Phi']]),
#                         'h':   float(parts[data_idx['H']]),
#                         'k':   float(parts[data_idx['K']]),
#                         'l':   float(parts[data_idx['L']])
#                     })
#                 results.append(rec)
#                 counter += 1
#     # return results
#     # def parse_all_scans(self):
#     #     # copy implementation from previous SpecParser
#     #     # reads scan_number, data_number, type, tth, th, chi, phi, h, k, l
#     #     o0_names = []
#     #     with open(self.filename) as f:
#     #         for line in f:
#     #             if line.startswith('#O0 '):
#     #                 o0_names = line[4:].split()
#     #                 break
#     #     if not o0_names:
#     #         raise RuntimeError("Missing #O0 in SPEC file for ScanAngles.")

#     #     results = []
#     #     cur_scan = None
#     #     cur_type = None
#     #     p0_map   = {}
#     #     data_idx = {}
#     #     in_data  = False
#     #     counter  = 0

#     #     with open(self.filename) as f:
#     #         for raw in f:
#     #             line = raw.strip()
#     #             if line.startswith('#S '):
#     #                 parts = line.split()
#     #                 cur_scan = int(parts[1])
#     #                 cur_type = parts[2] if len(parts)>2 else ''
#     #                 p0_map.clear(); data_idx.clear(); in_data=False; counter=0
#     #                 continue
#     #             if cur_scan and line.startswith('#P0 '):
#     #                 vals = [float(x) for x in line.split()[1:]]
#     #                 p0_map = {name: vals[i] for i, name in enumerate(o0_names)}
#     #                 continue
#     #             if cur_scan and line.startswith('#L '):
#     #                 cols = line.split()[1:]
#     #                 if cur_type.lower()=='ascan':
#     #                     axes = [c for c in self.ASCAN_AXES if c in cols]
#     #                     if not axes:
#     #                         raise RuntimeError(f"Scan {cur_scan}: {self.ASCAN_AXES} not found in header")
#     #                     scan_col = axes[0]
#     #                     data_idx['scan_col'] = cols.index(scan_col)
#     #                     for hk in ('H','K','L'):
#     #                         data_idx[hk] = cols.index(hk)
#     #                 elif cur_type.lower()=='hklscan':
#     #                     for ax in self.HKL_AXES: data_idx[ax] = cols.index(ax)
#     #                     for hk in ('H','K','L'): data_idx[hk] = cols.index(hk)
#     #                 else:
#     #                     in_data=False; continue
#     #                 in_data=True; continue
#     #             if in_data:
#     #                 if not line or (line.startswith('#') and not line[1].isdigit()):
#     #                     in_data=False; continue
#     #                 parts = line.split()
#     #                 if len(parts) < max(data_idx.values())+1: continue
#     #                 rec = {
#     #                     'scan_number': f"{cur_scan:03d}",
#     #                     'data_number': f"{counter:03d}",
#     #                     'type': cur_type,
#     #                     # NEW: attach the UB from the Crystal object
#     #                     'ub': self.crystal.UB.copy()
#     #                 }
#     #                 if cur_type.lower()=='ascan':
#     #                     rec.update({
#     #                         'tth': p0_map.get('VTTH'),
#     #                         'th':  p0_map.get('VTH'),
#     #                         'chi': p0_map.get('Chi'),
#     #                         'phi': p0_map.get('Phi'),
#     #                         'h':   float(parts[data_idx['H']]),
#     #                         'k':   float(parts[data_idx['K']]),
#     #                         'l':   float(parts[data_idx['L']])
#     #                     })
#     #                     val = float(parts[data_idx['scan_col']])
#     #                     if scan_col=='VTTH': rec['tth'] = val
#     #                     elif scan_col=='VTH': rec['th'] = val
#     #                     elif scan_col=='Phi': rec['phi'] = val
#     #                     elif scan_col=='Chi': rec['chi'] = val
#     #                 else:
#     #                     rec.update({
#     #                         'tth': float(parts[data_idx['VTTH']]),
#     #                         'th':  float(parts[data_idx['VTH']]),
#     #                         'chi': float(parts[data_idx['Chi']]),
#     #                         'phi': float(parts[data_idx['Phi']]),
#     #                         'h':   float(parts[data_idx['H']]),
#     #                         'k':   float(parts[data_idx['K']]),
#     #                         'l':   float(parts[data_idx['L']])
#     #                     })
#     #                 results.append(rec)
#     #                 counter += 1
#     #     return results

#     def to_pandas(self):
#         data = self.parse_all_scans()
#         return pd.DataFrame(data)
# # ...existing code...
# class SpecParser:
#     def __init__(self, filename, npartitions=1):
#         self.filename = filename
#         self.setup    = ExperimentSetup()
#         self.crystal  = Crystal.from_spec(filename)
#         # Pass the crystal to ScanAngles so each record can get its UB.
#         self.scans    = ScanAngles(filename, self.crystal, npartitions=npartitions)
#     def to_pandas(self):
#         df = self.scans.to_pandas()
#         return df
#     def to_dask(self):
#         df = self.to_pandas()
#         return dd.from_pandas(df, npartitions=self.npartitions)

# # class ScanAngles:
# #     """
# #     Parses scan data (#S, #P0, #L) from a SPEC file into a DataFrame.
# #     """
# #     ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
# #     HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

# #     def __init__(self, filename, npartitions=1):
# #         self.filename    = filename
# #         self.npartitions = npartitions

# #     def parse_all_scans(self):
# #         # copy implementation from previous SpecParser
# #         # reads scan_number, data_number, type, tth, th, chi, phi, h, k, l
# #         o0_names = []
# #         with open(self.filename) as f:
# #             for line in f:
# #                 if line.startswith('#O0 '):
# #                     o0_names = line[4:].split()
# #                     break
# #         if not o0_names:
# #             raise RuntimeError("Missing #O0 in SPEC file for ScanAngles.")

# #         results = []
# #         cur_scan = None
# #         cur_type = None
# #         p0_map   = {}
# #         data_idx = {}
# #         in_data  = False
# #         counter  = 0

# #         with open(self.filename) as f:
# #             for raw in f:
# #                 line = raw.strip()
# #                 if line.startswith('#S '):
# #                     parts = line.split()
# #                     cur_scan = int(parts[1])
# #                     cur_type = parts[2] if len(parts)>2 else ''
# #                     p0_map.clear(); data_idx.clear(); in_data=False; counter=0
# #                     continue
# #                 if cur_scan and line.startswith('#P0 '):
# #                     vals = [float(x) for x in line.split()[1:]]
# #                     p0_map = {name:vals[i] for i,name in enumerate(o0_names)}
# #                     continue
# #                 if cur_scan and line.startswith('#L '):
# #                     cols = line.split()[1:]
# #                     if cur_type.lower()=='ascan':
# #                         axes=[c for c in self.ASCAN_AXES if c in cols]
# #                         scan_col=axes[0]; data_idx['scan_col']=cols.index(scan_col)
# #                         for hk in ('H','K','L'): data_idx[hk]=cols.index(hk)
# #                     elif cur_type.lower()=='hklscan':
# #                         for ax in self.HKL_AXES: data_idx[ax]=cols.index(ax)
# #                         for hk in ('H','K','L'): data_idx[hk]=cols.index(hk)
# #                     else:
# #                         in_data=False; continue
# #                     in_data=True; continue
# #                 if in_data:
# #                     if not line or (line.startswith('#') and not line[1].isdigit()):
# #                         in_data=False; continue
# #                     parts=line.split()
# #                     if len(parts)<max(data_idx.values())+1: continue
# #                     rec={'scan_number':f"{cur_scan:03d}", 'data_number':f"{counter:03d}", 'type':cur_type}
# #                     if cur_type.lower()=='ascan':
# #                         rec.update({
# #                             'tth':p0_map.get('VTTH'), 'th':p0_map.get('VTH'),
# #                             'chi':p0_map.get('Chi'), 'phi':p0_map.get('Phi'),
# #                             'h':float(parts[data_idx['H']]), 'k':float(parts[data_idx['K']]), 'l':float(parts[data_idx['L']])
# #                         })
# #                         val=float(parts[data_idx['scan_col']])
# #                         if scan_col=='VTTH': rec['tth']=val
# #                         elif scan_col=='VTH': rec['th']=val
# #                         elif scan_col=='Phi': rec['phi']=val
# #                         elif scan_col=='Chi': rec['chi']=val
# #                     else:
# #                         rec.update({
# #                             'tth':float(parts[data_idx['VTTH']]), 'th':float(parts[data_idx['VTH']]),
# #                             'chi':float(parts[data_idx['Chi']]), 'phi':float(parts[data_idx['Phi']]),
# #                             'h':float(parts[data_idx['H']]), 'k':float(parts[data_idx['K']]), 'l':float(parts[data_idx['L']])
# #                         })
# #                     results.append(rec);
# #                     counter+=1
# #         return results

# #     def to_pandas(self):
# #         return pd.DataFrame(self.parse_all_scans())

# #     def to_dask(self):
# #         return dd.from_pandas(self.to_pandas(), npartitions=self.npartitions)

# # class SpecParser:
# #     """
# #     Aggregates ExperimentSetup, Crystal, and ScanAngles for a SPEC file.
# #     """
# #     def __init__(self, filename, npartitions=1):
# #         self.filename = filename
# #         self.setup    = ExperimentSetup()
# #         self.crystal  = Crystal.from_spec(filename)
# #         self.scans    = ScanAngles(filename, npartitions=npartitions)

# #     def to_pandas(self):
# #         df = self.scans.to_pandas()
# #         # you can join with setup/crystal info if desired
# #         return df

# #     def to_dask(self):
# #         return self.scans.to_dask()


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
        # detector geometry
        self.distance = float(distance)
        self.pitch = float(pitch)
        self.ycenter = int(ycenter)
        self.xcenter = int(xcenter)
        self.xpixels = int(xpixels)
        self.ypixels = int(ypixels)

        # beam energy (keV)
        self.energy = float(energy)
        self.energy_keV = float(energy)

        # basic validation
        if self.distance <= 0:
            raise ValueError("ExperimentSetup: 'distance' must be > 0")
        if self.pitch <= 0:
            raise ValueError("ExperimentSetup: 'pitch' must be > 0")
        if self.xpixels <= 0 or self.ypixels <= 0:
            raise ValueError("ExperimentSetup: 'xpixels' and 'ypixels' must be > 0")
        if self.energy_keV <= 0:
            raise ValueError("ExperimentSetup: 'energy' (keV) must be > 0")

        # wavelength handling: allow None, numeric, or numeric-string; fallback to energy
        lam_A: float | None = None
        if wavelength is not None:
            try:
                lam_A = float(wavelength)
            except (TypeError, ValueError):
                lam_A = None

        # if given in meters (small positive), convert to Å
        if lam_A is not None and 0.0 < lam_A < 1e-3:
            lam_A *= 1e10

        # if missing or non-positive, compute from energy
        if lam_A is None or lam_A <= 0.0:
            lam_A = self._energy_keV_to_lambda_A(self.energy_keV)

        if lam_A <= 0.0:
            raise ValueError("ExperimentSetup: computed wavelength is non-positive")
        self.wavelength = lam_A

    @staticmethod
    def _energy_keV_to_lambda_A(E_keV: float) -> float:
        """λ[Å] = 12.398419843320026 / E[keV]."""
        return 12.398419843320026 / float(E_keV)

    # ---------- YAML helpers ----------
    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            # tolerate strings like "1_024" or with spaces
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
        """
        Accept either flat YAML or nested mappings.
        Priority:
          1) data["ExperimentSetup"]
          2) data["experiment"]
          3) data["experiment_setup"]
          4) flat top-level (data itself)
          5) any nested dict that seems to contain necessary keys
        """
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping of keys to values.")

        # common section names
        for key in ("ExperimentSetup", "experiment", "experiment_setup"):
            sec = data.get(key)
            if isinstance(sec, dict):
                return sec

        # flat?
        if any(k in data for k in cls.REQUIRED_KEYS):
            return data

        # last resort: scan nested dicts
        for v in data.values():
            if isinstance(v, dict) and any(k in v for k in cls.REQUIRED_KEYS):
                return v

        # nothing suitable found
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

        # Build a merged view that allows some keys to be at top-level and some inside the section
        merged = {}
        for k in cls.REQUIRED_KEYS + ("wavelength",):
            if k in sec:
                merged[k] = sec[k]
            elif k in doc:
                merged[k] = doc[k]

        # Validate presence
        missing = [k for k in cls.REQUIRED_KEYS if merged.get(k) in (None, "", "None", "null")]
        if missing:
            raise ValueError(f"Missing required keys in YAML: {missing}")

        # Coerce types
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

# class ExperimentSetup:
#     """
#     Load experiment parameters from a YAML file.  Wavelength is optional:
#       • if provided and >1e-3 Å, used directly
#       • if provided in meters (<1e-3), converted to Å
#       • if omitted or non‐positive, computed from energy [Å] = 12.3984193 / E[keV]
#     Required YAML keys: distance, pitch, ycenter, xcenter,
#                         xpixels, ypixels, phi, theta, dtheta, energy
#     Optional key: wavelength
#     """
#     REQUIRED_KEYS = (
#         "distance", "pitch", "ycenter", "xcenter",
#         "xpixels", "ypixels", "energy",
#     )

#     def __init__(
#         self,
#         distance: float,
#         pitch: float,
#         ycenter: int,
#         xcenter: int,
#         xpixels: int,
#         ypixels: int,
#         energy: float,
#         wavelength: float | None = None,
#     ):
#         # detector geometry
#         self.distance = distance
#         self.pitch = pitch
#         self.ycenter = ycenter
#         self.xcenter = xcenter
#         self.xpixels = xpixels
#         self.ypixels = ypixels
#         # beam energy
#         self.energy = energy
#         self.energy_keV = energy

#         # wavelength handling
#         # wavelength handling: allow None, numeric, or numeric‐string; fallback to energy
#         lam_A: float | None = None
#         if wavelength is not None:
#             try:
#                 lam_A = float(wavelength)
#             except (TypeError, ValueError):
#                 lam_A = None

#         # if given in meters (small positive), convert to Å
#         if lam_A is not None and 0.0 < lam_A < 1e-3:
#             lam_A *= 1e10

#         # if missing or non-positive, compute from energy
#         if lam_A is None or lam_A <= 0.0:
#             if self.energy_keV > 0.0:
#                 lam_A = self._energy_keV_to_lambda_A(self.energy_keV)
#             else:
#                 raise ValueError("ExperimentSetup: energy must be > 0 to derive wavelength")

#         if lam_A <= 0.0:
#             raise ValueError("ExperimentSetup: computed wavelength is non-positive")
#         self.wavelength = lam_A
       
#     def _energy_keV_to_lambda_A(self, E_keV: float) -> float:
#         """λ[Å] = 12.398419843320026 / E[keV]."""
#         return 12.398419843320026 / float(E_keV)

#     @classmethod
#     def from_yaml(cls, path: str | Path):
#         p = Path(path)
#         if not p.is_file():
#             raise FileNotFoundError(f"Experiment YAML not found: {p}")
#         with p.open("r", encoding="utf-8") as f:
#             data = yaml.safe_load(f)
#         if not isinstance(data, dict):
#             raise ValueError("Top-level YAML must be a mapping of keys to values.")
#         missing = [k for k in cls.REQUIRED_KEYS if k not in data]
#         if missing:
#             raise ValueError(f"Missing required keys in YAML: {missing!r}")
#         # extract required params + optional wavelength
#         params = {k: data[k] for k in cls.REQUIRED_KEYS}
#         params["wavelength"] = data.get("wavelength", None)
#         return cls(**params)

#     def __repr__(self):
#         return (
#             f"<ExperimentSetup: distance={self.distance} m, pitch={self.pitch} m, "
#             f"xcenter={self.xcenter}, ycenter={self.ycenter}, "
#             f"xpixels={self.xpixels}, ypixels={self.ypixels}, "
#             f"energy={self.energy} eV, wavelength={self.wavelength} Å>"
#         )

# class ExperimentSetup:
#     """Load experiment parameters strictly from a YAML file (no defaults)."""
#     REQUIRED_KEYS = (
#         "distance","pitch","ycenter","xcenter",
#         "xpixels","ypixels","phi","theta","dtheta","energy","wavelength"
#     )

#     def __init__(
#         self,
#         distance: float,
#         pitch: float,
#         ycenter: int,
    #     xcenter: int,
    #     xpixels: int,
    #     ypixels: int,
    #     phi: float,
    #     theta: float,
    #     dtheta: float,
    #     energy: float,
    #     wavelength: float,
    # ):
    #     self.distance = distance
    #     self.pitch = pitch
    #     self.ycenter = ycenter
    #     self.xcenter = xcenter
    #     self.xpixels = xpixels
    #     self.ypixels = ypixels
    #     self.phi = phi
    #     self.theta = theta
    #     self.dtheta = dtheta
    #     self.energy = energy
    #     self.energy_keV = energy * 1e-3
    #     self.wavelength = wavelength

    # @classmethod
    # def from_yaml(cls, path: str | Path):
    #     p = Path(path)
    #     if not p.is_file():
    #         raise FileNotFoundError(f"Experiment YAML not found: {p}")
    #     with p.open("r", encoding="utf-8") as f:
    #         data = yaml.safe_load(f)
    #     if not isinstance(data, dict):
    #         raise ValueError("Top-level YAML must be a mapping.")
    #     missing = [k for k in cls.REQUIRED_KEYS if k not in data]
    #     if missing:
    #         raise ValueError(f"Missing keys in YAML: {missing}")
    #     return cls(**{k: data[k] for k in cls.REQUIRED_KEYS})

    # def __repr__(self):
    #     return (
    #         f"ExperimentSetup(distance={self.distance} m, pitch={self.pitch} m, "
    #         f"ycenter={self.ycenter}, xcenter={self.xcenter}, xpixels={self.xpixels}, "
    #         f"ypixels={self.ypixels}, wavelength={self.wavelength} Å, phi={self.phi}°, "
    #         f"theta={self.theta}°, dtheta={self.dtheta}°, energy={self.energy} eV)"
    #     )

# class ExperimentSetup:
#     """
#     Encapsulates the experimental geometry and detector parameters.
#     """
#     def __init__(self,
#                  distance=781.05*1e-3,  # convert mm to m
#                  pitch=75*1e-6,    # convert µm to m
#                  ycenter=257,
#                  xcenter=515,
#                  xpixels=1030,
#                  ypixels=514,
#                 #  wavelength=0.283383,
#                  phi=0.0,
#                  theta=15.3069,
#                  dtheta=0.04,
#                  energy=11470.0):
#         self.distance   = distance 
#         self.pitch      = pitch 
#         self.ycenter    = ycenter
#         self.xcenter    = xcenter
#         self.xpixels    = xpixels
#         self.ypixels    = ypixels
#         # self.wavelength = wavelength
#         self.phi        = phi
#         self.theta      = theta
#         self.dtheta     = dtheta
#         self.energy_keV     = energy * 1e-3

#     def __repr__(self):
#         return (
#             f"ExperimentSetup(distance={self.distance} m, pitch={self.pitch} m, "
#             f"ycenter={self.ycenter}, xcenter={self.xcenter}, xpixels={self.xpixels}, "
#             f"ypixels={self.ypixels}, wavelength={self.wavelength} Å, phi={self.phi}°, "
#             f"theta={self.theta}°, dtheta={self.dtheta}°, energy={self.energy} eV)"
#         )

class Crystal:
    """
    Holds crystal lattice parameters and orientation matrix (UB).
    """
    def __init__(self, params, ub_matrix):
        # Parse crystal lattice parameters from #G1 line
        self.a, self.b, self.c = params[0:3]
        self.alpha, self.beta, self.gamma = params[3:6]
        self.a_hkl, self.b_hkl, self.c_hkl = params[6:9]
        self.alpha_hkl, self.beta_hkl, self.gamma_hkl = params[9:12]
        self.H_or0, self.K_or0, self.L_or0 = params[12:15]
        self.H_or1, self.K_or1, self.L_or1 = params[15:18]
        (self.u00, self.u01, self.u02,
         self.u03, self.u04, self.u05) = params[18:24]
        (self.u10, self.u11, self.u12,
         self.u13, self.u14, self.u15) = params[24:30]
        self.lambda0, self.lambda1 = params[30:32]
        self.u06, self.u16 = params[32:34]
        # Build UB matrix from the global #G3 (if any)
        ub = np.array(ub_matrix, dtype=float)
        self.UB = ub.reshape((3, 3))

    @classmethod
    def from_spec(cls, filename):
        """
        Parse the first encountered #G1 and #G3 lines from the SPEC file to construct a Crystal.
        """
        g1_vals, g3_vals = None, None
        with open(filename) as f:
            for line in f:
                if line.startswith('#G1 '):
                    g1_vals = [float(x) for x in line.split()[1:]]
                elif line.startswith('#G3 '):
                    g3_vals = [float(x) for x in line.split()[1:]]
                if g1_vals is not None and g3_vals is not None:
                    break
        if g1_vals is None or g3_vals is None:
            raise RuntimeError("Missing #G1 or #G3 in SPEC file for Crystal.")
        return cls(g1_vals, g3_vals)

    def __repr__(self):
        return (
            f"Crystal(a={self.a}, b={self.b}, c={self.c}, alpha={self.alpha}, beta={self.beta}, "
            f"gamma={self.gamma}, UB=\n{self.UB})"
        )

class ScanAngles:
    ASCAN_AXES = ('VTTH', 'VTH', 'Phi', 'Chi')
    HKL_AXES   = ('VTTH', 'VTH', 'Chi', 'Phi')

    # Modified __init__ to also accept a Crystal object (if needed later)
    def __init__(self, filename, crystal, npartitions=1):
        self.filename    = filename
        self.npartitions = npartitions
        self.crystal     = crystal  # store the Crystal object

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
        p0_map   = {}
        data_idx = {}
        in_data  = False
        counter  = 0
        current_ub = None   # per-scan UB

        with open(self.filename) as f:
            for raw in f:
                line = raw.strip()

                # New scan: reset state variables.
                if line.startswith('#S '):
                    parts = line.split()
                    cur_scan = int(parts[1])
                    cur_type = parts[2] if len(parts) > 2 else ''
                    p0_map.clear()
                    data_idx.clear()
                    in_data = False
                    counter = 0
                    current_ub = None   # reset per-scan UB for new scan
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

                # Data header: using "#L" line.
                if cur_scan is not None and line.startswith('#L '):
                    cols = line.split()[1:]
                    if cur_type.lower() == 'ascan':
                        axes = [c for c in self.ASCAN_AXES if c in cols]
                        if len(axes) != 1:
                            raise RuntimeError(f"Scan {cur_scan} (ascan): expected one of {self.ASCAN_AXES} in header")
                        scan_col = axes[0]
                        data_idx['scan_col'] = cols.index(scan_col)
                        for hk in ('H', 'K', 'L'):
                            data_idx[hk] = cols.index(hk)
                    elif cur_type.lower() == 'hklscan':
                        for ax in self.HKL_AXES:
                            data_idx[ax] = cols.index(ax)
                        for hk in ('H', 'K', 'L'):
                            data_idx[hk] = cols.index(hk)
                    else:
                        in_data = False
                        continue
                    in_data = True
                    continue

                # Data rows.
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
                        # Attach this scan's UB read from "#G3"
                        'ub': current_ub.copy() if current_ub is not None else None
                    }
                    if cur_type.lower() == 'ascan':
                        rec.update({
                            'tth': p0_map.get('VTTH'),
                            'th':  p0_map.get('VTH'),
                            'chi': p0_map.get('Chi'),
                            'phi': p0_map.get('Phi'),
                            'h':   float(parts[data_idx['H']]),
                            'k':   float(parts[data_idx['K']]),
                            'l':   float(parts[data_idx['L']])
                        })
                        val = float(parts[data_idx['scan_col']])
                        if scan_col == 'VTTH':
                            rec['tth'] = val
                        elif scan_col == 'VTH':
                            rec['th'] = val
                        elif scan_col == 'Phi':
                            rec['phi'] = val
                        elif scan_col == 'Chi':
                            rec['chi'] = val
                    else:  # For hklscan.
                        rec.update({
                            'tth': float(parts[data_idx['VTTH']]),
                            'th':  float(parts[data_idx['VTH']]),
                            'chi': float(parts[data_idx['Chi']]),
                            'phi': float(parts[data_idx['Phi']]),
                            'h':   float(parts[data_idx['H']]),
                            'k':   float(parts[data_idx['K']]),
                            'l':   float(parts[data_idx['L']])
                        })
                    results.append(rec)
                    counter += 1
        return results

    def to_pandas(self):
        data = self.parse_all_scans()
        return pd.DataFrame(data)

    def to_dask(self):
        return dd.from_pandas(self.to_pandas(), npartitions=self.npartitions)

class SpecParser:
    """
    Aggregates ExperimentSetup, Crystal, and ScanAngles for a SPEC file.
    """
    def __init__(self, filename: str, setup_yaml: str, npartitions: int = 1):
        self.filename = filename
        self.setup = ExperimentSetup.from_yaml(setup_yaml)
        self.crystal = Crystal.from_spec(filename)
        self.scans = ScanAngles(filename, self.crystal, npartitions=npartitions)

    # def __init__(self, filename, npartitions=1):
    #     self.filename = filename
    #     self.setup    = ExperimentSetup()
    #     # For the global Crystal, use the first #G1 and #G3 found in the file.
    #     self.crystal  = Crystal.from_spec(filename)
    #     # Pass the Crystal to ScanAngles so each record can get its per-scan UB (from #G3).
    #     self.scans    = ScanAngles(filename, self.crystal, npartitions=npartitions)

    def to_pandas(self):
        df = self.scans.to_pandas()
        return df

    def to_dask(self):
        return self.scans.to_dask()
from rsm3d.rsm3d import RSMBuilder  # <-- 4-circle xrayutilities builder
spec_file = '/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/setup_6oct23'
tiff_dir  = '/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/data_6oct23_tiff'
scan_list = (17,18)     # any list/tuple of scan numbers
out_vtr   = '/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/rsm_hkl.vtr'    # output file

# Build with 4-circle (ZXZ: φ(Z) → χ(X) → ω(Z))
builder = RSMBuilder(
    spec_file, tiff_dir,
    selected_scans=scan_list,
    ub_includes_2pi=False,        # set False if your UB is "no-2π"
    center_is_one_based=False,   # True if SPEC xcenter/ycenter are 1-based
    fourc_mode="ZXZ", )


print("buidler made")
# Compute per-pixel Q & HKL
Q_samp, hkl, intensity = builder.compute_full()
print("buidler made Q and hkl")

# Qc, Hc, Ic = builder.crop_by_positions(y_bound=(220, 510), x_bound=(380, 610))
import tifffile
import numpy as np
tifffile.imwrite('/Users/xiaogangyang/BNL.GOV Dropbox/Xiaogang Yang/isr_rsm3d/intensity.tiff', intensity.astype(np.float32))
# Regrid with xrayutilities gridder (mean or sum)
grid, (xax, yax, zax) = builder.regrid_xu(
    space="hkl",                 # "hkl" or "q"
    grid_shape=(200, 200, 200),  # adjust to taste / memory
    ranges=None,                 # or ((xmin,xmax),(ymin,ymax),(zmin,zmax)) to lock axes
    fuzzy=False,                 # True → FuzzyGridder3D; add width=... for footprint
    normalize="mean",            # "mean" or "sum"
    stream=True                  # frame-by-frame accumulation (RAM friendly)
)
grid.shape, len(xax), len(yax), len(zax)
print("regridding done")
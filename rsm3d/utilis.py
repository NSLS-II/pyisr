import numpy as np
from pathlib import Path
import tifffile

def remove_dead_pixels_in_dir(
    tiff_dir: str,
    output_dir: str | None = None,
    threshold: float = 0.0,
    threshold_fraction: float = 1.0,
    neighborhood_size: int = 3,
    min_valid_fraction: float = 0.5,
):
    """
    Improved dead‐pixel removal in a directory of TIFF frames.

    A pixel is flagged as 'dead' if it is ≤ threshold in at least
    `threshold_fraction` fraction of all frames.  Each dead pixel in
    each frame is then replaced by the median of its neighboring
    non‐dead pixels (within a square window of side `neighborhood_size`).

    The output frames preserve their original dtype and value range.

    Parameters
    ----------
    tiff_dir : str
        Directory containing .tif / .tiff frames.
    output_dir : str | None
        If given, corrected frames are saved here (same filenames).
        Otherwise overwrite originals.
    threshold : float
        Absolute intensity below or equal to flag pixel as dead.
    threshold_fraction : float
        Fraction [0–1] of frames that must be ≤ threshold to mark pixel dead.
    neighborhood_size : int
        Side length of square neighborhood for median replacement (odd).
    min_valid_fraction : float
        Minimum fraction [0–1] of non‐dead neighbors required to compute median.
        If not met, dead pixel is left at `threshold`.

    Returns
    -------
    dead_mask : ndarray, shape (H, W)
        Boolean mask of pixels treated as dead.
    """
    # collect files
    tiff_dir = Path(tiff_dir)
    files = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {tiff_dir}")

    # load stack as float32, remember original dtypes
    stack = []
    orig_dtypes = []
    for p in files:
        img_orig = tifffile.imread(p)
        orig_dtypes.append(img_orig.dtype)
        stack.append(img_orig.astype(np.float32))
    stack = np.stack(stack, axis=0)  # (N_frames, H, W)

    # identify dead pixels across frames
    frac_below = np.mean(stack <= threshold, axis=0)
    dead_mask = frac_below >= threshold_fraction

    # prepare output dir
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # parameters for neighborhood window
    Nf, H, W = stack.shape
    r = neighborhood_size // 2
    window_area = (2 * r + 1) ** 2
    min_valid = max(1, int(np.ceil(window_area * min_valid_fraction)))

    # process each frame
    for idx, p in enumerate(files):
        img = stack[idx]
        img_fixed = img.copy()

        # replace each dead pixel
        ys, xs = np.nonzero(dead_mask)
        for y, x in zip(ys, xs):
            i0, i1 = max(0, y - r), min(H, y + r + 1)
            j0, j1 = max(0, x - r), min(W, x + r + 1)
            neigh = img[i0:i1, j0:j1]
            neigh_mask = dead_mask[i0:i1, j0:j1]
            valid = neigh[~neigh_mask]
            if valid.size >= min_valid:
                img_fixed[y, x] = np.median(valid)
            else:
                img_fixed[y, x] = threshold

        # restore original dtype & clamp to valid range
        dtype_cur = orig_dtypes[idx]
        if np.issubdtype(dtype_cur, np.integer):
            info = np.iinfo(dtype_cur)
        else:
            info = np.finfo(dtype_cur)
        img_out = np.clip(img_fixed, info.min, info.max).astype(dtype_cur)

        # write out
        target = (out_dir / p.name) if out_dir else p
        tifffile.imwrite(target, img_out)

    return dead_mask
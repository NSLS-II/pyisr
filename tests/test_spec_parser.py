import os
import tempfile

from rsm3d.spec_parser import ScanAngles

SPEC_WITH_BARE_HASH = """#O0 VTTH VTH Chi Phi
#S 1 hklscan  1 1 1 1 1 1 10
#G3 1 0 0 0 1 0 0 0 1
#L VTTH VTH Chi Phi H K L Epoch
10.0 5.0 0.0 0.0 1.0 0.0 0.0 100
#
11.0 5.0 0.0 0.0 1.1 0.0 0.0 101
"""


def _parse(text):
    with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return ScanAngles(path, crystal=None).parse_all_scans()
    finally:
        os.unlink(path)


def test_bare_hash_line_does_not_crash():
    # A lone '#' inside the data block must end the block, not raise IndexError.
    recs = _parse(SPEC_WITH_BARE_HASH)
    assert len(recs) == 1
    assert recs[0]["h"] == 1.0


def test_normal_data_block_still_parses():
    text = SPEC_WITH_BARE_HASH.replace("#\n", "")
    recs = _parse(text)
    assert len(recs) == 2

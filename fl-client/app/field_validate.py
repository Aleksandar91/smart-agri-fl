"""Load a Flower global checkpoint in parameter order.

Used by evaluate_global.py. Scored PV-19 jobs store weights as
arr_0, arr_1, ... in the same order as the client state_dict.
"""

from pathlib import Path
from typing import List


def load_checkpoint_ordered(npz_path: Path) -> List:
    """Load a Flower-saved .npz in the same arr_0, arr_1, ... order as get_weights()."""
    import numpy as np  # type: ignore

    data = np.load(str(npz_path))
    names = sorted(data.files, key=lambda n: int(n.split("_")[1]))
    return [data[n] for n in names]

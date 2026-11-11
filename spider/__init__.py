"""Learning spider: one continuing Cadence brain wired to the sixteen joints of an eight-legged
body, learning to move forward in a simulated world from the world's reward alone.

The brain never resets. It is composed once, lives through ``Brain.live``, and is saved and
reloaded as the same brain, with its pending outcome, between runs of the server.
"""

import os as _os

# a small dense brain settles faster on one BLAS thread
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_name, "1")

from .body import DT, JOINTS, LEGS, Spider
from .senses import INPUTS, input_names, observe

__all__ = ["DT", "INPUTS", "JOINTS", "LEGS", "Spider", "input_names", "observe"]

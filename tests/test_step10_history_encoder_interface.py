"""Step 10 (new chain) gate test for history encoder interface."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.interfaces.history_encoder import HistoryEncoderModule


def main() -> None:
    # ABC contract must not be directly instantiable.
    try:
        HistoryEncoderModule()
        raise AssertionError("HistoryEncoderModule ABC should not be instantiable.")
    except TypeError:
        pass

    print("Step 10 gate passed: HistoryEncoderModule interface OK")


if __name__ == "__main__":
    main()

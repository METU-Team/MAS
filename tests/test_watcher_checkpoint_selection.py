"""Unit test for watcher checkpoint selection helper."""

import os
import sys
import tempfile
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from cola_framework.watchers.policy_watcher import find_latest_checkpoint


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        old_path = os.path.join(tmp_dir, "old_model.pth")
        new_path = os.path.join(tmp_dir, "new_model.pth")
        txt_path = os.path.join(tmp_dir, "ignore.txt")

        with open(old_path, "w", encoding="utf-8") as f:
            f.write("old")
        time.sleep(1)
        with open(new_path, "w", encoding="utf-8") as f:
            f.write("new")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("ignore")

        latest = find_latest_checkpoint(tmp_dir)
        assert latest == new_path

    print("Watcher helper test passed: latest checkpoint selection OK")


if __name__ == "__main__":
    main()

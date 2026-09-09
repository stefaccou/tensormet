# main.py
import os
import sys

from tensormet.parsing import parse_vector_run_config
from tensormet.launch import launch_vector_creation


if __name__ == "__main__":
    cfg = parse_vector_run_config()
    print(cfg)
    print(cfg.output_dir())
    summary = launch_vector_creation(cfg)
    print(summary)

    # Skip interpreter teardown: a leftover HF `datasets` retry thread can
    # SIGABRT during normal exit after an early break from streaming.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
# 4_config_based_launch.py
import sys, pprint, time as _t
_t0 = _t.time()
def _ts():
    elapsed = _t.time() - _t0
    return f"[{_t.strftime('%H:%M:%S')} +{elapsed:6.1f}s]"

print(f"{_ts()} Python started, importing tensormet.parsing...", flush=True)
from tensormet.parsing import parse_run_config
print(f"{_ts()} tensormet.parsing ok, importing tensormet.launch...", flush=True)
from tensormet.launch import launch_nnt_decomposition
print(f"{_ts()} tensormet.launch ok, importing tensormet.utils...", flush=True)
from tensormet.utils import select_gpu
print(f"{_ts()} all top-level imports done", flush=True)

if __name__ == "__main__":
    print(f"{_ts()} Starting python script")
    cfg = parse_run_config()
    pprint.pprint(cfg)
    print(f"{_ts()} run_id: {cfg.run_id()}")
    tucker = launch_nnt_decomposition(cfg)

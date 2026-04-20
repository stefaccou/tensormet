# 4_config_based_launch.py
from tensormet.parsing import parse_run_config
from tensormet.launch import launch_nnt_decomposition
from tensormet.utils import select_gpu

if __name__ == "__main__":
    print("Starting python script")
    # device = select_gpu()
    cfg = parse_run_config()
    print(cfg)
    print("run_id:", cfg.run_id())
    # print("artifacts:", cfg.artifact_paths())
    tucker = launch_nnt_decomposition(cfg)
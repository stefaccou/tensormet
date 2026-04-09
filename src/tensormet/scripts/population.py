# 4_config_based_launch.py
from tensormet.parsing import parse_population_run_config
from tensormet.launch import launch_tensor_population

if __name__ == "__main__":
    cfg = parse_population_run_config()
    print("passed config:", cfg)
    pop = launch_tensor_population(cfg)
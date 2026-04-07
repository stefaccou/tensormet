# main.py
from tensormet.parsing import parse_vector_run_config
from tensormet.launch import launch_vector_creation


if __name__ == "__main__":
    cfg = parse_vector_run_config()
    print(cfg)
    print(cfg.output_dir())
    summary = launch_vector_creation(cfg)
    print(summary)
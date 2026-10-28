"""Load a dataset config (config/<name>.py or a path) for steps 1-3.

    from config.load_config import load_dataset_config      # python
    eval "$(python config/load_config.py vlsplat_render_scannetpp --shell)"  # bash: cfg_<key> variables from the DATASET section
"""
import importlib.util
import os
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_NAME = "vlsplat_render_scannetpp"


def dataset_config_name(argv=None):
    """--dataset-config <name> on the command line, else $DATASET_CONFIG, else vlsplat_render_scannetpp."""
    argv = sys.argv if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--dataset-config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--dataset-config="):
            return a.split("=", 1)[1]
    return os.environ.get("DATASET_CONFIG", DEFAULT_NAME)


def load_dataset_config(name=None):
    name = name or dataset_config_name()
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"dataset config not found: {path}")
    spec = importlib.util.spec_from_file_location(f"dataset_config_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"DATASET": dict(mod.DATASET), "GRAPH": dict(mod.GRAPH), "REFINE": dict(mod.REFINE)}


def _shell_value(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return "'" + str(v).replace("'", "'\\''") + "'"


if __name__ == "__main__":
    cfg = load_dataset_config(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None)
    if "--shell" in sys.argv:
        d = cfg["DATASET"]
        for k, v in d.items():
            print(f"cfg_{k}={_shell_value(v)}")
        # training-mode flags, so the shell scripts need no conditionals
        train = (["--train_split", "--eval"] if d["train_test_split"] else []) + (
            ["--fixed-gaussian", "--no-densify"] if d["fixed_gaussian"] else [])
        print(f"cfg_train_args={_shell_value(' '.join(train))}")
        print(f"cfg_step3_args={_shell_value('--unified-lift' if d['unified_lift'] else '')}")

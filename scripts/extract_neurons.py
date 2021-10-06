import importlib
import pandas

read_cfg = importlib.import_module("read_config")


def main(fn_cfg):
    cfg = read_cfg.read(fn_cfg)
    return cfg


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1]))
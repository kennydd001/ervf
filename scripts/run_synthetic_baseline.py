import argparse

import torch

from moe_lab.reporting import envelope, write_json
from moe_lab.synthetic import SyntheticConfig, run_synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = run_synthetic(SyntheticConfig(tokens=args.tokens), device)
    path = write_json(
        f"synthetic_{device.type}.json", envelope("synthetic_moe_baseline", result)
    )
    print(path)
    print(result["baselines"])

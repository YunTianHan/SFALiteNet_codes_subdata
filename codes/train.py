"""从零初始化训练五个固定种子，默认不访问测试集指标。"""

import argparse
import json

from sfa_litenet.config import load_config, resolve_path
from sfa_litenet.data import load_manifest
from sfa_litenet.engine import summarize_runs, train_run
from sfa_litenet.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--seed", type=int, default=None)
    selection.add_argument("--all-seeds", action="store_true", help="以相同数据划分依次训练配置中的种子")
    parser.add_argument("--resume", default=None, help="继续单种子训练，必须为 last.pt")
    args = parser.parse_args()
    if args.resume and args.all_seeds:
        parser.error("--resume 仅支持单种子")
    config = load_config(args.config)
    seed = args.seed
    if args.resume and seed is None:
        seed = load_checkpoint(resolve_path(args.resume))["seed"]
    seeds = config["training"]["seeds"] if args.all_seeds else [seed if seed is not None else config["training"]["seeds"][0]]
    if any(not 0 <= value < 2**32 for value in seeds):
        parser.error("随机种子必须位于 [0,2^32)")
    manifest = load_manifest(config["data"])
    results = [train_run(config, value, manifest=manifest, resume=args.resume) for value in seeds]
    if args.all_seeds:
        output = resolve_path(config["project"]["output_dir"])/config["project"]["experiment"]/"validation_summary.json"
        summary = summarize_runs(results, output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

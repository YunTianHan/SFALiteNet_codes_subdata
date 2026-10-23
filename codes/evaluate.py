"""在独立验证/测试集合计算逐图指标或五种子均值及标准差。"""

import argparse
import json

from sfa_litenet.config import load_config, resolve_path
from sfa_litenet.engine import evaluate_checkpoint, run_directory, summarize_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", help="本项目 best.pt 或 last.pt")
    selection.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default=None, help="单检查点评估输出目录")
    args = parser.parse_args()
    if args.output and args.all_seeds:
        parser.error("多种子评估使用各自运行目录，不能指定 --output")
    config = load_config(args.config)
    if args.all_seeds:
        results = [evaluate_checkpoint(config, run_directory(config, seed)/"best.pt", args.split)
                   for seed in config["training"]["seeds"]]
        signatures = {row["manifest_signature"] for row in results}
        if len(signatures) != 1:
            raise ValueError("不同运行使用了不同数据，不能汇总比较")
        output = resolve_path(config["project"]["output_dir"])/config["project"]["experiment"]/f"{args.split}_summary.json"
        result = summarize_runs(results, output)
    else:
        output = resolve_path(args.output) if args.output else None
        result = evaluate_checkpoint(config, args.checkpoint, args.split, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

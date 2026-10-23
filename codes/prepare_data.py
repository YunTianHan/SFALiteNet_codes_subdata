"""核对真实数据并在独立输出目录生成可复现的分组清单。"""

import argparse
import json

from sfa_litenet.config import load_config
from sfa_litenet.data import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="用户 YAML，路径相对于代码根目录")
    parser.add_argument("--overwrite", action="store_true", help="明确要求重新生成清单；已有模型仍保留原划分快照")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_manifest(config["data"], overwrite=args.overwrite)
    print(json.dumps({"counts": manifest["counts"], "grouping": manifest["provenance"]["mode"],
                      "signature": manifest["signature"]}, ensure_ascii=False, indent=2))
    if manifest["provenance"]["mode"] == "consecutive_blocks":
        print("注意：当前缺少真实来源元数据；编号分组不能证明原图或采集区段独立，详见 README。")


if __name__ == "__main__":
    main()

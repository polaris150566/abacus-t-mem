
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tmpdb run lists for batch_test.")
    parser.add_argument("--all-pdb-dir", required=True, help="Directory with raw pdb files.")
    parser.add_argument("--cluster-dict", required=True, help="Path to cluster_dict.npy.")
    parser.add_argument("--cluster-centers-txt", required=True, help="Path to cluster_centers.txt.")
    parser.add_argument(
        "--selection-mode",
        choices=["values", "keys", "centers_txt"],
        default="values",
        help="How to choose pdb ids from the clustering files.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--pdb-path-list", required=True, help="Output txt with one pdb path per line.")
    parser.add_argument("--target-id-list", required=True, help="Output txt with one target id per line.")
    return parser.parse_args()


def load_selected_ids(cluster_dict_path, cluster_centers_txt, selection_mode):
    cluster_dict = np.load(cluster_dict_path, allow_pickle=True).item()

    if selection_mode == "values":
        return sorted({str(item).lower() for values in cluster_dict.values() for item in values})

    if selection_mode == "keys":
        return sorted({key.split(",")[0].split("_")[0].lower() for key in cluster_dict})

    selected = []
    lines = cluster_centers_txt.read_text(encoding="utf-8").splitlines()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        selected.append(text.split(",")[0].split("_")[0].lower())
    return list(dict.fromkeys(selected))


def build_path_map(all_pdb_dir):
    path_map = {}
    for path in sorted(all_pdb_dir.glob("*.pdb")):
        path_map[path.stem.lower()] = str(path.resolve())
    return path_map


def main():
    args = parse_args()

    all_pdb_dir = Path(args.all_pdb_dir)
    cluster_dict_path = Path(args.cluster_dict)
    cluster_centers_txt = Path(args.cluster_centers_txt)
    pdb_path_list = Path(args.pdb_path_list)
    target_id_list = Path(args.target_id_list)

    selected_ids = load_selected_ids(cluster_dict_path, cluster_centers_txt, args.selection_mode)
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    path_map = build_path_map(all_pdb_dir)
    missing = [pdb_id for pdb_id in selected_ids if pdb_id not in path_map]
    if missing:
        raise SystemExit(f"missing pdb files for {len(missing)} ids, sample={missing[:10]}")

    pdb_paths = [path_map[pdb_id] for pdb_id in selected_ids]
    target_ids = [Path(path).stem for path in pdb_paths]

    pdb_path_list.parent.mkdir(parents=True, exist_ok=True)
    target_id_list.parent.mkdir(parents=True, exist_ok=True)
    pdb_path_list.write_text("\n".join(pdb_paths) + "\n", encoding="utf-8")
    target_id_list.write_text("\n".join(target_ids) + "\n", encoding="utf-8")

    print(f"prepared {len(target_ids)} targets")
    print(f"selection_mode: {args.selection_mode}")
    print(f"pdb path list: {pdb_path_list}")
    print(f"target id list: {target_id_list}")
    print(f"first targets: {target_ids[:10]}")


if __name__ == "__main__":
    main()

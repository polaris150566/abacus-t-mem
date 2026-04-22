"""
用法: python3 calc_weighted_recovery.py <recovery_stats.csv或包含多个checkpoint的目录>
"""
import csv, sys
from pathlib import Path

WEIGHT_PATH = '/home/chenty/public_data/tmpdb_after_202505/data_storage/cluster_inverse_size_weights.csv'
WEIGHT_VALUE_FIELDS = ['inverse_cluster_weight', 'weight']
KEY_FIELDS = ['pdbid', 'pdbname', 'name', 'id']


def normalize_pdbname(name):
    return str(name).strip().lower()[:4]


def load_weights(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        key_field = next((x for x in KEY_FIELDS if x in fields), None)
        weight_field = next((x for x in WEIGHT_VALUE_FIELDS if x in fields), None)
        if key_field is None or weight_field is None:
            raise SystemExit(f'unsupported weight file columns: {fields}')
        weights = {}
        for row in reader:
            weights[normalize_pdbname(row[key_field])] = float(row[weight_field])
    return weights


def calc_weighted(csv_path, weights):
    wsum, wrecov, n_match, n_total = 0.0, 0.0, 0, 0
    all_recov = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pdb = normalize_pdbname(row['pdbname'])
            recov = float(row['mean_recovery'])
            all_recov.append(recov)
            n_total += 1
            if pdb in weights:
                wsum += weights[pdb]
                wrecov += weights[pdb] * recov
                n_match += 1
    wm = wrecov / wsum if wsum > 0 else 0.0
    um = sum(all_recov) / len(all_recov) if all_recov else 0.0
    return wm, um, n_match, n_total


def main():
    if len(sys.argv) < 2:
        print('用法: python3 calc_weighted_recovery.py <csv_path 或 dir>')
        sys.exit(1)

    weights = load_weights(WEIGHT_PATH)
    target = Path(sys.argv[1])

    if target.is_file():
        csv_files = [(target.parent.name, target)]
    else:
        csv_files = sorted((p.parent.name, p) for p in target.rglob('recovery_stats.csv'))

    if not csv_files:
        print(f'未找到 recovery_stats.csv in {target}')
        sys.exit(1)

    print('{:<30} {:>12} {:>14} {:>10}'.format('name', 'weighted', 'unweighted', 'matched'))
    print('-' * 70)
    for name, path in csv_files:
        wm, um, nm, nt = calc_weighted(path, weights)
        print('{:<30} {:>12.4f} {:>14.4f} {:>5}/{}'.format(name, wm, um, nm, nt))


if __name__ == '__main__':
    main()

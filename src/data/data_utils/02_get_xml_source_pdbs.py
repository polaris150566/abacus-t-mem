from pathlib import Path
from tqdm import tqdm
import joblib
from protein_utils.pdb_parser import Pdb_processer

def batch_get_pdbs_without_hetatms(input_dir, output_dir, n_jobs = 8):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ent_files = list(input_dir.glob('*.ent'))
    joblib.Parallel(n_jobs=n_jobs, backend='loky')( joblib.delayed(Pdb_processer.utils.copy_pdb_file_without_hetatms)(f, output_dir)
        for f in tqdm(ent_files, desc='copying...')
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove HETATM from PDB .ent files')
    parser.add_argument('input_dir',  help='directory containing *.ent files')
    parser.add_argument('output_dir', help='directory to save cleaned files')
    parser.add_argument('-j', '--jobs', type=int, default=8, help='number of parallel workers (default: 8)')
    args = parser.parse_args()

    batch_get_pdbs_without_hetatms(
        input_dir  = args.input_dir,
        output_dir = args.output_dir,
        n_jobs     = args.jobs
    )
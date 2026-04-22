import numpy as np
from pdbsingle import PDBinstance
from distogram import Distogram
def main():
    pdbinstance = PDBinstance('./1f4w_H_L_.pdb')
    distogram = Distogram(pdbinstance.atom3, num_bins=31, min_dist=2.0, max_dist=20.0)
    distogram.forward()
    distogram.plot_distogram()
if __name__ == "__main__":
    main()
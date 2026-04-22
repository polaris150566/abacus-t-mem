
npydir=/home/chenty/public_data/beta2AR/abacust_design_result/npys
pdb_list=/home/chenty/abacust_mem/src/inference/pdb_list.txt  # txt file with one pdb path per line
root_dir=/home/chenty/abacust_mem/src/inference/pdbs          # directory for pdbtm json files
save_dir=/home/chenty/public_data/beta2AR/abacust_design_result/seqs
batchsize=10
device=3
ckpt=/home/chenty/abacust_mem/src/experiments/abacust_mem_raw/checkpoint/checkpoint_best_noise650M.pt

mkdir -p $npydir $save_dir

find $npydir -type f -name '*.npy' -delete

echo "Running structure preprocess"
python3 /home/chenty/abacust_mem/src/data/data_utils/make_feature_from_pdb.py \
    --pdb_list $pdb_list \
    --out_dir $npydir \
    --protein_chain A \
    --mode inference

echo "Running design"
CUDA_VISIBLE_DEVICES=$device python3 /home/chenty/abacust_mem/src/inference/seq_design_with_lig_raw_pdb.py \
    --npy_dir $npydir/all_npy \
    --save_dir $save_dir \
    --iter_num 10 \
    --temperature 0.1 \
    --suffix esm_refined_beta2AR_active \
    --checkpoint $ckpt \
    --consider_lig \
    --embed_unimol_reprs \
    --nar \
    --esm_refinement \
    --root_dir $root_dir \
    --batchsize $batchsize \
    --PLM_selfcond 1 \
    --PLM_param_dir /database/lyf_database/pretrain_lm/esm/param \
    --selfcondPLM 650M \
    --augment_eps 0.2 \
    --mask_mode aatype_nll \
    --write_allatom_model \
    --tm_raw raw \
    --max_lig_num 3

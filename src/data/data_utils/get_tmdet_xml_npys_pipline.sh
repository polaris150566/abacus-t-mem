root_dir=/home/chenty/public_data/abacust_mem_data/data_storage
tmdet_dir="$root_dir/tmdet_result"
# mkdir -p "$tmdet_dir"

# rm -f "$tmdet_dir/input"

# # 复制“目录内所有文件”而不带外层目录
# # rsync -av --progress "$root_dir/assembled_pdbs/" "$tmdet_dir/input/"
# python 02_get_xml_source_pdbs.py "$root_dir/assembled_pdbs" "$tmdet_dir/input" -j 32


# python /home/chenty/Tmdet/data_utils/run_tmdet/batch_tmdet_prediction.py "$tmdet_dir" --n_jobs 16

INPUT_PATH="$tmdet_dir/xml"
JSON_PATH="$tmdet_dir/jsons"
NPY_PATH="$tmdet_dir/npys"
N_JOBS=32

# 执行Python脚本
python 03_xml2npy.py \
    --input "$INPUT_PATH" \
    --json "$JSON_PATH" \
    --npy "$NPY_PATH" \
    --jobs "$N_JOBS"
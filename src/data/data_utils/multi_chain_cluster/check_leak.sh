#!/usr/bin/env bash
# 用法：bash check_leak.sh query.fasta target.fasta
set -e
rm -rf leak_check
QUERY=$1
TARGET=$2
OUT=leak_check
mkdir -p $OUT

# 1. 建库
mmseqs createdb $TARGET $OUT/target_db
mmseqs createdb $QUERY  $OUT/query_db

# 2. 搜索
mmseqs search $OUT/query_db $OUT/target_db $OUT/aln_db $OUT/tmp \
       --min-seq-id 0.3 -c 0.8 --cov-mode 1 --max-seqs 1000

# 3. 生成 hits 列表（只有 query 名）
mmseqs convertalis $OUT/query_db $OUT/target_db $OUT/aln_db $OUT/hits.tsv \
       --format-output query

# 4. 得出报告
echo "query_id,leak_status" > $OUT/leak_report.csv
# 先收集所有命中
cut -f1 $OUT/hits.tsv | sort -u > $OUT/leaked.lst
# 再逐条 query 判断
grep '^>' $QUERY | cut -d' ' -f1 | tr -d '>' | \
while read q; do
    if grep -qx "$q" $OUT/leaked.lst; then
        echo "$q,leaked"
    else
        echo "$q,safe"
    fi
done >> $OUT/leak_report.csv
total=$(grep -c '^>' "$QUERY")
leaked=$(wc -l < "$OUT/leaked.lst")
safe=$((total - leaked))
echo "===== 统计 ====="
echo "total_query,$total"
echo "leaked,$leaked"
echo "safe,$safe"
echo "Done! 报告：$OUT/leak_report.csv"
# 6. 按 PDBID（前4字符）二次汇总
awk -F, 'NR>1 && $2=="leaked" {pdbid=substr($1,1,4); leak[pdbid]=1}
         END{for(id in leak) print id,",leaked"}' "$OUT/leak_report.csv" \
> "$OUT/pdb_leak_summary.csv"

# 如果想同时列出 safe 的 PDBID 也行（全量）
awk -F, 'NR>1 {pdbid=substr($1,1,4);
               if($2=="leaked") leak[pdbid]=1; else safe[pdbid]=1}
         END{
           for(id in leak)  print id,",leaked";
           for(id in safe)  if(!(id in leak)) print id,",safe"
         }' "$OUT/leak_report.csv" > "$OUT/pdb_leak_summary.csv"

# 最终计数
leak_pdb=$(awk -F, '$2=="leaked"' "$OUT/pdb_leak_summary.csv" | wc -l)
safe_pdb=$(awk -F, '$2=="safe"'  "$OUT/pdb_leak_summary.csv" | wc -l)
echo "===== PDB 级别汇总 ====="
echo "leaked_pdb,$leak_pdb"
echo "safe_pdb,$safe_pdb"
echo "PDB 汇总文件：$OUT/pdb_leak_summary.csv"
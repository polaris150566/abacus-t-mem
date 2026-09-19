#!/bin/bash
# 等一张 >=20GB 空闲且连续 3 分钟稳定的卡, 再启动 afdb_1_1_tmw15 (全量微调, 实测峰值分配 ~16GB)
LOG=/home/chenty/abacust_mem/src/scripts/log/wait_gpu_afdb_1_1_tmw15.log
SCRIPT=/home/chenty/abacust_mem/src/scripts/train_afdb_1_1_tmw15_b36_t0.sh
mkdir -p "$(dirname $LOG)"
echo "[$(date)] watcher started: threshold 20480MiB, 3 consecutive polls, deadline 12h" >> $LOG
DEADLINE=$(( $(date +%s) + 12*3600 ))
prev=-1; cnt=0
while [ $(date +%s) -lt $DEADLINE ]; do
  if pgrep -f "train_afdb_1_1_tmw15_b36_t0.sh" > /dev/null; then
    echo "[$(date)] already running; watcher exit" >> $LOG; exit 0
  fi
  INFO=$(timeout -k 5 20 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
  PICK=-1
  if [ -n "$INFO" ]; then
    for i in 0 1 2 3 4 5 6; do
      free=$(echo "$INFO" | awk -F', ' -v g=$i '$1==g {print $2}')
      if [ -n "$free" ] && [ "$free" -ge 20480 ]; then PICK=$i; break; fi
    done
  fi
  if [ "$PICK" != "-1" ] && [ "$PICK" = "$prev" ]; then cnt=$((cnt+1)); else cnt=0; fi
  prev=$PICK
  echo "[$(date)] poll: pick=GPU$PICK stable=$cnt" >> $LOG
  if [ "$PICK" != "-1" ] && [ "$cnt" -ge 3 ]; then
    echo "[$(date)] GPU $PICK stable -> launching afdb_1_1_tmw15" >> $LOG
    GPU=$PICK setsid nohup bash $SCRIPT > /home/chenty/abacust_mem/src/scripts/launch_afdb_1_1_tmw15_b36_t0.log 2>&1 < /dev/null &
    sleep 90
    if pgrep -f "train_afdb_1_1_tmw15_b36_t0.sh" > /dev/null; then
      echo "[$(date)] OK launched on GPU $PICK" >> $LOG; exit 0
    fi
    echo "[$(date)] launch died; keep waiting" >> $LOG; prev=-1; cnt=0
  fi
  sleep 60
done
echo "[$(date)] deadline reached; giving up" >> $LOG

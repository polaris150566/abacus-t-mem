#!/bin/bash
# 等一张 >=20GB 空闲且连续 3 分钟稳定的卡, 再启动 memedge16_enc3
# 连续稳定判断用于避免"另一个训练刚启动、还没分配显存"时被误判为空闲
LOG=/home/chenty/abacust_mem/src/scripts/log/wait_gpu_memedge16_enc3.log
SCRIPT=/home/chenty/abacust_mem/src/scripts/train_memedge16_enc3_b36_t0.sh
mkdir -p "$(dirname $LOG)"
echo "[$(date)] watcher(v2) started: threshold 20480MiB, need 3 consecutive polls, deadline 6h" >> $LOG
DEADLINE=$(( $(date +%s) + 6*3600 ))
prev=-1; cnt=0
while [ $(date +%s) -lt $DEADLINE ]; do
  # 若 B 已经在跑就直接退出
  if pgrep -f "train_memedge16_enc3_b36_t0.sh" > /dev/null; then
    echo "[$(date)] memedge16_enc3 already running; watcher exit" >> $LOG; exit 0
  fi
  INFO=$(timeout -k 5 20 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
  PICK=-1
  if [ -n "$INFO" ]; then
    for i in 3 4 5 6; do
      free=$(echo "$INFO" | awk -F', ' -v g=$i '$1==g {print $2}')
      if [ -n "$free" ] && [ "$free" -ge 20480 ]; then PICK=$i; break; fi
    done
  fi
  if [ "$PICK" != "-1" ] && [ "$PICK" = "$prev" ]; then cnt=$((cnt+1)); else cnt=0; fi
  prev=$PICK
  echo "[$(date)] poll: pick=GPU$PICK stable_count=$cnt" >> $LOG
  if [ "$PICK" != "-1" ] && [ "$cnt" -ge 3 ]; then
    echo "[$(date)] GPU $PICK stable for $cnt polls -> launching memedge16_enc3" >> $LOG
    GPU=$PICK setsid nohup bash $SCRIPT > /home/chenty/abacust_mem/src/scripts/launch_memedge16_enc3_b36_t0.log 2>&1 < /dev/null &
    sleep 90
    if pgrep -f "train_memedge16_enc3_b36_t0.sh" > /dev/null; then
      echo "[$(date)] OK launched on GPU $PICK; watcher exit" >> $LOG; exit 0
    fi
    echo "[$(date)] launch on GPU $PICK died; keep waiting" >> $LOG
    prev=-1; cnt=0
  fi
  sleep 60
done
echo "[$(date)] deadline reached; giving up" >> $LOG

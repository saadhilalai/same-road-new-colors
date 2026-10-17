#!/bin/bash
# Logs RAM, swap and GPU (memory MiB, temperature C, utilization %) every 60 s to logs/memory.log.
# Used to check that two trainings at once are safe for memory and heat.
cd "$(dirname "$0")/.." && mkdir -p logs
while true; do
  read -r used avail <<< "$(free -m | awk '/^Mem:/ {print $3, $7}')"
  swap=$(free -m | awk '/^Swap:/ {print $3}')
  gpu=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') ram_used_mb=$used ram_avail_mb=$avail swap_used_mb=$swap gpu=$gpu" >> logs/memory.log
  sleep 60
done

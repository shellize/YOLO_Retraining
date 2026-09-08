#!/bin/sh
set -u

# 精简版资源分析:系统负载 + 我的 CPU/内存占用
ME="${USER:-lijinhe}"

echo ""
echo "================ 系统概览 ================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S') | 主机: $(hostname) | 核数: $(nproc) | 运行: $(uptime -p | sed 's/^up //')"
echo "负载: $(uptime | grep -o 'load average.*' | sed 's/load average: //')"

echo ""
echo "================ CPU ================"
idle=$(top -bn1 | awk -F'[, ]+' '/^%Cpu/{print $8}')
total_cpu=$(awk "BEGIN{printf \"%.1f\", 100 - $idle}")
echo "整机 CPU 使用率: ${total_cpu}% (${idle}% 空闲)"

my_cpu=$(top -bn1 -u "$ME" | awk -v u="$ME" '$2==u && $9+0>0 {s+=$9} END{printf "%.0f", s}')
echo "我的 CPU 占用:   ${my_cpu}% ≈ $(awk "BEGIN{printf \"%.1f\", $my_cpu/100}") 个核 (占整机 $(awk "BEGIN{printf \"%.1f\", $my_cpu/(100*$(nproc))*100}")%)"

echo ""
echo "================ 内存 ================"
mem_total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
mem_avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
mem_used_kb=$((mem_total_kb - mem_avail_kb))
echo "整机内存: 共 $(awk "BEGIN{printf \"%.0f\", $mem_total_kb/1024/1024}")G, 已用 $(awk "BEGIN{printf \"%.0f\", $mem_used_kb/1024/1024}")G ($(awk "BEGIN{printf \"%.1f\", $mem_used_kb*100/$mem_total_kb}")%), 可用 $(awk "BEGIN{printf \"%.0f\", $mem_avail_kb/1024/1024}")G"

my_mem_kb=$(ps -u "$ME" -o rss= 2>/dev/null | awk '{s+=$1} END{print s+0}')
echo "我的内存占用:   $(awk "BEGIN{printf \"%.1f\", $my_mem_kb/1024/1024}")G (占整机 $(awk "BEGIN{printf \"%.1f\", $my_mem_kb*100/$mem_total_kb}")%)"

echo ""
echo "================ Swap ================"
swap_total_kb=$(awk '/SwapTotal/{print $2}' /proc/meminfo)
swap_free_kb=$(awk '/SwapFree/{print $2}' /proc/meminfo)
echo "Swap: 已用 $(awk "BEGIN{printf \"%.1f\", ($swap_total_kb-$swap_free_kb)/1024/1024}")G / $(awk "BEGIN{printf \"%.1f\", $swap_total_kb/1024/1024}")G ($(awk "BEGIN{printf \"%.0f\", ($swap_total_kb-$swap_free_kb)*100/$swap_total_kb}")%)"

echo ""
echo "================ 磁盘 ================"
df -h -x tmpfs -x overlay -x squashfs -x devtmpfs -x proc -x sysfs 2>/dev/null | awk 'NR==1 || $5+0 > 0'

echo ""
echo "--- 磁盘高水位检查 ---"
df -h -x tmpfs -x overlay -x squashfs -x devtmpfs 2>/dev/null | awk 'NR>1 && $5+0 >= 90 {print "[警告] 磁盘空间紧张: "$6" 已用 "$5" (剩 "$4")"}'
df -h -x tmpfs -x overlay -x squashfs -x devtmpfs 2>/dev/null | awk 'NR>1 && $5+0 < 90 && $5+0 >= 80 {print "[注意] 磁盘使用偏高: "$6" 已用 "$5" (剩 "$4")"}'

echo ""
echo "--- 我的磁盘占用 ---"
if [ -d "$HOME" ]; then
    echo "我的 home 目录 ($HOME): $(du -sh "$HOME" 2>/dev/null | cut -f1)"
fi
if [ -d "$HOME/work" ]; then
    echo "我的 work 目录: $(du -sh "$HOME/work" 2>/dev/null | cut -f1)"
fi

echo ""
echo "================ GPU ================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx util mem_used mem_total; do
        echo "GPU $idx: 利用率 ${util}%, 显存 ${mem_used}MiB / ${mem_total}MiB"
    done
    echo ""
    echo "--- GPU 进程详情 ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r gpu_uuid pid mem; do
        gpu_uuid=$(echo "$gpu_uuid" | tr -d ' ')
        pid=$(echo "$pid" | tr -d ' ')
        mem=$(echo "$mem" | tr -d ' ')
        # 从 index,uuid 列表反查 GPU 编号
        gpu_index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
            | awk -F',' -v u="$gpu_uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2==u) print $1}')
        [ -z "$gpu_index" ] && gpu_index="?"
        user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -z "$user" ] && user="unknown"
        # 取 argv[0] 可执行文件路径,过长时截断为 nvidia-smi 风格 "...尾部"
        pname=$(ps -o args= -p "$pid" 2>/dev/null | awk '{p=$1; if (length(p)>30) p="..." substr(p, length(p)-27); print p}')
        [ -z "$pname" ] && pname="?"
        echo "GPU $gpu_index | 用户: $user | 显存: ${mem}MiB | 进程: $pname"
    done
else
    echo "无 nvidia-smi"
fi

echo ""
echo "================ 我的进程 TOP 5 ================"
ps -u "$ME" -o pid,%cpu,%mem,rss,etime,cmd --sort=-%cpu 2>/dev/null | awk 'NR==1{print} NR>1{printf "%-8s %5s %5s %6.1fG %9s  %s\n", $1, $2, $3, $4/1024/1024, $5, substr($0, index($0,$6), 60)}' | head -6

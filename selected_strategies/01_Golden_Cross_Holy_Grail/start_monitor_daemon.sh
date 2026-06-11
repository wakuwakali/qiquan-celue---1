#!/bin/bash

echo "=============================================="
echo "终极圣杯 V10 - Linux 守护进程 (Daemon) 启动成功"
echo "此脚本将保证 live_monitor.py 永不掉线"
echo "即使报错崩溃，也会在 10 秒后满血复活自动重启"
echo "=============================================="

# 无限循环保护
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在拉起 live_monitor.py..."
    
    # 请确保您在 Linux 环境中安装了相关依赖 (pip install -r requirements.txt)
    # 并使用正确的 python3 解释器路径运行
    python3 live_monitor.py
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 警告：监控脚本异常退出！"
    echo "系统将在 10 秒后自动重新启动监控进程..."
    sleep 10
done

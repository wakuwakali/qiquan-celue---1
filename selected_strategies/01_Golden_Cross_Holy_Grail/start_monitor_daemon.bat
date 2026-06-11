@echo off
title V10_Holy_Grail_Monitor_Daemon
color 0A

echo ==============================================
echo 终极圣杯 V10 - 守护进程 (Daemon) 启动成功
echo 此脚本将保证 live_monitor.py 永不掉线
echo 即使报错崩溃，也会在 10 秒后满血复活自动重启
echo ==============================================

:loop
echo [%date% %time%] 正在拉起 live_monitor.py...
D:\anaconda3\envs\xbx_quant\python.exe e:\项目\期权回测\selected_strategies\01_Golden_Cross_Holy_Grail\live_monitor.py

echo.
echo [%date% %time%] 警告：监控脚本异常退出！
echo 系统将在 10 秒后自动重新启动监控进程...
timeout /t 10
goto loop

import requests
import json
from datetime import datetime
from live_monitor import fetch_binance_klines, send_wechat_work_msg

def run_test():
    print("========================================")
    print("开始运行通讯测试 (Binance API + 微信推送)")
    print("========================================")
    
    # 1. 测试 Binance API 通讯
    print("\n[1/2] 正在测试拉取币安交易所数据...")
    try:
        df = fetch_binance_klines('BTCUSDT', '1h', limit=5)
        latest_close = df['close'].iloc[-1]
        print(f"[OK] 成功连接币安！获取到最新 BTC 价格: {latest_close:.2f} USDT")
    except Exception as e:
        print(f"[ERROR] 币安连接失败: {e}")
        return

    # 2. 测试企业微信通知
    print("\n[2/2] 正在向您的企业微信发送测试报警消息...")
    try:
        test_msg = (
            "🔔 【量化系统连通性测试】\n\n"
            "您好，我是 Antigravity AI。\n"
            "这条消息说明您的企业微信机器人配置完全正确！\n"
            f"同时，交易所接口也通讯正常，当前 BTC 最新价格读取为: {latest_close:.2f} USDT。\n\n"
            "实盘监控程序（live_monitor.py）随时可以挂机运行。"
        )
        send_wechat_work_msg(test_msg)
        print("[OK] 微信推送指令已发出，请查看您的企业微信群！")
    except Exception as e:
        print(f"[ERROR] 微信推送失败: {e}")

if __name__ == '__main__':
    run_test()

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
import schedule
import traceback

from Holy_Grail_Engine import holy_grail_strategy

# ==========================================
# 终极圣杯 V10 - 微信实盘监控与报警系统
# 机制：每小时第 1 分钟运行，计算最新策略信号
# 报警：一旦状态改变，通过企业微信机器人推送
# ==========================================

# 请在此处填写您的企业微信机器人 Webhook URL
# 参考配置案例: https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxxxxxxxxxxxxx
WECHAT_WEBHOOK_URL = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=3276b389-ac32-4741-a8fd-b01203fef7c1'

SYMBOL = 'BTCUSDT'
TIMEFRAME = '1h'
STATE_FILE = 'current_position.json'

def send_wechat_work_msg(content):
    if not WECHAT_WEBHOOK_URL:
        print('未配置 WECHAT_WEBHOOK_URL，跳过微信发送')
        return
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": content + '\n\n' + f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
        }
        r = requests.post(WECHAT_WEBHOOK_URL, data=json.dumps(data), timeout=10)
        print(f'调用企业微信接口返回：{r.text}')
    except Exception as e:
        print(f"发送企业微信失败: {e}")
        print(traceback.format_exc())

def fetch_binance_klines(symbol, interval, limit=1000):
    """拉取币安现货 K 线数据"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if isinstance(data, dict) and 'code' in data:
        raise Exception(f"Binance API 返回错误: {data['msg']}")
        
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def run_monitor():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行实盘扫描...")
    try:
        # 1. 获取数据
        df = fetch_binance_klines(SYMBOL, '1h', limit=1000)
        
        # 2. 运行 V10 策略引擎
        signals = holy_grail_strategy(df)
        
        # 3. 获取最新一根 K 线的指示信号
        latest_signal = int(signals.iloc[-1])
        latest_close = df['close'].iloc[-1]
        
        # 4. 读取上一次的状态
        last_position = 0
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                try:
                    state = json.load(f)
                    last_position = state.get('position', 0)
                except:
                    pass
        
        # 5. 判断是否发生变化并报警
        if latest_signal != last_position:
            action = "【买入做多】信号触发！" if latest_signal == 1 else "【平仓空仓】信号触发！"
            
            msg = f"【终极圣杯 V10 策略预警】\n标的: {SYMBOL}\n动作: {action}\n当前价格: {latest_close:.2f}\n"
            if latest_signal == 1:
                msg += "建议：可配置 2倍杠杆或远月期权合成多头。"
            else:
                msg += "建议：立刻平仓，将资金转入活期理财或备兑卖出收息。"
                
            print(msg)
            send_wechat_work_msg(msg)
            
            # 更新状态文件
            with open(STATE_FILE, 'w') as f:
                json.dump({'position': latest_signal, 'last_update': str(datetime.now())}, f)
        else:
            print(f"信号无变化，继续保持 {'持仓' if latest_signal == 1 else '空仓'} 状态。")
            
    except Exception as e:
        error_msg = f"实盘监控程序发生异常: {e}"
        print(error_msg)
        send_wechat_work_msg(error_msg)

def send_heartbeat():
    """每日发送一次心跳消息，证明程序没有掉线"""
    try:
        df = fetch_binance_klines(SYMBOL, '1h', limit=5)
        latest_close = df['close'].iloc[-1]
        
        last_position = 0
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                try:
                    state = json.load(f)
                    last_position = state.get('position', 0)
                except:
                    pass
                    
        pos_str = "持有看涨多头" if last_position == 1 else "空仓吃息中"
        msg = (
            f"🟢 【系统心跳正常】\n"
            f"V10 量化监控系统已在线 24 小时。\n"
            f"当前 {SYMBOL} 最新价格: {latest_close:.2f}\n"
            f"当前策略指导仓位: {pos_str}\n"
            f"未触发换仓信号，继续挂机。"
        )
        send_wechat_work_msg(msg)
    except Exception as e:
        print(f"心跳播报异常: {e}")

if __name__ == '__main__':
    print("========================================")
    print("启动 V10 终极圣杯 微信监控预警系统")
    print("监控频率：每小时的第 1 分钟执行一次")
    print("心跳播报：每天 15:00 发送一次在线确认")
    print("========================================")
    
    # 程序启动时立即发送一条确认消息
    startup_msg = "🚀 【系统启动】终极圣杯 V10 实盘监控程序已成功在服务器启动并进入守护模式！\n行情监控与定时播报已就绪。"
    send_wechat_work_msg(startup_msg)
    
    # 设置定时任务：每小时的第 1 分钟运行
    schedule.every().hour.at(":01").do(run_monitor)
    
    # 设置心跳任务：每天 15:00 发送一条证明在线的消息
    schedule.every().day.at("15:00").do(send_heartbeat)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

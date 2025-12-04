import ccxt
import pandas as pd
import time
import schedule
import json
import os
from datetime import datetime

# ==========================================
# [1] 설정 영역
# ==========================================
SYMBOL = 'BTC/USDT'      # 거래 대상
TIMEFRAME_TREND = '1h'   # 추세 판단용 (1시간봉)
TIMEFRAME_ENTRY = '15m'  # 진입 타점용 (15분봉)

DATA_FILE = 'trading_data.json' # 데이터 저장 파일
INITIAL_BALANCE = 1000000       # 초기 자본금 (100만원)
INVEST_RATE = 0.5               # 1회 진입 비중 (50%)
STOP_LOSS_RATE = 0.05           # 손절 기준 (-5%)

# 바이낸스 객체 (시세 조회용)
exchange = ccxt.binance()

# ==========================================
# [2] 데이터 파일 관리 (JSON 저장)
# ==========================================
def load_data():
    """파일에서 투자 정보를 읽어옵니다."""
    if not os.path.exists(DATA_FILE):
        data = {
            "balance": INITIAL_BALANCE,
            "holding": None, # {'price': 0, 'amount': 0} 형태
            "history": []
        }
        save_data(data)
        return data
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    """투자 정보를 파일에 저장합니다."""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def execute_trade(action, price, market_data, reason):
    """거래를 실행하고 데이터를 갱신합니다."""
    data = load_data()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    amount = 0
    balance_before = data['balance']
    balance_after = balance_before

    # [매수 로직]
    if action == 'BUY':
        invest_amount = balance_before * INVEST_RATE
        amount = invest_amount / price
        balance_after = balance_before - invest_amount
        
        data['balance'] = balance_after
        data['holding'] = {'price': price, 'amount': amount}
        print(f"⚡ [매수] {amount:.6f} BTC (평단: {price:,.0f}) - {reason}")

    # [매도 로직]
    elif action == 'SELL':
        if not data['holding']: return # 보유량 없으면 리턴
        
        amount = data['holding']['amount']
        buy_price = data['holding']['price']
        
        # 수익금 계산 (매도금액 - 매수금액)
        sell_total = amount * price
        
        balance_after = balance_before + sell_total
        data['balance'] = balance_after
        data['holding'] = None # 포지션 초기화
        
        profit = (price - buy_price) * amount
        profit_rate = (price - buy_price) / buy_price * 100
        print(f"💰 [매도] 수익: {profit:,.0f} ({profit_rate:.2f}%) - {reason}")

    # 로그 기록
    record = {
        "time": now,
        "action": action,
        "price": price,
        "amount": amount,
        "balance_after": balance_after,
        "reason": reason,
        "analysis": market_data
    }
    data['history'].append(record)
    save_data(data)

# ==========================================
# [3] 시장 데이터 분석
# ==========================================
def get_market_status():
    """현재 차트(추세, 볼린저밴드, RSI)를 분석합니다."""
    try:
        # A. 1시간봉 (추세 확인)
        ohlcv_trend = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_TREND, limit=50)
        df_trend = pd.DataFrame(ohlcv_trend, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        ma_trend = df_trend['close'].rolling(20).mean().iloc[-1]
        is_uptrend = df_trend['close'].iloc[-1] > ma_trend

        # B. 15분봉 (타점 확인)
        ohlcv_entry = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_ENTRY, limit=100) # RSI 계산 위해 넉넉히
        df_entry = pd.DataFrame(ohlcv_entry, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        
        # 볼린저 밴드
        df_entry['ma'] = df_entry['close'].rolling(20).mean()
        df_entry['std'] = df_entry['close'].rolling(20).std()
        df_entry['upper'] = df_entry['ma'] + (df_entry['std'] * 2)
        df_entry['lower'] = df_entry['ma'] - (df_entry['std'] * 2)
        
        # RSI (Wilder's Smoothing 방식 적용 - 더 정확함)
        delta = df_entry['close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # ewm(com=13)은 기간 14의 Wilder's Smoothing과 유사
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        df_entry['rsi'] = 100 - (100 / (1 + rs))
        
        curr = df_entry.iloc[-1]
        
        return {
            "is_uptrend": is_uptrend,
            "price": curr['close'],
            "lower": curr['lower'],
            "upper": curr['upper'],
            "rsi": curr['rsi'],
            "trend_ma": ma_trend
        }
    except Exception as e:
        print(f"⚠️ 데이터 조회 실패: {e}")
        return None

# ==========================================
# [4] 봇 실행 로직
# ==========================================
def run_simulation():
    try:
        data = load_data()
        market = get_market_status()
        if market is None: return

        now_time = datetime.now().strftime('%H:%M:%S')
        price = market['price']
        
        # 자산 가치 계산
        total_asset = data['balance']
        if data['holding']:
            total_asset += data['holding']['amount'] * price
        
        yield_rate = ((total_asset - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        
        # 상태 로그 출력
        trend_icon = "📈상승세" if market['is_uptrend'] else "📉하락세"
        status_str = "무포지션"
        holding_profit_rate = 0
        
        if data['holding']:
            buy_price = data['holding']['price']
            holding_profit_rate = (price - buy_price) / buy_price # 소수점 비율
            status_str = f"보유중({holding_profit_rate*100:+.2f}%)"

        print(f"[{now_time}] {trend_icon} | RSI: {market['rsi']:.1f} | 자산: {int(total_asset):,} ({yield_rate:+.2f}%) | 상태: {status_str}")

        # 분석 데이터 (로그용)
        analysis_info = {
            "rsi": round(market['rsi'], 2),
            "bb_lower": round(market['lower'], 2),
            "bb_upper": round(market['upper'], 2),
        }

        # --- [매매 판단 로직] ---

        # 1. 매수 (무포지션 + 상승추세 + 볼린저 하단 터치)
        if data['holding'] is None:
            if market['is_uptrend'] and price <= market['lower']:
                # RSI가 과매도(30 이하)일 때만 살 수도 있음 (옵션)
                execute_trade('BUY', price, analysis_info, "상승장 눌림목(BB하단)")

        # 2. 매도 (보유중)
        elif data['holding'] is not None:
            # A. 익절 (볼린저 상단 터치)
            if price >= market['upper']:
                execute_trade('SELL', price, analysis_info, "볼린저 상단 익절")
            
            # B. 손절 (진입가 대비 -5% 하락 시)
            elif holding_profit_rate <= -STOP_LOSS_RATE:
                execute_trade('SELL', price, analysis_info, f"손절매 발동(-{STOP_LOSS_RATE*100}%)")
                
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        # 상세 에러 확인용 (필요시 주석 해제)
        # import traceback
        # traceback.print_exc()

# ==========================================
# [5] 실행부
# ==========================================
print(f"=== 트레이딩 봇 시작 ===")
print(f"대상: {SYMBOL}")
print(f"전략: {TIMEFRAME_TREND} 추세 + {TIMEFRAME_ENTRY} BB역추세")

# 스케줄러 설정 (1분마다 실행)
schedule.every(1).minutes.do(run_simulation)

run_simulation() # 시작 즉시 1회 실행

while True:
    schedule.run_pending()
    time.sleep(1)
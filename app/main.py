from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

app = FastAPI(title="Smart Trading Bot API")

# --- CORS 설정 (프론트엔드 연동) ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://arbihunter.github.io",
    "https://felicityprojects.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 데이터 모델 ---

class LatestRsiResponse(BaseModel):
    symbol: str
    timeframe: str
    rsi: float
    ema_short: Optional[float] = None # EMA 50
    ema_long: Optional[float] = None  # EMA 200
    bb_upper: Optional[float] = None  # 볼린저 상단
    bb_lower: Optional[float] = None  # 볼린저 하단
    divergence: Optional[str] = None  # BULLISH, BEARISH
    signal: str                       # 종합 한글 신호
    updated_at: datetime

class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float 
    rsi: float
    ema_short: Optional[float] = None
    ema_long: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    signal: Optional[str] = None

class RecentCandlesResponse(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]

# --- 유틸리티 및 계산 함수 ---

def to_binance_interval(tf: str) -> str:
    """프론트엔드 타임프레임을 바이낸스 포맷으로 변환"""
    mapping = {
        "1m": "1m", "3m": "3m", "5m": "5m", 
        "10m": "15m", "15m": "15m", "30m": "30m", 
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    if tf not in mapping:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 timeframe: {tf}")
    return mapping[tf]

def fetch_klines_from_binance(symbol: str, interval: str, limit: int = 1000):
    """바이낸스 데이터 가져오기 (EMA 정확도를 위해 1000개 권장)"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance API 오류: {resp.status_code}")
    
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")
    return data

def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI(14) 계산"""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_ema(closes: pd.Series, period: int) -> pd.Series:
    """EMA 계산"""
    return closes.ewm(span=period, adjust=False).mean()

def compute_bollinger_bands(closes: pd.Series, period: int = 20, std_dev: int = 2):
    """볼린저 밴드 (20, 2) 계산"""
    sma = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

# --- [핵심 1] 다이버전스 감지 로직 ---
def detect_divergence(df: pd.DataFrame, lookback: int = 30) -> Optional[str]:
    """최근 데이터에서 상승/하락 다이버전스 감지"""
    if len(df) < lookback: return None
    curr = df.iloc[-1]
    
    # 최근 2개 제외하고 과거 30개 구간 조사 (노이즈 제거)
    window = df.iloc[-(lookback + 2):-2]
    if window.empty: return None

    # 1. 상승 다이버전스 (Bullish): 가격 신저가 + RSI 저점 상승
    min_price_idx = window['close'].idxmin()
    past_low_candle = window.loc[min_price_idx]
    
    if (curr['close'] <= past_low_candle['close'] and 
        curr['rsi'] > (past_low_candle['rsi'] + 1.0) and 
        curr['rsi'] < 45):
        return "BULLISH"

    # 2. 하락 다이버전스 (Bearish): 가격 신고가 + RSI 고점 하락
    max_price_idx = window['close'].idxmax()
    past_high_candle = window.loc[max_price_idx]
    
    if (curr['close'] >= past_high_candle['close'] and 
        curr['rsi'] < (past_high_candle['rsi'] - 1.0) and 
        curr['rsi'] > 55):
        return "BEARISH"

    return None

# --- [핵심 2] 종합 신호 판별 로직 (수정됨) ---
def determine_comprehensive_signal(
    price: float, 
    ema_short: float, # EMA 50
    ema_long: float,  # EMA 200
    rsi: float, 
    divergence: Optional[str],
    bb_upper: float, 
    bb_lower: float
) -> str:
    """
    모든 지표를 종합하여 신호를 판별합니다.
    특히 '가격이 EMA 200 아래로 무너진 상황'을 정확히 필터링합니다.
    """
    if pd.isna(ema_long) or pd.isna(bb_upper):
        return "데이터 준비중"

    # 1. [최우선] 다이버전스 (강력한 반전 신호)
    if divergence == "BULLISH": return "강력 매수 (상승 다이버전스)"
    if divergence == "BEARISH": return "강력 매도 (하락 다이버전스)"

    # 2. [변동성] 볼린저 밴드 이탈 (단기 과열/침체)
    # 하락장이라도 밴드 하단을 찢고 RSI가 30 미만이면 기술적 반등 구간
    if price < bb_lower and rsi < 30:
        return "단기 매수 (볼린저 하단 과매도)"
    
    if price > bb_upper and rsi > 70:
        return "단기 매도 (볼린저 상단 과매수)"

    # --- 추세 판단 변수 ---
    is_golden_cross = ema_short > ema_long   # 50일선이 200일선 위에 있음 (골든크로스 상태)
    is_death_cross = ema_short < ema_long    # 50일선이 200일선 아래에 있음 (데드크로스 상태)
    price_above_trend = price > ema_long     # 현재 가격이 장기 추세선(200) 위에 있는가?

    # 3. [상황별 정밀 진단]

    # CASE A: 지표는 골든크로스 상태지만...
    if is_golden_cross:
        if price_above_trend:
            # 가격도 추세선 위에 있음 -> 진짜 상승장
            if rsi < 45: return "매수 (상승장 눌림목)"
            if rsi > 70: return "관망 (단기 과열)"
            return "매수 관점 (상승 추세 지속)"
        else:
            # [중요] 가격이 추세선 아래로 무너짐 -> 가짜 상승장 (하락 전환 중)
            if rsi < 30: return "단기 매수 (과낙폭 기술적 반등)"
            return "관망 (추세 이탈 / 데드크로스 주의)"

    # CASE B: 지표가 데드크로스 상태 (하락장)
    if is_death_cross:
        if not price_above_trend:
            # 가격도 추세선 아래에 있음 -> 진짜 하락장
            if rsi > 60: return "매도 (하락장 기술적 반등)"
            if rsi < 30: return "관망 (하락장 과매도)"
            return "매도 관점 (하락 추세 지속)"
        else:
            # 가격이 추세선 위로 올라옴 -> 추세 전환 시도 중
            return "관망 (추세 회복 시도 중)"

    return "관망 (중립)"

# --- API Endpoints ---

@app.get("/api/indicators/latest-rsi", response_model=LatestRsiResponse)
async def get_latest_rsi(
    symbol: str = Query(..., min_length=1, description="예: BTCUSDT"),
    timeframe: str = Query(..., min_length=1, description="예: 1h, 4h"),
):
    binance_interval = to_binance_interval(timeframe)
    klines = fetch_klines_from_binance(symbol, binance_interval, limit=1000)

    # 데이터프레임 변환
    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "vol", "ct", "qav", "nt", "tbb", "tbq", "ig"])
    df["close"] = df["close"].astype(float)
    
    # 지표 계산
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["ema_short"] = compute_ema(df["close"], period=50)  # EMA 50
    df["ema_long"] = compute_ema(df["close"], period=200)  # EMA 200
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"], period=20, std_dev=2)
    
    latest = df.iloc[-1]
    
    # 다이버전스 감지
    div_status = detect_divergence(df, lookback=30)
    
    # 종합 신호 판별
    sig = determine_comprehensive_signal(
        price=latest["close"],
        ema_short=latest["ema_short"],
        ema_long=latest["ema_long"],
        rsi=latest["rsi"],
        divergence=div_status,
        bb_upper=latest["bb_upper"],
        bb_lower=latest["bb_lower"]
    )

    return LatestRsiResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        rsi=round(latest["rsi"], 2),
        ema_short=round(latest["ema_short"], 2) if not pd.isna(latest["ema_short"]) else None,
        ema_long=round(latest["ema_long"], 2) if not pd.isna(latest["ema_long"]) else None,
        bb_upper=round(latest["bb_upper"], 2) if not pd.isna(latest["bb_upper"]) else None,
        bb_lower=round(latest["bb_lower"], 2) if not pd.isna(latest["bb_lower"]) else None,
        divergence=div_status,
        signal=sig,
        updated_at=datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc),
    )

@app.get("/api/indicators/recent-candles", response_model=RecentCandlesResponse)
async def get_recent_candles(
    symbol: str = Query(..., min_length=1),
    timeframe: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=200),
):
    binance_interval = to_binance_interval(timeframe)
    klines = fetch_klines_from_binance(symbol, binance_interval, limit=1000)

    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "vol", "ct", "qav", "nt", "tbb", "tbq", "ig"])
    cols = ["open", "high", "low", "close", "volume"]
    df.rename(columns={"vol": "volume"}, inplace=True)
    df[cols] = df[cols].astype(float)
    
    # 지표 계산
    df["rsi"] = compute_rsi(df["close"], period=14)
    df["ema_short"] = compute_ema(df["close"], period=50)
    df["ema_long"] = compute_ema(df["close"], period=200)
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"], period=20, std_dev=2)

    df_res = df.iloc[-limit:].copy()
    
    candles_list = []
    for _, row in df_res.iterrows():
        # 각 과거 시점별 신호 계산 (다이버전스 제외)
        sig = determine_comprehensive_signal(
            price=row["close"],
            ema_short=row["ema_short"],
            ema_long=row["ema_long"],
            rsi=row["rsi"],
            divergence=None,
            bb_upper=row["bb_upper"],
            bb_lower=row["bb_lower"]
        )
        
        candles_list.append(Candle(
            time=datetime.fromtimestamp(row["open_time"] / 1000, tz=timezone.utc),
            open=row["open"], high=row["high"], low=row["low"], close=row["close"],
            volume=row["volume"],
            rsi=round(row["rsi"], 2),
            ema_short=round(row["ema_short"], 2) if not pd.isna(row["ema_short"]) else None,
            ema_long=round(row["ema_long"], 2) if not pd.isna(row["ema_long"]) else None,
            bb_upper=round(row["bb_upper"], 2) if not pd.isna(row["bb_upper"]) else None,
            bb_lower=round(row["bb_lower"], 2) if not pd.isna(row["bb_lower"]) else None,
            signal=sig
        ))

    return RecentCandlesResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        candles=candles_list
    )

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Smart Trading Bot API"}
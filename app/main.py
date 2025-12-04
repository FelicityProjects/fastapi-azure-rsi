from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import json
import os

app = FastAPI(title="Smart Trading Bot API")

# --- [1] CORS 설정 (프론트엔드 연동) ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://arbihunter.github.io",
    "https://felicityprojects.github.io",
    "*" # 개발 중 편의를 위해 모든 출처 허용 (배포 시 제거 권장)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [2] 설정 변수 ---
DATA_FILE = 'trading_data.json'  # 봇이 데이터를 저장하는 파일명

# --- [3] Pydantic 데이터 모델 (응답 구조 정의) ---

# 3-1. 지표 관련 모델
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

# 3-2. 트레이딩 내역 관련 모델 (새로 추가됨)
class TradeRecord(BaseModel):
    time: str
    action: str
    price: float
    amount: float
    balance_after: float
    reason: str
    analysis: Optional[Dict[str, Any]] = None

class HoldingInfo(BaseModel):
    price: float
    amount: float

class TradeHistoryResponse(BaseModel):
    balance: float
    holding: Optional[HoldingInfo] = None
    history: List[TradeRecord]

# --- [4] 유틸리티 및 계산 함수 ---

def load_trade_data():
    """trading_data.json 파일을 읽어서 반환"""
    if not os.path.exists(DATA_FILE):
        return {
            "balance": 0,
            "holding": None,
            "history": []
        }
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"파일 읽기 에러: {e}")
        return {"balance": 0, "holding": None, "history": []}

def to_binance_interval(tf: str) -> str:
    mapping = {
        "1m": "1m", "3m": "3m", "5m": "5m", 
        "10m": "15m", "15m": "15m", "30m": "30m", 
        "1h": "1h", "4h": "4h", "1d": "1d"
    }
    if tf not in mapping:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 timeframe: {tf}")
    return mapping[tf]

def fetch_klines_from_binance(symbol: str, interval: str, limit: int = 1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Binance API 오류: {resp.status_code}")
    
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")
    return data

def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()

def compute_bollinger_bands(closes: pd.Series, period: int = 20, std_dev: int = 2):
    sma = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    return sma + (std * std_dev), sma - (std * std_dev)

def detect_divergence(df: pd.DataFrame, lookback: int = 30) -> Optional[str]:
    if len(df) < lookback: return None
    curr = df.iloc[-1]
    window = df.iloc[-(lookback + 2):-2]
    if window.empty: return None

    # 상승 다이버전스 (Bullish)
    min_price_idx = window['close'].idxmin()
    past_low = window.loc[min_price_idx]
    if (curr['close'] <= past_low['close'] and 
        curr['rsi'] > (past_low['rsi'] + 1.0) and curr['rsi'] < 45):
        return "BULLISH"

    # 하락 다이버전스 (Bearish)
    max_price_idx = window['close'].idxmax()
    past_high = window.loc[max_price_idx]
    if (curr['close'] >= past_high['close'] and 
        curr['rsi'] < (past_high['rsi'] - 1.0) and curr['rsi'] > 55):
        return "BEARISH"
    return None

def determine_comprehensive_signal(price, ema_short, ema_long, rsi, divergence, bb_upper, bb_lower):
    if pd.isna(ema_long) or pd.isna(bb_upper): return "데이터 준비중"
    
    if divergence == "BULLISH": return "강력 매수 (상승 다이버전스)"
    if divergence == "BEARISH": return "강력 매도 (하락 다이버전스)"

    if price < bb_lower and rsi < 30: return "단기 매수 (볼린저 하단 과매도)"
    if price > bb_upper and rsi > 70: return "단기 매도 (볼린저 상단 과매수)"

    is_golden_cross = ema_short > ema_long
    is_death_cross = ema_short < ema_long
    price_above_trend = price > ema_long

    if is_golden_cross:
        if price_above_trend:
            if rsi < 45: return "매수 (상승장 눌림목)"
            if rsi > 70: return "관망 (단기 과열)"
            return "매수 관점 (상승 추세 지속)"
        else:
            if rsi < 30: return "단기 매수 (과낙폭 기술적 반등)"
            return "관망 (추세 이탈 / 데드크로스 주의)"

    if is_death_cross:
        if not price_above_trend:
            if rsi > 60: return "매도 (하락장 기술적 반등)"
            if rsi < 30: return "관망 (하락장 과매도)"
            return "매도 관점 (하락 추세 지속)"
        else:
            return "관망 (추세 회복 시도 중)"

    return "관망 (중립)"

# --- [5] API Endpoints ---

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Smart Trading Bot API"}

# 5-1. 매매 내역 조회 (NEW)
@app.get("/api/trading/history", response_model=TradeHistoryResponse)
async def get_trade_history():
    """봇이 기록한 매매 내역 조회"""
    data = load_trade_data()
    return data

# 5-2. 최신 지표 조회
@app.get("/api/indicators/latest-rsi", response_model=LatestRsiResponse)
async def get_latest_rsi(
    symbol: str = Query(..., min_length=1),
    timeframe: str = Query(..., min_length=1),
):
    binance_interval = to_binance_interval(timeframe)
    klines = fetch_klines_from_binance(symbol, binance_interval, limit=1000)

    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "vol", "ct", "qav", "nt", "tbb", "tbq", "ig"])
    df["close"] = df["close"].astype(float)
    
    df["rsi"] = compute_rsi(df["close"])
    df["ema_short"] = compute_ema(df["close"], 50)
    df["ema_long"] = compute_ema(df["close"], 200)
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    
    latest = df.iloc[-1]
    div_status = detect_divergence(df)
    sig = determine_comprehensive_signal(
        latest["close"], latest["ema_short"], latest["ema_long"], latest["rsi"],
        div_status, latest["bb_upper"], latest["bb_lower"]
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

# 5-3. 과거 캔들 차트 데이터 조회
@app.get("/api/indicators/recent-candles", response_model=RecentCandlesResponse)
async def get_recent_candles(
    symbol: str = Query(..., min_length=1),
    timeframe: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=200),
):
    binance_interval = to_binance_interval(timeframe)
    klines = fetch_klines_from_binance(symbol, binance_interval, limit=1000)

    df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "vol", "ct", "qav", "nt", "tbb", "tbq", "ig"])
    cols = ["open", "high", "low", "close"]
    df.rename(columns={"vol": "volume"}, inplace=True)
    df[cols + ["volume"]] = df[cols + ["volume"]].astype(float)
    
    df["rsi"] = compute_rsi(df["close"])
    df["ema_short"] = compute_ema(df["close"], 50)
    df["ema_long"] = compute_ema(df["close"], 200)
    df["bb_upper"], df["bb_lower"] = compute_bollinger_bands(df["close"])

    df_res = df.iloc[-limit:].copy()
    candles_list = []
    
    for _, row in df_res.iterrows():
        sig = determine_comprehensive_signal(
            row["close"], row["ema_short"], row["ema_long"], row["rsi"],
            None, row["bb_upper"], row["bb_lower"]
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
# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np

app = FastAPI(title="RSI Indicator API (Binance Spot)")

# --- CORS (프론트 도메인 맞게 수정) ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://arbihunter.github.io",
    "https://felicityprojects.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # 개발 중엔 ["*"]로 풀어도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 ---

class LatestRsiResponse(BaseModel):
    symbol: str
    timeframe: str
    rsi: float
    updated_at: datetime


class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float


class RecentCandlesResponse(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]


# --- 프론트 타임프레임 -> 바이낸스 interval 매핑 ---
# Binance 지원 interval: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M :contentReference[oaicite:2]{index=2}
def to_binance_interval(tf: str) -> str:
    mapping = {
        "1m": "1m",
        "5m": "5m",
        # "10m": 없음! -> 일단 15m로 올리거나 에러 처리
        "10m": "15m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    if tf not in mapping:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 timeframe: {tf}")
    return mapping[tf]


# --- RSI 계산 함수 (EMA 방식, TradingView와 유사) ---
def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- 바이낸스에서 OHLC 가져오는 함수 ---
def fetch_klines_from_binance(symbol: str, interval: str, limit: int = 200):
    """
    Binance Spot /api/v3/klines 호출
    https://api.binance.com/api/v3/klines :contentReference[oaicite:3]{index=3}
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Binance 오류: {resp.status_code} {resp.text}",
        )
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise HTTPException(status_code=404, detail="Binance에서 데이터 없음")
    return data



# --- 최신 RSI 1개 반환 ---
@app.get("/api/indicators/latest-rsi", response_model=LatestRsiResponse)
async def get_latest_rsi(
    symbol: str = Query(..., min_length=1, description="예: BTCUSDT, ETHUSDT"),
    timeframe: str = Query(..., min_length=1, description="예: 1m, 5m, 1h, 4h, 1d"),
):
    binance_interval = to_binance_interval(timeframe)
    klines = fetch_klines_from_binance(symbol, binance_interval, limit=200)

    closes = pd.Series([float(k[4]) for k in klines])  # k[4] = 종가 :contentReference[oaicite:4]{index=4}
    rsi_series = compute_rsi(closes, period=14)
    latest_rsi = rsi_series.iloc[-1]

    # 마지막 캔들 시간 (open time ms)
    last_open_time_ms = klines[-1][0]
    last_open_time = datetime.fromtimestamp(
        last_open_time_ms / 1000, tz=timezone.utc
    )

    return LatestRsiResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        rsi=float(round(latest_rsi, 2)),
        updated_at=last_open_time,
    )


# --- 최근 N개 캔들 + 각 캔들의 RSI 반환 ---
@app.get("/api/indicators/recent-candles", response_model=RecentCandlesResponse)
async def get_recent_candles(
    symbol: str = Query(..., min_length=1, description="예: BTCUSDT, ETHUSDT"),
    timeframe: str = Query(..., min_length=1, description="예: 1m, 5m, 1h, 4h, 1d"),
    limit: int = Query(10, ge=1, le=200, description="리턴할 캔들 개수"),
):
    binance_interval = to_binance_interval(timeframe)
    # RSI 계산 위해서는 limit보다 조금 더 많이 받아오는게 안전
    raw_limit = max(limit + 50, 100)

    klines = fetch_klines_from_binance(symbol, binance_interval, limit=raw_limit)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["rsi"] = compute_rsi(df["close"], period=14)

    # 최신 순으로 limit개만 사용
    df = df.iloc[-limit:]

    candles: List[Candle] = []
    for _, row in df.iterrows():
        t = datetime.fromtimestamp(row["open_time"] / 1000, tz=timezone.utc)
        candles.append(
            Candle(
                time=t,
                open=float(round(row["open"], 4)),
                high=float(round(row["high"], 4)),
                low=float(round(row["low"], 4)),
                close=float(round(row["close"], 4)),
                volume=float(round(row["volume"], 4)),
                rsi=float(round(row["rsi"], 2)),
            )
        )

    return RecentCandlesResponse(
        symbol=symbol.upper(),
        timeframe=timeframe,
        candles=candles,
    )

# --- 헬스 체크 엔드포인트 ---
@app.get("/")
async def health_check():
    """
    API 서버 상태 확인 (Health Check)
    """
    return {"status": "ok", "service": "RSI Indicator API", "server_time": datetime.now(timezone.utc)}

# 로컬 실행:
# uvicorn main:app --reload --port 8000

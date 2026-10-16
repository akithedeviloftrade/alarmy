import os
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta

# =========================================================================
# 1. 환경 설정 (스마트폰 ntfy 앱에서 구독한 본인의 토픽명)
# =========================================================================
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "whatever-you-want")
IS_SCHEDULED = os.getenv("GITHUB_EVENT_NAME") == "schedule"
KST = timezone(timedelta(hours=9))

# =========================================================================
# 2. 감시 대상 36개 종목 정의
# =========================================================================
US_STOCKS = {
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "TSLA": "TSLA",
    "AMZN": "AMZN", "GOOGL": "GOOGL", "META": "META", "AMD": "AMD", "QQQ": "QQQ"
}

KR_STOCKS = {
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "LG엔솔": "373220.KS",
    "삼바": "207940.KS", "현대차": "005380.KS", "기아": "000270.KS",
    "셀트리온": "068270.KS", "NAVER": "035420.KS", "카카오": "035720.KS"
}

MAJOR_COINS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT"
]

ALT_COINS = [
    "APT/USDT", "NEAR/USDT", "RENDER/USDT", "FET/USDT", "SEI/USDT",
    "INJ/USDT", "TIA/USDT", "ARB/USDT", "SUI/USDT"
]

# =========================================================================
# 3. 트레이딩뷰 Pine Script 100% 일치 수식 (SMA 시딩 적용)
# =========================================================================
def tv_ema(series: pd.Series, length: int) -> pd.Series:
    """Pine Script ta.ema()와 동일: 첫 length개의 SMA로 초기값 시딩 후 지수평활"""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) < length:
        return pd.Series(out, index=series.index)
    out[length - 1] = np.mean(vals[:length])
    alpha = 2.0 / (length + 1.0)
    for i in range(length, len(vals)):
        out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)

def tv_rma(series: pd.Series, length: int) -> pd.Series:
    """Pine Script ta.rma()와 동일: Wilder's Smoothing (alpha = 1/length)"""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) < length:
        return pd.Series(out, index=series.index)
    out[length - 1] = np.mean(vals[:length])
    alpha = 1.0 / length
    for i in range(length, len(vals)):
        out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)

def tv_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Pine Script ta.rsi()와 동일"""
    delta = close.diff()
    gain = delta.clip(lower=0.0).iloc[1:]
    loss = (-delta.clip(upper=0.0)).iloc[1:]
    avg_gain = tv_rma(gain, length)
    avg_loss = tv_rma(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi.reindex(close.index)

# =========================================================================
# 4. 캔들 시점 판별 및 저점 반등 조건 검사
# =========================================================================
def check_bottom_signal(df: pd.DataFrame, last_ts_utc: datetime) -> str:
    if len(df) < 50:
        return ""

    now_utc = datetime.now(timezone.utc)
    elapsed_min = (now_utc - last_ts_utc).total_seconds() / 60.0

    # 정기 스케줄 실행 시 이미 장이 마감된 지 2시간 이상 지난 주식은 중복 알림 제외
    if IS_SCHEDULED and elapsed_min > 125:
        return ""

    # 마지막 봉이 시작된 지 50분 미만이면(현재 진행 중인 미완성 봉) -> 직전 완성봉(-2) 검사
    # 장이 마감되었거나 봉 마감 직전(50분 이상 경과)이면 -> 마지막 봉(-1) 검사
    if elapsed_min < 50:
        curr = -2
        prev = -3
    else:
        curr = -1
        prev = -2

    close = df["close"]
    open_ = df["open"]
    low = df["low"]

    # 1) RSI (14)
    rsi = tv_rsi(close, 14)

    # 2) 볼린저 밴드 (20, 2.0)
    basis = close.rolling(20).mean()
    dev = 2.0 * close.rolling(20).std(ddof=0)
    lower_bb = basis - dev

    # 3) EMA (5, 20)
    ema5 = tv_ema(close, 5)
    ema20 = tv_ema(close, 20)

    reasons = []
    # 조건 1: RSI 45 이하에서 상향 턴어라운드
    if rsi.iloc[prev] < 45.0 and rsi.iloc[curr] > rsi.iloc[prev]:
        reasons.append(f"RSI턴:{rsi.iloc[prev]:.1f}→{rsi.iloc[curr]:.1f}")

    # 조건 2: 볼린저 밴드 하단 터치 후 양봉 반등
    if (low.iloc[prev] <= lower_bb.iloc[prev] or low.iloc[curr] <= lower_bb.iloc[curr]) and (close.iloc[curr] > open_.iloc[curr]):
        reasons.append("BB하단반등")

    # 조건 3: 5 EMA의 20 EMA 골든크로스
    if ema5.iloc[prev] <= ema20.iloc[prev] and ema5.iloc[curr] > ema20.iloc[curr]:
        reasons.append(f"EMA골크")

    if not reasons:
        return ""

    # 검사한 캔들의 한국시간(KST) 표시
    candle_time_kst = df["dt_utc"].iloc[curr].astimezone(KST).strftime("%H:%M")
    return f"[{candle_time_kst}봉] {', '.join(reasons)}"

# =========================================================================
# 5. 주식(yfinance) 및 코인(ccxt) 스캔
# =========================================================================
def scan_stocks(ticker_map: dict) -> list[str]:
    hits = []
    symbols = list(ticker_map.values())
    try:
        # 3개월치(약 450개 1시간 봉) + 무수정 주가(auto_adjust=False)로 트레이딩뷰와 1:1 일치
        data = yf.download(
            symbols, period="3mo", interval="1h",
            group_by="ticker", auto_adjust=False, prepost=False, progress=False
        )
        for name, sym in ticker_map.items():
            try:
                sub = data[sym][["Open", "High", "Low", "Close"]].dropna().copy()
                if sub.empty:
                    continue
                sub.columns = ["open", "high", "low", "close"]
                idx_utc = pd.to_datetime(sub.index, utc=True)
                sub["dt_utc"] = idx_utc
                last_ts_utc = idx_utc[-1].to_pydatetime()

                res = check_bottom_signal(sub, last_ts_utc)
                if res:
                    hits.append(f"• {name} {res}")
            except Exception:
                continue
    except Exception as e:
        print(f"Stock scan error: {e}")
    return hits

def scan_crypto(symbol_list: list) -> list[str]:
    hits = []
    # GitHub Actions(미국 IP) 차단을 우회하는 바이낸스 공식 데이터 엔드포인트 사용
    binance = ccxt.binance({
        "enableRateLimit": True,
        "urls": {
            "api": {
                "public": "https://data-api.binance.vision/api/v3",
                "sapi": "https://data-api.binance.vision/api/v3",
            }
        }
    })
    for sym in symbol_list:
        try:
            # 500개 캔들을 가져와 EMA(20) 및 RSI(14) 오차 제거
            ohlcv = binance.fetch_ohlcv(sym, timeframe="1h", limit=500)
            if not ohlcv:
                continue
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            last_ts_utc = df["dt_utc"].iloc[-1].to_pydatetime()

            res = check_bottom_signal(df, last_ts_utc)
            if res:
                short_name = sym.replace("/USDT", "")
                hits.append(f"• {short_name} {res}")
        except Exception as e:
            print(f"Crypto scan error ({sym}): {e}")
            continue
    return hits

# =========================================================================
# 6. ntfy 스마트폰 긴급 팝업 푸시 발송
# =========================================================================
def send_ntfy(title: str, message: str):
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": "urgent",
        "Tags": "chart_with_upwards_trend,bell"
    }
    requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=10)

if __name__ == "__main__":
    us_hits = scan_stocks(US_STOCKS)
    kr_hits = scan_stocks(KR_STOCKS)
    mc_hits = scan_crypto(MAJOR_COINS)
    ac_hits = scan_crypto(ALT_COINS)

    sections = []
    if us_hits: sections.append("🇺🇸 [미국주식]\n" + "\n".join(us_hits))
    if kr_hits: sections.append("🇰🇷 [한국주식]\n" + "\n".join(kr_hits))
    if mc_hits: sections.append("🪙 [메이저코인]\n" + "\n".join(mc_hits))
    if ac_hits: sections.append("🚀 [알트코인]\n" + "\n".join(ac_hits))

    if sections:
        now_str = datetime.now(KST).strftime("%H:%M")
        title = f"🔥 [1H 저점 매수 타점] {now_str}"
        body = "\n\n".join(sections)
        print(f"{title}\n{body}")
        send_ntfy(title, body)
    else:
        print("현재 1시간 봉 저점 반등 신호가 없습니다.")
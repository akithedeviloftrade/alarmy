import os
import ccxt
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# =========================================================================
# 1. ntfy 설정 (스마트폰 앱에서 구독한 본인의 토픽명)
# =========================================================================
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "whatever-you-want")

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
# 3. 1시간 봉 저점 반등 수식 (Pine Script 동일 로직)
# =========================================================================
def check_bottom_signal(df: pd.DataFrame) -> list[str]:
    if len(df) < 30:
        return []

    close = df["close"]
    open_ = df["open"]
    low = df["low"]

    # 1) RSI (14) - Wilder's Smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 2) 볼린저 밴드 (20, 2.0)
    basis = close.rolling(20).mean()
    dev = 2.0 * close.rolling(20).std(ddof=0)
    lower_bb = basis - dev

    # 3) EMA (5, 20)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    # 최근 확정 봉(-2)과 직전 봉(-3) 비교
    curr = -2
    prev = -3

    reasons = []
    # 조건 1: RSI 45 이하에서 턴
    if rsi.iloc[prev] < 45 and rsi.iloc[curr] > rsi.iloc[prev]:
        reasons.append(f"RSI({rsi.iloc[curr]:.1f})")

    # 조건 2: BB 하단 터치 후 양봉
    if (low.iloc[prev] <= lower_bb.iloc[prev] or low.iloc[curr] <= lower_bb.iloc[curr]) and (close.iloc[curr] > open_.iloc[curr]):
        reasons.append("BB하단반등")

    # 조건 3: 5 EMA 골든크로스
    if ema5.iloc[prev] <= ema20.iloc[prev] and ema5.iloc[curr] > ema20.iloc[curr]:
        reasons.append("EMA골크")

    return reasons

# =========================================================================
# 4. 주식 / 코인 스캔
# =========================================================================
def scan_stocks(ticker_map: dict) -> list[str]:
    hits = []
    symbols = list(ticker_map.values())
    data = yf.download(symbols, period="5d", interval="1h", group_by="ticker", progress=False)
    for name, sym in ticker_map.items():
        try:
            df = data[sym].dropna()
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
            reasons = check_bottom_signal(df)
            if reasons:
                hits.append(f"{name}({', '.join(reasons)})")
        except Exception:
            continue
    return hits

def scan_crypto(symbol_list: list) -> list[str]:
    hits = []
    binance = ccxt.binance({"enableRateLimit": True})
    for sym in symbol_list:
        try:
            ohlcv = binance.fetch_ohlcv(sym, timeframe="1h", limit=50)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            reasons = check_bottom_signal(df)
            if reasons:
                short_name = sym.replace("/USDT", "")
                hits.append(f"{short_name}({', '.join(reasons)})")
        except Exception:
            continue
    return hits

# =========================================================================
# 5. ntfy 스마트폰 긴급 팝업 푸시 발송
# =========================================================================
def send_ntfy(title: str, message: str):
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {
        "Title": title.encode("utf-8"),
        "Priority": "urgent",  # 화면 켜짐 및 소리/진동 강제 트리거
        "Tags": "chart_with_upwards_trend,bell"
    }
    requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=10)

if __name__ == "__main__":
    us_hits = scan_stocks(US_STOCKS)
    kr_hits = scan_stocks(KR_STOCKS)
    mc_hits = scan_crypto(MAJOR_COINS)
    ac_hits = scan_crypto(ALT_COINS)

    sections = []
    if us_hits: sections.append("🇺🇸 미국: " + ", ".join(us_hits))
    if kr_hits: sections.append("🇰🇷 한국: " + ", ".join(kr_hits))
    if mc_hits: sections.append("🪙 메이저: " + ", ".join(mc_hits))
    if ac_hits: sections.append("🚀 알트: " + ", ".join(ac_hits))

    if sections:
        now_str = datetime.now().strftime("%H:%M")
        title = f"🔥 [1H 저점 매수 타점] {now_str}"
        body = "\n".join(sections)
        print(f"{title}\n{body}")
        send_ntfy(title, body)
    else:
        print("현재 1시간 봉 저점 반등 신호가 없습니다.")
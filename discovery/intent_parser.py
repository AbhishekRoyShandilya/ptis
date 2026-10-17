# discovery/intent_parser.py
"""
Stage 1 — Intent Parser

Takes raw natural-language input like:
    "Look at Reliance 15-min chart, today's breakout at 10:30 with
     solid momentum, find similar across FNO stocks"

Extracts structured PatternQuery with:
    symbol, timeframe, event_time, pattern_type, scope, similarity_threshold

Two modes:
  1. LLM mode (preferred) — sends to Claude for extraction
  2. Rule-based fallback  — regex + keyword heuristics
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from discovery.models import PatternQuery, PatternType

logger = logging.getLogger(__name__)

# ── FNO stock tickers (NSE) — top 200 ──────────────────────────────────
# This list is used for ticker extraction from free-form text.
# Sorted alphabetically for binary search.
_FNO_TICKERS = sorted([
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC",
    "ADANIENT", "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP",
    "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL",
    "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK",
    "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL",
    "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT",
    "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL",
    "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELTACORP", "DIVISLAB",
    "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS",
    "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK", "GMRINFRA",
    "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM",
    "GUJGASLTD", "HAL", "HAVELLS", "HCLTECH", "HDFC",
    "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "HONAUT", "IBULHSGFIN",
    "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB",
    "IEX", "IGL", "INDHOTEL", "INDIACEM", "INDIAMART",
    "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "IOC",
    "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JKCEMENT",
    "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "L&TFH", "LALPATHLAB",
    "LAURUSLABS", "LICHSGFIN", "LT", "LTIMINDTREE", "LTTS",
    "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO",
    "MARUTI", "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL",
    "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NBCC", "NESTLEIND",
    "NMDC", "NTPC", "OBEROIRLTY", "OFSS", "ONGC",
    "PAGEIND", "PEL", "PERSISTENT", "PETRONET", "PFC",
    "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POWERGRID",
    "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE",
    "SAL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV",
    "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO",
    "ZEEL", "ZYDUSLIFE",
    # Index futures
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
])
_FNO_SET = set(t.upper() for t in _FNO_TICKERS)

# ── Common aliases ────────────────────────────
_ALIASES: Dict[str, str] = {
    "reliance": "RELIANCE", "ril": "RELIANCE",
    "hdfc bank": "HDFCBANK", "hdfcbank": "HDFCBANK",
    "icici bank": "ICICIBANK", "icicibank": "ICICIBANK",
    "infy": "INFY", "infosys": "INFY",
    "tcs": "TCS",
    "sbi": "SBIN", "state bank": "SBIN",
    "axis bank": "AXISBANK", "axisbank": "AXISBANK",
    "kotak bank": "KOTAKBANK", "kotak": "KOTAKBANK",
    "nifty": "NIFTY", "nifty 50": "NIFTY",
    "bank nifty": "BANKNIFTY", "banknifty": "BANKNIFTY",
    "tata motors": "TATAMOTORS", "tatamotors": "TATAMOTORS",
    "tata steel": "TATASTEEL", "tatasteel": "TATASTEEL",
    "maruti": "MARUTI",
    "bajaj finance": "BAJFINANCE", "bajfinance": "BAJFINANCE",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH", "hcltech": "HCLTECH",
    "tech mahindra": "TECHM", "techm": "TECHM",
    "l&t": "LT", "larsen": "LT",
    "titan": "TITAN",
    "asian paints": "ASIANPAINT", "asianpaint": "ASIANPAINT",
    "dr reddy": "DRREDDY", "drreddy": "DRREDDY",
    "sun pharma": "SUNPHARMA", "sunpharma": "SUNPHARMA",
    "adani ent": "ADANIENT", "adanient": "ADANIENT",
    "adani ports": "ADANIPORTS", "adaniports": "ADANIPORTS",
    "itc": "ITC",
    "nifty fut": "NIFTY",
}

# ── Pattern type keywords ─────────────────────
_PATTERN_KEYWORDS: Dict[PatternType, list[str]] = {
    PatternType.MOMENTUM_BREAKOUT: [
        "momentum", "breakout", "broke out", "strong move", "explosive",
        "surge", "ripped", "blasted", "power move",
    ],
    PatternType.VOLUME_BREAKOUT: [
        "volume breakout", "volume spike", "heavy volume", "vol surge",
    ],
    PatternType.PULLBACK_ENTRY: [
        "pullback", "dip buy", "retracement", "bounce", "test of",
        "pulled back", "came back to",
    ],
    PatternType.REVERSAL: [
        "reversal", "reversed", "v-shape", "bottom", "top", "turning point",
        "rejection", "hammer",
    ],
    PatternType.GAP_PLAY: [
        "gap up", "gap down", "gapped", "gap fill", "opening gap",
    ],
    PatternType.RANGE_BREAKOUT: [
        "range breakout", "consolidation breakout", "box breakout",
        "broke the range", "sideways break",
    ],
    PatternType.VWAP_RECLAIM: [
        "vwap reclaim", "reclaimed vwap", "crossed vwap", "vwap bounce",
    ],
    PatternType.SQUEEZE_BREAKOUT: [
        "squeeze", "bb squeeze", "bollinger squeeze", "compression",
        "coiled", "tight range then",
    ],
    PatternType.TREND_CONTINUATION: [
        "continuation", "trend continuation", "follow through",
        "resumed trend", "kept going",
    ],
}

# ── Timeframe patterns ────────────────────────
_TF_PATTERNS = [
    (r"(\d+)\s*min(ute)?s?\b", lambda m: f"{m.group(1)}m"),
    (r"(\d+)\s*m\b", lambda m: f"{m.group(1)}m"),
    (r"(\d+)\s*h(ou)?r?s?\b", lambda m: f"{m.group(1)}h"),
    (r"daily|1\s*d\b", lambda m: "1d"),
    (r"weekly|1\s*w\b", lambda m: "1w"),
]


class IntentParser:
    """
    Parses natural-language market observations into PatternQuery.

    Usage:
        parser = IntentParser(llm_client=my_llm)   # or None for rule-based
        query = await parser.parse("Look at Reliance 15min breakout at 10:30...")
    """

    def __init__(self, llm_client: Any = None):
        """
        Args:
            llm_client: Optional LLM client with an async `generate(prompt)` method.
                        If None, uses pure rule-based extraction.
        """
        self._llm = llm_client

    async def parse(self, raw_input: str) -> PatternQuery:
        """Parse natural-language input into a structured PatternQuery."""
        if self._llm:
            try:
                return await self._parse_with_llm(raw_input)
            except Exception as e:
                logger.warning("LLM parse failed, falling back to rules: %s", e)

        return self._parse_rule_based(raw_input)

    # ── LLM-based extraction ──────────────────────────────────────────

    async def _parse_with_llm(self, raw_input: str) -> PatternQuery:
        prompt = f"""You are a trading intent parser. Extract structured fields from
the user's natural-language market observation.

USER INPUT: {raw_input}

Return ONLY valid JSON with these fields:
{{
    "symbol": "<NSE ticker, e.g. RELIANCE>",
    "timeframe": "<e.g. 15m, 5m, 1h>",
    "event_time": "<HH:MM IST or null>",
    "event_date": "<YYYY-MM-DD or null>",
    "pattern_type": "<one of: momentum_breakout, volume_breakout, pullback_entry, reversal, gap_play, range_breakout, vwap_reclaim, squeeze_breakout, trend_continuation, custom>",
    "description": "<1-2 sentence description of the market event>",
    "scope": "<fno, nifty50, or comma-separated tickers>",
    "similarity_threshold": <0.85 to 0.95>,
    "lookback_years": <1 to 5>
}}"""

        from agents.prompts import INTENT_PARSER as _SYS
        response = await self._llm.generate(prompt, system_prompt=_SYS, caller="discovery:intent_parse")
        # Extract JSON from response
        text = response if isinstance(response, str) else str(response)
        json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in LLM response")

        data = json.loads(json_match.group())
        query = PatternQuery(raw_input=raw_input)

        query.symbol = data.get("symbol", "").upper()
        query.timeframe = data.get("timeframe", "15m")
        query.event_time = data.get("event_time")
        query.event_date = data.get("event_date")
        query.description = data.get("description", "")
        query.scope = data.get("scope", "fno")
        query.similarity_threshold = float(data.get("similarity_threshold", 0.90))
        query.lookback_years = int(data.get("lookback_years", 3))

        pt = data.get("pattern_type", "custom")
        try:
            query.pattern_type = PatternType(pt)
        except ValueError:
            query.pattern_type = PatternType.CUSTOM

        return query

    # ── Rule-based extraction ─────────────────────────────────────────

    def _parse_rule_based(self, raw_input: str) -> PatternQuery:
        query = PatternQuery(raw_input=raw_input)
        text = raw_input.lower()

        # 1. Extract symbol
        query.symbol = self._extract_symbol(text)

        # 2. Extract timeframe
        query.timeframe = self._extract_timeframe(text)

        # 3. Extract time of day
        query.event_time = self._extract_time(text)

        # 4. Extract date (default: today)
        query.event_date = self._extract_date(text)

        # 5. Classify pattern type
        query.pattern_type = self._classify_pattern(text)

        # 6. Extract scope
        query.scope = self._extract_scope(text)

        # 7. Extract similarity threshold
        sim_match = re.search(r'(\d{2,3})\s*%?\s*similar', text)
        if sim_match:
            val = int(sim_match.group(1))
            query.similarity_threshold = val / 100 if val > 1 else val

        # 8. Extract lookback
        lb_match = re.search(r'(\d+)\s*year', text)
        if lb_match:
            query.lookback_years = min(int(lb_match.group(1)), 10)

        # 9. Build description
        query.description = self._build_description(query)

        return query

    def _extract_symbol(self, text: str) -> str:
        # Check aliases first (longer matches first)
        for alias in sorted(_ALIASES.keys(), key=len, reverse=True):
            if alias in text:
                return _ALIASES[alias]
        # Check raw tickers
        words = re.findall(r'[A-Za-z&-]+', text.upper())
        for w in words:
            if w in _FNO_SET:
                return w
        return ""

    def _extract_timeframe(self, text: str) -> str:
        for pattern, extractor in _TF_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return extractor(m)
        return "15m"  # default for intraday

    def _extract_time(self, text: str) -> Optional[str]:
        # "at 10:30", "around 10:30 AM", "10:30 IST"
        m = re.search(r'(?:at|around|near)\s*(\d{1,2})[:\.](\d{2})\s*(am|pm|ist)?',
                       text, re.IGNORECASE)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            ampm = (m.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            return f"{hour:02d}:{minute:02d}"

        # "10:30"
        m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
        if m:
            return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        if "today" in text:
            return datetime.now().strftime("%Y-%m-%d")
        if "yesterday" in text:
            from datetime import timedelta
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        # "2026-03-10" or "10 march" etc.
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if m:
            return m.group()
        return datetime.now().strftime("%Y-%m-%d")

    def _classify_pattern(self, text: str) -> PatternType:
        best_type = PatternType.CUSTOM
        best_score = 0
        for ptype, keywords in _PATTERN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_type = ptype
        return best_type

    def _extract_scope(self, text: str) -> str:
        if "nifty 50" in text or "nifty50" in text:
            return "nifty50"
        if "all stocks" in text or "all nse" in text:
            return "all"
        if "fno" in text or "f&o" in text or "futures" in text:
            return "fno"
        return "fno"  # default

    def _build_description(self, query: PatternQuery) -> str:
        parts = []
        if query.symbol:
            parts.append(f"{query.symbol}")
        parts.append(f"{query.timeframe} timeframe")
        parts.append(f"{query.pattern_type.value.replace('_', ' ')}")
        if query.event_time:
            parts.append(f"at {query.event_time} IST")
        if query.event_date:
            parts.append(f"on {query.event_date}")
        return " — ".join(parts) if parts else "Market pattern observation"

import asyncio
import dataclasses
import json
import logging
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Optional, Protocol

import aiohttp
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("hedge-bot")


class NonRetryableCycleError(RuntimeError):
    pass


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _symbol_key(raw: str) -> str:
    s = raw.strip().upper()
    s = s.replace("/USDT-P", "").replace("/USDT", "").replace("-PERP", "").replace("-USD", "")
    for sep in ("/", "-"):
        if sep in s:
            s = s.split(sep)[0]
            break
    return s


def _parse_map_str(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in _split_csv(raw):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[_symbol_key(k)] = v.strip()
    return out


def _parse_map_int(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in _parse_map_str(raw).items():
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


def _parse_map_float(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in _parse_map_str(raw).items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


@dataclasses.dataclass(frozen=True)
class BotConfig:
    symbols: list[str]
    hold_seconds_min: int
    hold_seconds_max: int
    position_size_tokens: float
    position_size_tokens_by_symbol: dict[str, float]
    position_notional_usd: float
    extended_market_default: str
    extended_markets_by_symbol: dict[str, str]
    nado_product_id_default: int
    nado_product_ids_by_symbol: dict[str, int]
    nado_size_increment_x18_default: int
    nado_size_increment_x18_by_symbol: dict[str, int]
    pair_select_mode: str
    take_profit_pct: float
    stop_loss_pct: float
    price_check_interval_sec: float
    native_tpsl_enabled: bool
    spread_min_bps: float
    max_daily_loss_usd: float
    max_cycles: int
    cycle_pause_seconds: int
    cycle_retry_count: int
    cycle_retry_pause_seconds: int
    taker_fee_bps: float
    estimated_funding_bps_per_hour: float
    execution_mode: str
    dry_run: bool
    fallback_price: float
    long_exchange: str
    side_mix_mode: str
    require_first_symbol_success: bool
    size_jitter_min_pct: float
    size_jitter_max_pct: float
    size_jitter_usd_min: float
    size_jitter_usd_max: float
    min_fill_ratio: float
    fill_confirm_timeout_sec: float
    fill_confirm_poll_sec: float
    watch_interval_sec: float
    max_cross_price_gap_pct: float

    @staticmethod
    def from_env() -> "BotConfig":
        legacy_symbol = os.getenv("HEDGE_SYMBOL", "BTC-PERP")
        symbols_raw = os.getenv("HEDGE_SYMBOLS", legacy_symbol)
        symbols = [_symbol_key(x) for x in _split_csv(symbols_raw)]
        if not symbols:
            symbols = ["BTC"]
        return BotConfig(
            symbols=symbols,
            hold_seconds_min=int(os.getenv("HOLD_SECONDS_MIN", "45")),
            hold_seconds_max=int(os.getenv("HOLD_SECONDS_MAX", "90")),
            position_size_tokens=float(os.getenv("POSITION_SIZE_TOKENS", "0")),
            position_size_tokens_by_symbol=_parse_map_float(os.getenv("POSITION_SIZE_TOKENS_BY_PAIR", "")),
            position_notional_usd=float(os.getenv("POSITION_NOTIONAL_USD", "5")),
            extended_market_default=(os.getenv("EXTENDED_MARKET_NAME") or "BTC-USD").strip(),
            extended_markets_by_symbol=_parse_map_str(os.getenv("EXTENDED_MARKETS_BY_PAIR", "")),
            nado_product_id_default=int(os.getenv("NADO_PRODUCT_ID", "2")),
            nado_product_ids_by_symbol=_parse_map_int(os.getenv("NADO_PRODUCT_IDS_BY_PAIR", "")),
            nado_size_increment_x18_default=int(os.getenv("NADO_SIZE_INCREMENT_X18", "50000000000000")),
            nado_size_increment_x18_by_symbol=_parse_map_int(os.getenv("NADO_SIZE_INCREMENTS_X18_BY_PAIR", "")),
            pair_select_mode=(os.getenv("PAIR_SELECT_MODE", "random").strip().lower()),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0")),
            price_check_interval_sec=float(os.getenv("PRICE_CHECK_INTERVAL_SEC", "1.0")),
            native_tpsl_enabled=_env_bool("NATIVE_TPSL_ENABLED", False),
            spread_min_bps=float(os.getenv("SPREAD_MIN_BPS", "0")),
            max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "2")),
            max_cycles=int(os.getenv("MAX_CYCLES", "30")),
            cycle_pause_seconds=int(os.getenv("CYCLE_PAUSE_SECONDS", "5")),
            cycle_retry_count=int(os.getenv("CYCLE_RETRY_COUNT", "3")),
            cycle_retry_pause_seconds=int(os.getenv("CYCLE_RETRY_PAUSE_SECONDS", "4")),
            taker_fee_bps=float(os.getenv("TAKER_FEE_BPS", "5")),
            estimated_funding_bps_per_hour=float(os.getenv("EST_FUNDING_BPS_PER_HOUR", "1.5")),
            execution_mode=os.getenv("EXECUTION_MODE", "stub").strip().lower(),
            dry_run=_env_bool("DRY_RUN", True),
            fallback_price=float(os.getenv("REFERENCE_PRICE_USD", "600")),
            long_exchange=os.getenv("LONG_EXCHANGE", "extended").strip().lower(),
            side_mix_mode=os.getenv("SIDE_MIX_MODE", "alternate").strip().lower(),
            require_first_symbol_success=_env_bool("REQUIRE_FIRST_SYMBOL_SUCCESS", True),
            size_jitter_min_pct=float(os.getenv("POSITION_SIZE_JITTER_MIN_PCT", "-8")),
            size_jitter_max_pct=float(os.getenv("POSITION_SIZE_JITTER_MAX_PCT", "8")),
            size_jitter_usd_min=float(os.getenv("POSITION_SIZE_JITTER_USD_MIN", "0")),
            size_jitter_usd_max=float(os.getenv("POSITION_SIZE_JITTER_USD_MAX", "0")),
            min_fill_ratio=float(os.getenv("MIN_FILL_RATIO", "0.6")),
            fill_confirm_timeout_sec=float(os.getenv("FILL_CONFIRM_TIMEOUT_SEC", "8")),
            fill_confirm_poll_sec=float(os.getenv("FILL_CONFIRM_POLL_SEC", "0.5")),
            watch_interval_sec=float(os.getenv("WATCH_INTERVAL_SEC", "300")),
            max_cross_price_gap_pct=float(os.getenv("MAX_CROSS_PRICE_GAP_PCT", "20")),
        )

    def symbol_for_cycle(self, cycle_no: int) -> str:
        if self.pair_select_mode == "rotate":
            return self.symbols[(cycle_no - 1) % len(self.symbols)]
        return random.choice(self.symbols)

    def position_size_for_symbol(self, symbol: str, mid_price: float) -> float:
        k = _symbol_key(symbol)
        val = self.position_size_tokens_by_symbol.get(k, 0.0)
        if val > 0:
            return val
        if self.position_size_tokens > 0:
            return self.position_size_tokens
        # legacy fallback in USD if token size wasn't set
        return max(self.position_notional_usd / max(mid_price, 0.0001), 0.0000001)

    def randomized_position_size_for_symbol(self, symbol: str, mid_price: float, prev_size: Optional[float]) -> float:
        base = self.position_size_for_symbol(symbol, mid_price)
        lo = min(self.size_jitter_min_pct, self.size_jitter_max_pct) / 100.0
        hi = max(self.size_jitter_min_pct, self.size_jitter_max_pct) / 100.0
        usd_lo = min(self.size_jitter_usd_min, self.size_jitter_usd_max)
        usd_hi = max(self.size_jitter_usd_min, self.size_jitter_usd_max)
        if lo == 0 and hi == 0:
            candidate = base
        else:
            candidate = base
            eps = max(base * 0.01, 1e-9)
            for _ in range(8):
                factor = 1.0 + random.uniform(lo, hi)
                if factor <= 0:
                    continue
                c = max(base * factor, 0.0000001)
                c = round(c, 10)
                if prev_size is None or abs(c - prev_size) >= eps:
                    candidate = c
                    break
                candidate = c
        if usd_hi > 0 and mid_price > 0:
            usd_mag = random.uniform(max(0.0, usd_lo), usd_hi)
            usd_sign = -1.0 if random.choice([True, False]) else 1.0
            usd_delta = usd_sign * usd_mag
            token_delta = usd_delta / max(mid_price, 0.0000001)
            candidate = max(candidate + token_delta, 0.0000001)
            candidate = round(candidate, 10)
        eps = max(base * 0.01, 1e-9)
        if prev_size is not None and abs(candidate - prev_size) < eps:
            nudge = eps if random.choice([True, False]) else -eps
            candidate = max(candidate + nudge, 0.0000001)
            candidate = round(candidate, 10)
        return candidate

    def extended_market_for(self, symbol: str) -> str:
        return self.extended_markets_by_symbol.get(_symbol_key(symbol), self.extended_market_default)

    def nado_product_for(self, symbol: str) -> int:
        return self.nado_product_ids_by_symbol.get(_symbol_key(symbol), self.nado_product_id_default)

    def nado_size_increment_for(self, symbol: str) -> int:
        return self.nado_size_increment_x18_by_symbol.get(_symbol_key(symbol), self.nado_size_increment_x18_default)


@dataclasses.dataclass
class Fill:
    exchange: str
    symbol: str
    side: str
    qty: float
    price: float
    fee_usd: float
    ts: float
    order_id: Optional[str] = None


@dataclasses.dataclass
class LegResult:
    exchange: str
    realized_pnl_usd: float
    fees_usd: float
    funding_usd: float
    traded_volume_usd: float


class ExchangeAdapter(Protocol):
    name: str

    async def mid_price(self, symbol: str) -> float:
        ...

    async def market_order(self, symbol: str, side: str, size_tokens: float, reduce_only: bool = False) -> Fill:
        ...

    async def close_market_order(self, open_fill: Fill) -> Fill:
        ...

    async def place_native_tpsl(self, open_fill: Fill, tp_pct: float, sl_pct: float) -> None:
        ...

    async def position_size(self, symbol: str) -> float:
        ...

    async def normalize_order_size(self, symbol: str, size_tokens: float) -> float:
        ...

    async def account_balance_usd(self) -> float:
        ...

    async def pre_cycle_cleanup(self, symbol: str) -> None:
        ...


class TinyLiveStubExchange:
    def __init__(self, name: str, cfg: BotConfig, seed_offset: int = 0) -> None:
        self.name = name
        self.cfg = cfg
        self._rng = random.Random(1000 + seed_offset)
        self._base_price = cfg.fallback_price
        self._positions: dict[str, float] = {}

    async def mid_price(self, symbol: str) -> float:
        drift = self._rng.uniform(-0.0015, 0.0015)
        self._base_price *= 1 + drift
        return self._base_price

    async def market_order(self, symbol: str, side: str, size_tokens: float, reduce_only: bool = False) -> Fill:
        mid = await self.mid_price(symbol)
        qty = size_tokens
        sign = 1.0 if side.lower() == "buy" else -1.0
        cur = self._positions.get(symbol, 0.0)
        nxt = cur + (sign * qty)
        if reduce_only:
            # Do not allow increasing exposure in reduce-only mode.
            if cur > 0 and nxt > cur:
                nxt = cur
            elif cur < 0 and nxt < cur:
                nxt = cur
        self._positions[symbol] = nxt
        fee = (qty * mid) * (self.cfg.taker_fee_bps / 10_000)
        return Fill(self.name, symbol, side.lower(), qty, mid, fee, time.time(), order_id="stub-order")

    async def close_market_order(self, open_fill: Fill) -> Fill:
        close_side = "sell" if open_fill.side == "buy" else "buy"
        return await self.market_order(open_fill.symbol, close_side, open_fill.qty)

    async def place_native_tpsl(self, open_fill: Fill, tp_pct: float, sl_pct: float) -> None:
        return None

    async def position_size(self, symbol: str) -> float:
        return float(self._positions.get(symbol, 0.0))

    async def normalize_order_size(self, symbol: str, size_tokens: float) -> float:
        return max(float(size_tokens), 0.0000001)

    async def account_balance_usd(self) -> float:
        return float(self.cfg.fallback_price)

    async def pre_cycle_cleanup(self, symbol: str) -> None:
        return None


class NadoLiveExchange:
    def __init__(self, cfg: BotConfig) -> None:
        self.name = "nado"
        self.cfg = cfg
        self.mode = (os.getenv("NADO_MODE", "mainnet") or "mainnet").strip().lower()
        self.market_slippage = float(os.getenv("NADO_MARKET_SLIPPAGE", "0.02"))
        self.wallet = (os.getenv("NADO_MAIN_WALLET_ADDRESS") or "").strip()
        self.subaccount = (os.getenv("NADO_SUBACCOUNT") or "default").strip()
        self.signer_key = (os.getenv("NADO_SIGNER_PRIVATE_KEY") or "").strip()
        self.gateway = (os.getenv("NADO_GATEWAY_REST_ENDPOINT") or "https://gateway.prod.nado.xyz/v1").rstrip("/")
        self.bridge_python = (os.getenv("NADO_BRIDGE_PYTHON") or ".venv_nado/bin/python").strip()
        self.bridge_script = (os.getenv("NADO_BRIDGE_SCRIPT") or "nado_bridge.py").strip()
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        self._actual_inc_cache: dict[int, int] = {}

    def _run_bridge(self, product_id: int, side: str, qty: float, reduce_only: bool = False) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "market",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
            "--side",
            side,
            "--qty",
            str(qty),
            "--slippage",
            str(self.market_slippage),
            "--reduce-only",
            "true" if reduce_only else "false",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado bridge invalid output: {(res.stdout or '')[:280]}")

    def _get_price_bridge(self, product_id: int) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "price",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado price bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado price bridge invalid output: {(res.stdout or '')[:4000]}")

    def _get_position_bridge(self, product_id: int) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "position",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado position bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado position bridge invalid output: {(res.stdout or '')[:4000]}")

    def _get_balance_bridge(self) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "balance",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(self.cfg.nado_product_for(self.cfg.symbols[0])),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado balance bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado balance bridge invalid output: {(res.stdout or '')[:4000]}")

    def _get_product_meta_bridge(self, product_id: int) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "product_meta",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado product meta bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado product meta bridge invalid output: {(res.stdout or '')[:4000]}")

    def _actual_size_increment_for(self, symbol: str) -> int:
        product_id = self.cfg.nado_product_for(symbol)
        cached = self._actual_inc_cache.get(product_id)
        if cached is not None and cached > 0:
            return cached
        inc = 0
        try:
            meta = self._get_product_meta_bridge(product_id)
            if bool(meta.get("ok")):
                inc = int(meta.get("size_increment_x18", 0) or 0)
        except Exception as e:
            logger.warning("NADO_META_FALLBACK | symbol=%s product_id=%s reason=%s", symbol, product_id, repr(e))
        if inc <= 0:
            inc = max(self.cfg.nado_size_increment_for(symbol), 1)
        self._actual_inc_cache[product_id] = inc
        return inc

    def _run_trigger_bridge(
        self,
        *,
        product_id: int,
        price_x18: int,
        amount_x18: int,
        trigger_price_x18: int,
        trigger_type: str,
    ) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "trigger",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
            "--price-x18",
            str(price_x18),
            "--amount-x18",
            str(amount_x18),
            "--trigger-price-x18",
            str(trigger_price_x18),
            "--trigger-type",
            trigger_type,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado trigger bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado trigger bridge invalid output: {(res.stdout or '')[:4000]}")

    def _cancel_trigger_bridge(self, product_id: int) -> dict[str, Any]:
        cmd = [
            self.bridge_python,
            self.bridge_script,
            "--action",
            "cancel_triggers",
            "--mode",
            self.mode,
            "--signer-key",
            self.signer_key,
            "--wallet",
            self.wallet,
            "--subaccount",
            self.subaccount,
            "--product-id",
            str(product_id),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            stdout = (res.stdout or "").strip()
            details = "\n".join(part for part in [stderr, stdout] if part).strip()
            raise RuntimeError(f"nado cancel trigger bridge failed (exit={res.returncode}): {details[:4000]}")
        try:
            return json.loads((res.stdout or "").strip() or "{}")
        except Exception:
            raise RuntimeError(f"nado cancel trigger bridge invalid output: {(res.stdout or '')[:4000]}")

    def _normalize_qty(self, symbol: str, qty: float) -> float:
        # Nado requires amount*x1e18 to be divisible by product size increment.
        raw = int(Decimal(str(qty)) * (Decimal(10) ** 18))
        inc = self._actual_size_increment_for(symbol)
        if raw <= 0:
            raw = inc
        # Round up to the nearest valid increment to avoid rejection on tiny notionals.
        rounded = ((raw + inc - 1) // inc) * inc
        return float(Decimal(rounded) / (Decimal(10) ** 18))

    async def close(self) -> None:
        await self._session.close()

    async def mid_price(self, symbol: str) -> float:
        product_id = self.cfg.nado_product_for(symbol)
        try:
            data = self._get_price_bridge(product_id)
            mid = float(data.get("mid", 0) or 0)
            if mid > 0:
                return mid
            bid = float(data.get("bid", 0) or 0)
            ask = float(data.get("ask", 0) or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            if bid > 0:
                return bid
            if ask > 0:
                return ask
        except Exception as e:
            logger.warning("NADO_PRICE_FALLBACK | symbol=%s reason=%s", symbol, repr(e))
            return self.cfg.fallback_price
        logger.warning("NADO_PRICE_FALLBACK | symbol=%s reason=empty_price", symbol)
        return self.cfg.fallback_price

    async def market_order(self, symbol: str, side: str, size_tokens: float, reduce_only: bool = False) -> Fill:
        side_l = side.lower()
        mid = await self.mid_price(symbol)
        qty = self._normalize_qty(symbol, max(size_tokens, 0.0000001))
        fee = (qty * mid) * (self.cfg.taker_fee_bps / 10_000)
        if self.cfg.dry_run:
            return Fill(self.name, symbol, side_l, qty, mid, fee, time.time(), order_id="nado-dry-run")

        result = self._run_bridge(self.cfg.nado_product_for(symbol), side_l, qty, reduce_only=reduce_only)
        order_id = str(result.get("order_id", "")) or None
        return Fill(self.name, symbol, side_l, qty, mid, fee, time.time(), order_id=order_id)

    async def close_market_order(self, open_fill: Fill) -> Fill:
        # Close by current live position state, because TP/SL may have already flattened this leg.
        cur_pos = await self.position_size(open_fill.symbol)
        if abs(cur_pos) < 0.0000001:
            px = await self.mid_price(open_fill.symbol)
            return Fill(
                exchange=self.name,
                symbol=open_fill.symbol,
                side="buy" if open_fill.side == "sell" else "sell",
                qty=open_fill.qty,
                price=px,
                fee_usd=0.0,
                ts=time.time(),
                order_id="nado-already-flat",
            )
        close_side = "sell" if cur_pos > 0 else "buy"
        close_qty = abs(cur_pos)
        return await self.market_order(open_fill.symbol, close_side, close_qty, reduce_only=True)

    async def place_native_tpsl(self, open_fill: Fill, tp_pct: float, sl_pct: float) -> None:
        if tp_pct <= 0 and sl_pct <= 0:
            return
        if self.cfg.dry_run:
            return
        product_id = self.cfg.nado_product_for(open_fill.symbol)
        cancel_res = self._cancel_trigger_bridge(product_id)
        logger.info("NADO_TRIGGERS_CLEARED | symbol=%s resp=%s", open_fill.symbol, str(cancel_res)[:280])
        qty_x18 = int(Decimal(str(open_fill.qty)) * (Decimal(10) ** 18))
        close_amount_x18 = -qty_x18 if open_fill.side == "buy" else qty_x18
        entry = Decimal(str(open_fill.price))
        tp_price = entry * Decimal(str(1 + tp_pct / 100)) if open_fill.side == "buy" else entry * Decimal(str(1 - tp_pct / 100))
        sl_price = entry * Decimal(str(1 - sl_pct / 100)) if open_fill.side == "buy" else entry * Decimal(str(1 + sl_pct / 100))
        # Use trigger price as limit price for visibility and deterministic behavior.
        if tp_pct > 0:
            tp_x18 = int(tp_price * (Decimal(10) ** 18))
            tp_trigger = "mid_price_above" if open_fill.side == "buy" else "mid_price_below"
            tp_res = self._run_trigger_bridge(
                product_id=product_id,
                price_x18=tp_x18,
                amount_x18=close_amount_x18,
                trigger_price_x18=tp_x18,
                trigger_type=tp_trigger,
            )
            logger.info("NADO_TP_SET | symbol=%s trigger=%s resp=%s", open_fill.symbol, tp_trigger, str(tp_res)[:280])
        if sl_pct > 0:
            sl_x18 = int(sl_price * (Decimal(10) ** 18))
            sl_trigger = "mid_price_below" if open_fill.side == "buy" else "mid_price_above"
            sl_res = self._run_trigger_bridge(
                product_id=product_id,
                price_x18=sl_x18,
                amount_x18=close_amount_x18,
                trigger_price_x18=sl_x18,
                trigger_type=sl_trigger,
            )
            logger.info("NADO_SL_SET | symbol=%s trigger=%s resp=%s", open_fill.symbol, sl_trigger, str(sl_res)[:280])

    async def position_size(self, symbol: str) -> float:
        if self.cfg.dry_run:
            return 0.0
        product_id = self.cfg.nado_product_for(symbol)
        data = self._get_position_bridge(product_id)
        return float(data.get("position_size", 0.0) or 0.0)

    async def normalize_order_size(self, symbol: str, size_tokens: float) -> float:
        return self._normalize_qty(symbol, max(size_tokens, 0.0000001))

    async def account_balance_usd(self) -> float:
        if self.cfg.dry_run:
            return float(self.cfg.fallback_price)
        data = self._get_balance_bridge()
        return float(data.get("balance_usd", 0.0) or 0.0)

    async def pre_cycle_cleanup(self, symbol: str) -> None:
        if self.cfg.dry_run:
            return
        product_id = self.cfg.nado_product_for(symbol)
        try:
            res = self._cancel_trigger_bridge(product_id)
            logger.info("PRE_CLEAN | exchange=nado symbol=%s cancel_triggers=%s", symbol, str(res)[:200])
        except Exception as e:
            logger.warning("PRE_CLEAN_WARN | exchange=nado symbol=%s cancel_triggers_err=%s", symbol, repr(e))


class ExtendedLiveExchange:
    def __init__(self, cfg: BotConfig) -> None:
        self.name = "extended"
        self.cfg = cfg
        self.env = (os.getenv("EXTENDED_ENV") or "mainnet").strip().lower()
        self.api_key = (os.getenv("EXTENDED_API_KEY") or "").strip()
        self.public_key = (os.getenv("EXTENDED_PUBLIC_KEY") or "").strip()
        self.private_key = (os.getenv("EXTENDED_PRIVATE_KEY") or "").strip()
        self.vault = int((os.getenv("EXTENDED_VAULT") or "0").strip() or "0")
        self.slippage_pct = float(os.getenv("EXTENDED_MARKET_SLIPPAGE_PCT", "0.2"))
        self._client = None
        self._order_side_enum = None
        self._order_tpsl_type_enum = None
        self._order_trigger_price_type_enum = None
        self._order_price_type_enum = None
        self._time_in_force_enum = None
        self._order_tpsl_trigger_param_cls = None
        self._markets: dict[str, Any] = {}

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from x10.perpetual.accounts import StarkPerpetualAccount  # type: ignore
            from x10.perpetual.configuration import MAINNET_CONFIG, TESTNET_CONFIG  # type: ignore
            from x10.perpetual.order_object import OrderTpslTriggerParam  # type: ignore
            from x10.perpetual.orders import (  # type: ignore
                OrderPriceType,
                OrderSide,
                OrderTpslType,
                OrderTriggerPriceType,
                TimeInForce,
            )
            from x10.perpetual.trading_client import PerpetualTradingClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Extended SDK is missing. Install `x10-python-trading-starknet` and retry."
            ) from e

        account = StarkPerpetualAccount(
            vault=self.vault,
            private_key=self.private_key,
            public_key=self.public_key,
            api_key=self.api_key,
        )
        cfg = TESTNET_CONFIG if self.env in {"testnet", "sepolia"} else MAINNET_CONFIG
        self._client = PerpetualTradingClient(cfg, account)
        self._order_side_enum = OrderSide
        self._order_tpsl_type_enum = OrderTpslType
        self._order_trigger_price_type_enum = OrderTriggerPriceType
        self._order_price_type_enum = OrderPriceType
        self._time_in_force_enum = TimeInForce
        self._order_tpsl_trigger_param_cls = OrderTpslTriggerParam

    async def mid_price(self, symbol: str) -> float:
        self._ensure_client()
        assert self._client is not None
        try:
            market_name = self.cfg.extended_market_for(symbol)
            if market_name not in self._markets:
                markets = await self._client.markets_info.get_markets_dict()
                self._markets[market_name] = markets.get(market_name)
            market = self._markets.get(market_name)
            if market is not None:
                stats = market.market_stats
                return float(stats.mark_price or stats.last_price)
        except Exception:
            pass
        return self.cfg.fallback_price

    async def market_order(self, symbol: str, side: str, size_tokens: float, reduce_only: bool = False) -> Fill:
        side_l = side.lower()
        if side_l not in {"buy", "sell"}:
            raise RuntimeError(f"extended invalid side={side}")
        self._ensure_client()
        assert self._client is not None
        assert self._order_side_enum is not None
        assert self._order_tpsl_type_enum is not None
        assert self._order_trigger_price_type_enum is not None
        assert self._order_price_type_enum is not None
        assert self._time_in_force_enum is not None
        assert self._order_tpsl_trigger_param_cls is not None

        market_name = self.cfg.extended_market_for(symbol)
        if market_name not in self._markets:
            markets = await self._client.markets_info.get_markets_dict()
            self._markets[market_name] = markets.get(market_name)
        market = self._markets.get(market_name)
        if market is None:
            raise RuntimeError(f"extended market not found: {market_name}")

        stats = market.market_stats
        tc = market.trading_config
        ref_px = stats.ask_price if side_l == "buy" else stats.bid_price
        if ref_px <= 0:
            ref_px = stats.mark_price if stats.mark_price > 0 else stats.last_price
        if ref_px <= 0:
            ref_px = Decimal(str(self.cfg.fallback_price))

        raw_px = ref_px * Decimal(
            str(1.0 + self.slippage_pct / 100.0 if side_l == "buy" else 1.0 - self.slippage_pct / 100.0)
        )
        # Avoid custom cap/floor interpretation errors: exchange validates boundaries itself.
        px = tc.round_price(raw_px, ROUND_CEILING if side_l == "buy" else ROUND_FLOOR)

        qty_dec = Decimal(str(size_tokens))
        if qty_dec < tc.min_order_size:
            qty_dec = tc.min_order_size
        qty_dec = tc.round_order_size(qty_dec, rounding_direction=ROUND_FLOOR)
        qty = float(qty_dec)
        mid = float(ref_px)
        fee = (qty * mid) * (self.cfg.taker_fee_bps / 10_000)
        if self.cfg.dry_run:
            return Fill(self.name, symbol, side_l, qty, mid, fee, time.time(), order_id="extended-dry-run")

        side_enum = self._order_side_enum.BUY if side_l == "buy" else self._order_side_enum.SELL
        tp_sl_type = None
        tp_param = None
        sl_param = None
        if self.cfg.native_tpsl_enabled and (self.cfg.take_profit_pct > 0 or self.cfg.stop_loss_pct > 0) and not reduce_only:
            tp_price = ref_px * Decimal(str(1 + self.cfg.take_profit_pct / 100)) if side_l == "buy" else ref_px * Decimal(str(1 - self.cfg.take_profit_pct / 100))
            sl_price = ref_px * Decimal(str(1 - self.cfg.stop_loss_pct / 100)) if side_l == "buy" else ref_px * Decimal(str(1 + self.cfg.stop_loss_pct / 100))
            if self.cfg.take_profit_pct > 0:
                tp_r = tc.round_price(tp_price, ROUND_CEILING if side_l == "buy" else ROUND_FLOOR)
                tp_param = self._order_tpsl_trigger_param_cls(
                    trigger_price=tp_r,
                    trigger_price_type=self._order_trigger_price_type_enum.MARK,
                    price=tp_r,
                    price_type=self._order_price_type_enum.LIMIT,
                )
            if self.cfg.stop_loss_pct > 0:
                sl_r = tc.round_price(sl_price, ROUND_FLOOR if side_l == "buy" else ROUND_CEILING)
                sl_param = self._order_tpsl_trigger_param_cls(
                    trigger_price=sl_r,
                    trigger_price_type=self._order_trigger_price_type_enum.MARK,
                    price=sl_r,
                    price_type=self._order_price_type_enum.LIMIT,
                )
            tp_sl_type = self._order_tpsl_type_enum.ORDER
        async def _place_with_price(price_value: Decimal):
            return await self._client.place_order(
                market_name=market_name,
                amount_of_synthetic=qty_dec,
                price=price_value,
                side=side_enum,
                time_in_force=self._time_in_force_enum.IOC,
                reduce_only=reduce_only,
                tp_sl_type=tp_sl_type,
                take_profit=tp_param,
                stop_loss=sl_param,
            )

        try:
            placed = await _place_with_price(px)
        except Exception as e:
            msg = repr(e)
            if "code\":1141" in msg or "Invalid price value" in msg:
                # Price-ladder retry for transient price bound rejections.
                markets = await self._client.markets_info.get_markets_dict()
                self._markets[market_name] = markets.get(market_name)
                market2 = self._markets.get(market_name)
                if market2 is None:
                    raise RuntimeError(
                        f"extended place_order failed market={market_name} side={side_l} qty={qty_dec} price={px} ref={ref_px}"
                    ) from e
                s2 = market2.market_stats
                b2 = s2.bid_price if s2.bid_price > 0 else (s2.mark_price if s2.mark_price > 0 else s2.last_price)
                a2 = s2.ask_price if s2.ask_price > 0 else (s2.mark_price if s2.mark_price > 0 else s2.last_price)
                candidates = []
                if side_l == "buy":
                    candidates = [
                        tc.round_price(Decimal(str(a2)), ROUND_CEILING),
                        tc.round_price(Decimal(str(a2)) * Decimal("1.0015"), ROUND_CEILING),
                        tc.round_price(Decimal(str(a2)) * Decimal("1.0030"), ROUND_CEILING),
                    ]
                else:
                    candidates = [
                        tc.round_price(Decimal(str(b2)), ROUND_FLOOR),
                        tc.round_price(Decimal(str(b2)) * Decimal("0.9985"), ROUND_FLOOR),
                        tc.round_price(Decimal(str(b2)) * Decimal("0.9970"), ROUND_FLOOR),
                    ]
                last_exc = e
                placed = None
                for cpx in candidates:
                    try:
                        logger.warning(
                            "EXTENDED_RETRY_PRICE | symbol=%s side=%s old_px=%s retry_px=%s",
                            symbol,
                            side_l,
                            px,
                            cpx,
                        )
                        placed = await _place_with_price(cpx)
                        px = cpx
                        break
                    except Exception as e2:
                        last_exc = e2
                if placed is None:
                    raise RuntimeError(
                        f"extended place_order failed market={market_name} side={side_l} qty={qty_dec} price={px} ref={ref_px}"
                    ) from last_exc
            else:
                raise RuntimeError(
                    f"extended place_order failed market={market_name} side={side_l} qty={qty_dec} price={px} ref={ref_px}"
                ) from e
        oid = getattr(placed, "id", None)
        if oid is None and hasattr(placed, "data") and getattr(placed, "data") is not None:
            oid = getattr(getattr(placed, "data"), "id", None)
        return Fill(self.name, symbol, side_l, qty, mid, fee, time.time(), order_id=str(oid) if oid else None)

    async def close_market_order(self, open_fill: Fill) -> Fill:
        # Close by current live position state, not by assumed open leg,
        # because native TP/SL may have already flattened the position.
        cur_pos = await self.position_size(open_fill.symbol)
        if abs(cur_pos) < 0.0000001:
            px = await self.mid_price(open_fill.symbol)
            return Fill(
                exchange=self.name,
                symbol=open_fill.symbol,
                side="buy" if open_fill.side == "sell" else "sell",
                qty=open_fill.qty,
                price=px,
                fee_usd=0.0,
                ts=time.time(),
                order_id="extended-already-flat",
            )
        close_side = "sell" if cur_pos > 0 else "buy"
        close_qty = abs(cur_pos)
        try:
            return await self.market_order(open_fill.symbol, close_side, close_qty, reduce_only=True)
        except Exception as e:
            # Exchange may report missing reduce-only position due to race with TP/SL fills.
            if "Position is missing for reduce-only order" in repr(e):
                px = await self.mid_price(open_fill.symbol)
                return Fill(
                    exchange=self.name,
                    symbol=open_fill.symbol,
                    side=close_side,
                    qty=open_fill.qty,
                    price=px,
                    fee_usd=0.0,
                    ts=time.time(),
                    order_id="extended-flat-race",
                )
            raise

    async def place_native_tpsl(self, open_fill: Fill, tp_pct: float, sl_pct: float) -> None:
        # Extended TP/SL is attached natively during open order placement.
        return None

    async def position_size(self, symbol: str) -> float:
        if self.cfg.dry_run:
            return 0.0
        self._ensure_client()
        assert self._client is not None
        market_name = self.cfg.extended_market_for(symbol)
        resp = await self._client.account.get_positions(market_names=[market_name])
        positions = resp.data or []
        net = Decimal("0")
        for p in positions:
            if str(getattr(p, "market", "")) != market_name:
                continue
            side = str(getattr(p, "side", "")).upper()
            size = Decimal(str(getattr(p, "size", 0)))
            if side == "SHORT":
                net -= abs(size)
            else:
                net += abs(size)
        return float(net)

    async def normalize_order_size(self, symbol: str, size_tokens: float) -> float:
        self._ensure_client()
        assert self._client is not None
        market_name = self.cfg.extended_market_for(symbol)
        if market_name not in self._markets:
            markets = await self._client.markets_info.get_markets_dict()
            self._markets[market_name] = markets.get(market_name)
        market = self._markets.get(market_name)
        if market is None:
            raise RuntimeError(f"extended market not found: {market_name}")
        tc = market.trading_config
        qty_dec = Decimal(str(size_tokens))
        if qty_dec < tc.min_order_size:
            qty_dec = tc.min_order_size
        qty_dec = tc.round_order_size(qty_dec, rounding_direction=ROUND_FLOOR)
        qty = float(qty_dec)
        return max(qty, 0.0000001)

    async def account_balance_usd(self) -> float:
        if self.cfg.dry_run:
            return float(self.cfg.fallback_price)
        self._ensure_client()
        assert self._client is not None
        bal = await self._client.account.get_balance()
        data = bal.data
        if data is None:
            return 0.0
        return float(getattr(data, "equity", 0.0))

    async def pre_cycle_cleanup(self, symbol: str) -> None:
        if self.cfg.dry_run:
            return
        self._ensure_client()
        assert self._client is not None
        market_name = self.cfg.extended_market_for(symbol)
        try:
            await self._client.orders.mass_cancel(markets=[market_name], cancel_all=False)
            logger.info("PRE_CLEAN | exchange=extended symbol=%s mass_cancel=ok", symbol)
        except Exception as e:
            logger.warning("PRE_CLEAN_WARN | exchange=extended symbol=%s mass_cancel_err=%s", symbol, repr(e))


def pnl_for_leg(open_fill: Fill, close_fill: Fill) -> float:
    if open_fill.side == "buy":
        return (close_fill.price - open_fill.price) * open_fill.qty
    return (open_fill.price - close_fill.price) * open_fill.qty


def funding_cost_usd(notional_usd: float, hold_seconds: int, funding_bps_per_hour: float) -> float:
    hourly = notional_usd * (funding_bps_per_hour / 10_000)
    return hourly * (hold_seconds / 3600)


async def run_cycle(
    cfg: BotConfig,
    long_ex: ExchangeAdapter,
    short_ex: ExchangeAdapter,
    cycle_no: int,
) -> tuple[float, list[LegResult]]:
    hold_seconds = random.randint(cfg.hold_seconds_min, cfg.hold_seconds_max)
    symbol = cfg.symbol_for_cycle(cycle_no)
    await asyncio.gather(
        long_ex.pre_cycle_cleanup(symbol),
        short_ex.pre_cycle_cleanup(symbol),
    )
    await asyncio.sleep(0.2)

    async def _flatten_to_zero_before_open(ex: ExchangeAdapter, *, symbol: str) -> None:
        pos = await ex.position_size(symbol)
        if abs(pos) < 0.0000001:
            return
        logger.warning("PRE_CLEAN | exchange=%s symbol=%s residual_pos=%.8f -> flattening", ex.name, symbol, pos)
        for _ in range(4):
            side = "sell" if pos > 0 else "buy"
            await ex.market_order(symbol, side, abs(pos), reduce_only=True)
            await asyncio.sleep(0.6)
            pos = await ex.position_size(symbol)
            if abs(pos) < 0.0000001:
                logger.info("PRE_CLEAN | exchange=%s symbol=%s flatten_ok", ex.name, symbol)
                return
        raise RuntimeError(f"pre-clean failed: {ex.name} residual_pos={pos:.8f}")

    await asyncio.gather(
        _flatten_to_zero_before_open(long_ex, symbol=symbol),
        _flatten_to_zero_before_open(short_ex, symbol=symbol),
    )

    px_first, px_second = await asyncio.gather(long_ex.mid_price(symbol), short_ex.mid_price(symbol))
    if px_first <= 0 or px_second <= 0:
        raise NonRetryableCycleError(
            f"invalid market price for symbol={symbol}: {long_ex.name}={px_first:.8f} {short_ex.name}={px_second:.8f}"
        )
    gap_pct = abs(px_first - px_second) / max((px_first + px_second) / 2.0, 0.0000001) * 100.0
    if gap_pct > max(cfg.max_cross_price_gap_pct, 0.1):
        raise NonRetryableCycleError(
            f"cross price gap too high for symbol={symbol}: "
            f"{long_ex.name}={px_first:.8f} {short_ex.name}={px_second:.8f} gap_pct={gap_pct:.2f}"
        )
    ref_mid = max((px_first + px_second) / 2.0, 0.0000001)
    mode = cfg.side_mix_mode
    if mode == "spread":
        spread_bps = ((px_first - px_second) / ref_mid) * 10_000
        if abs(spread_bps) < cfg.spread_min_bps:
            # If spread too small, keep previous deterministic behavior (alternate) to avoid noise.
            first_is_buy = cycle_no % 2 == 1
        else:
            # first exchange (extended) expensive -> short it; cheap -> long it.
            first_is_buy = px_first <= px_second
    elif mode == "fixed_extended_long":
        first_is_buy = True
    elif mode == "fixed_extended_short":
        first_is_buy = False
    elif mode == "random":
        first_is_buy = random.choice([True, False])
    else:
        # alternate: odd cycle -> first exchange buy, even cycle -> first exchange sell
        first_is_buy = cycle_no % 2 == 1
    first_side = "buy" if first_is_buy else "sell"
    second_side = "sell" if first_is_buy else "buy"
    first_role = "long" if first_is_buy else "short"
    second_role = "short" if first_is_buy else "long"

    prev_size = getattr(run_cycle, "_last_size_by_symbol", {}).get(symbol)
    requested_size = cfg.randomized_position_size_for_symbol(symbol, ref_mid, prev_size)
    norm_long, norm_short = await asyncio.gather(
        long_ex.normalize_order_size(symbol, requested_size),
        short_ex.normalize_order_size(symbol, requested_size),
    )
    if norm_long <= 0 or norm_short <= 0:
        raise NonRetryableCycleError(
            f"invalid normalized size for symbol={symbol}: {long_ex.name}={norm_long:.8f} {short_ex.name}={norm_short:.8f}"
        )
    size_ratio = min(norm_long, norm_short) / max(norm_long, norm_short)
    if size_ratio < 0.90:
        raise NonRetryableCycleError(
            f"incompatible size increments for symbol={symbol}: "
            f"{long_ex.name}={norm_long:.8f} {short_ex.name}={norm_short:.8f}"
        )
    size_tokens = max(min(norm_long, norm_short), 0.0000001)
    if not hasattr(run_cycle, "_last_size_by_symbol"):
        setattr(run_cycle, "_last_size_by_symbol", {})
    getattr(run_cycle, "_last_size_by_symbol")[symbol] = size_tokens
    spread_bps = ((px_first - px_second) / ref_mid) * 10_000
    logger.info(
        "OPEN | symbol=%s hold=%ss %s=%s %s=%s size=%.8f requested=%.8f norm_%s=%.8f norm_%s=%.8f dry_run=%s mix=%s px_%s=%.2f px_%s=%.2f spread_bps=%.2f",
        symbol,
        hold_seconds,
        first_role,
        long_ex.name,
        second_role,
        short_ex.name,
        size_tokens,
        requested_size,
        long_ex.name,
        norm_long,
        short_ex.name,
        norm_short,
        cfg.dry_run,
        mode,
        long_ex.name,
        px_first,
        short_ex.name,
        px_second,
        spread_bps,
    )
    entry_mid = ref_mid

    # Open sequenced for safety: first leg then hedge leg.
    pre_pos_long, pre_pos_short = await asyncio.gather(
        long_ex.position_size(symbol),
        short_ex.position_size(symbol),
    )

    min_expected_abs = max(size_tokens * max(cfg.min_fill_ratio, 0.05), 0.0000001)
    poll_sec = max(cfg.fill_confirm_poll_sec, 0.1)
    timeout_sec = max(cfg.fill_confirm_timeout_sec, poll_sec)

    async def _wait_leg_delta(
        ex: ExchangeAdapter,
        *,
        base_pos: float,
        expected_side: str,
    ) -> tuple[bool, float]:
        expected_sign = 1.0 if expected_side == "buy" else -1.0
        deadline = time.time() + timeout_sec
        last_delta = 0.0
        while True:
            cur = await ex.position_size(symbol)
            last_delta = cur - base_pos
            if (last_delta * expected_sign) > 0 and abs(last_delta) >= min_expected_abs:
                return True, last_delta
            if time.time() >= deadline:
                return False, last_delta
            await asyncio.sleep(poll_sec)

    async def _reduce_delta_to_flat(ex: ExchangeAdapter, delta: float) -> None:
        if abs(delta) < 0.0000001:
            return
        close_side = "sell" if delta > 0 else "buy"
        await ex.market_order(symbol, close_side, abs(delta), reduce_only=True)

    async def _flatten_to_base(ex: ExchangeAdapter, *, base_pos: float, tries: int = 4) -> float:
        # Reconcile to the pre-cycle position to avoid naked exposure in case of delayed fills.
        cur_delta = 0.0
        for _ in range(max(1, tries)):
            cur = await ex.position_size(symbol)
            cur_delta = cur - base_pos
            if abs(cur_delta) < 0.0000001:
                return cur_delta
            close_side = "sell" if cur_delta > 0 else "buy"
            await ex.market_order(symbol, close_side, abs(cur_delta), reduce_only=True)
            await asyncio.sleep(max(0.2, poll_sec))
        cur = await ex.position_size(symbol)
        return cur - base_pos

    open_long: Optional[Fill] = None
    open_short: Optional[Fill] = None
    try:
        open_long = await long_ex.market_order(symbol, first_side, size_tokens)
    except Exception as e:
        logger.error("OPEN_FAILED | long_err=%s short_err=none", repr(e))
        raise RuntimeError("paired open failed; cycle aborted") from e

    ok_first, delta_first = await _wait_leg_delta(long_ex, base_pos=pre_pos_long, expected_side=first_side)
    if not ok_first:
        # Grace window for eventual consistency/API lag before treating as failure.
        await asyncio.sleep(max(1.0, poll_sec * 2))
        late_ok_first, late_delta_first = await _wait_leg_delta(
            long_ex,
            base_pos=pre_pos_long,
            expected_side=first_side,
        )
        if late_ok_first:
            ok_first = True
            delta_first = late_delta_first
    if not ok_first:
        logger.error(
            "OPEN_VERIFY_FAILED | exchange=%s expected_side=%s expected_min=%.8f delta=%.8f",
            long_ex.name,
            first_side,
            min_expected_abs,
            delta_first,
        )
        try:
            remain = await _flatten_to_base(long_ex, base_pos=pre_pos_long)
            if abs(remain) >= 0.0000001:
                logger.error(
                    "COMPENSATION_FAILED | could not fully flatten first leg residual=%.8f",
                    remain,
                )
            else:
                logger.warning("COMPENSATION | flattened first leg after failed first-leg open verification")
        except Exception as close_err:
            logger.error("COMPENSATION_FAILED | could not flatten first leg: %s", close_err)
        raise RuntimeError("paired open failed; first leg did not fill as expected")

    try:
        open_short = await short_ex.market_order(symbol, second_side, size_tokens)
    except Exception as e:
        logger.error("OPEN_FAILED | long_err=none short_err=%s", repr(e))
        try:
            await long_ex.close_market_order(open_long)
            logger.warning("COMPENSATION | closed opened long leg after paired open failure")
        except Exception as close_err:
            logger.error("COMPENSATION_FAILED | could not close long leg: %s", close_err)
        raise RuntimeError("paired open failed; cycle aborted") from e

    ok_second, delta_second = await _wait_leg_delta(short_ex, base_pos=pre_pos_short, expected_side=second_side)
    if not ok_second:
        await asyncio.sleep(max(1.0, poll_sec * 2))
        late_ok_second, late_delta_second = await _wait_leg_delta(
            short_ex,
            base_pos=pre_pos_short,
            expected_side=second_side,
        )
        if late_ok_second:
            ok_second = True
            delta_second = late_delta_second
    if not ok_second:
        logger.error(
            "OPEN_VERIFY_FAILED | exchange=%s expected_side=%s expected_min=%.8f delta=%.8f",
            short_ex.name,
            second_side,
            min_expected_abs,
            delta_second,
        )
        try:
            remain_long, remain_short = await asyncio.gather(
                _flatten_to_base(long_ex, base_pos=pre_pos_long),
                _flatten_to_base(short_ex, base_pos=pre_pos_short),
            )
            if abs(remain_long) >= 0.0000001 or abs(remain_short) >= 0.0000001:
                logger.error(
                    "COMPENSATION_FAILED | residuals after second-leg verification fail: long=%.8f short=%.8f",
                    remain_long,
                    remain_short,
                )
            else:
                logger.warning("COMPENSATION | flattened both legs after failed second-leg open verification")
        except Exception as close_err:
            logger.error("COMPENSATION_FAILED | could not flatten after failed second-leg verification: %s", close_err)
        raise RuntimeError("paired open failed; second leg did not fill as expected")

    exit_reason = "time"
    tp = cfg.take_profit_pct / 100.0
    sl = cfg.stop_loss_pct / 100.0
    if cfg.native_tpsl_enabled:
        # Register native trigger orders on exchanges that need separate trigger placement.
        await asyncio.gather(
            long_ex.place_native_tpsl(open_long, cfg.take_profit_pct, cfg.stop_loss_pct),
            short_ex.place_native_tpsl(open_short, cfg.take_profit_pct, cfg.stop_loss_pct),
        )

    close_long: Fill
    close_short: Fill
    try:
        deadline = time.time() + hold_seconds
        last_progress_log = time.time()
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            # If one leg got closed by native TP/SL (or manually), close the opposite leg immediately.
            pos_long, pos_short = await asyncio.gather(
                long_ex.position_size(symbol),
                short_ex.position_size(symbol),
            )
            long_flat = abs(pos_long) < 0.0000001
            short_flat = abs(pos_short) < 0.0000001
            if long_flat != short_flat:
                exit_reason = "leg_flat_sync_close"
                logger.warning(
                    "SYNC_CLOSE | symbol=%s %s_pos=%.8f %s_pos=%.8f -> closing opposite leg",
                    symbol,
                    long_ex.name,
                    pos_long,
                    short_ex.name,
                    pos_short,
                )
                break
            if tp <= 0 and sl <= 0:
                await asyncio.sleep(max(0.2, min(cfg.price_check_interval_sec, remaining)))
                continue
            cur_long, cur_short = await asyncio.gather(
                long_ex.mid_price(symbol),
                short_ex.mid_price(symbol),
            )
            upnl_long = (
                (cur_long - open_long.price) * open_long.qty
                if open_long.side == "buy"
                else (open_long.price - cur_long) * open_long.qty
            )
            upnl_short = (
                (cur_short - open_short.price) * open_short.qty
                if open_short.side == "buy"
                else (open_short.price - cur_short) * open_short.qty
            )
            long_ret = (
                (cur_long - open_long.price) / max(open_long.price, 0.0000001)
                if open_long.side == "buy"
                else (open_long.price - cur_long) / max(open_long.price, 0.0000001)
            )
            short_ret = (
                (cur_short - open_short.price) / max(open_short.price, 0.0000001)
                if open_short.side == "buy"
                else (open_short.price - cur_short) / max(open_short.price, 0.0000001)
            )
            if tp > 0 and (long_ret >= tp or short_ret >= tp):
                exit_reason = "tp"
                break
            if sl > 0 and (long_ret <= -sl or short_ret <= -sl):
                exit_reason = "sl"
                break
            if time.time() - last_progress_log >= cfg.watch_interval_sec:
                logger.info(
                    "WATCH | symbol=%s remaining=%ss %s_upnl=%.4f %s_upnl=%.4f total_upnl=%.4f | long_ret=%.4f%% short_ret=%.4f%% | %s_balance=%.2f %s_balance=%.2f",
                    symbol,
                    int(max(remaining, 0)),
                    long_ex.name,
                    upnl_long,
                    short_ex.name,
                    upnl_short,
                    upnl_long + upnl_short,
                    long_ret * 100,
                    short_ret * 100,
                    long_ex.name,
                    await long_ex.account_balance_usd(),
                    short_ex.name,
                    await short_ex.account_balance_usd(),
                )
                last_progress_log = time.time()
            await asyncio.sleep(max(0.2, min(cfg.price_check_interval_sec, remaining)))

        close_long, close_short = await asyncio.gather(
            long_ex.close_market_order(open_long),
            short_ex.close_market_order(open_short),
        )
    except asyncio.CancelledError:
        logger.warning("SHUTDOWN | cancellation received, closing open legs...")
        close_results = await asyncio.gather(
            long_ex.close_market_order(open_long),
            short_ex.close_market_order(open_short),
            return_exceptions=True,
        )
        for idx, res in enumerate(close_results, start=1):
            if isinstance(res, Exception):
                logger.error("SHUTDOWN_CLOSE_FAILED | leg=%s err=%s", idx, repr(res))
        raise

    notional_est = size_tokens * max(entry_mid, 0.0001)
    funding_leg = funding_cost_usd(notional_est, hold_seconds, cfg.estimated_funding_bps_per_hour)

    long_result = LegResult(
        exchange=long_ex.name,
        realized_pnl_usd=pnl_for_leg(open_long, close_long),
        fees_usd=open_long.fee_usd + close_long.fee_usd,
        funding_usd=funding_leg,
        traded_volume_usd=(open_long.qty * open_long.price) + (close_long.qty * close_long.price),
    )
    short_result = LegResult(
        exchange=short_ex.name,
        realized_pnl_usd=pnl_for_leg(open_short, close_short),
        fees_usd=open_short.fee_usd + close_short.fee_usd,
        funding_usd=funding_leg,
        traded_volume_usd=(open_short.qty * open_short.price) + (close_short.qty * close_short.price),
    )

    total = (
        long_result.realized_pnl_usd
        + short_result.realized_pnl_usd
        - long_result.fees_usd
        - short_result.fees_usd
        - long_result.funding_usd
        - short_result.funding_usd
    )
    logger.info("EXIT | symbol=%s reason=%s", symbol, exit_reason)
    return total, [long_result, short_result]


async def _validate_symbols(
    cfg: BotConfig,
    long_ex: ExchangeAdapter,
    short_ex: ExchangeAdapter,
) -> list[str]:
    active: list[str] = []
    for symbol in cfg.symbols:
        try:
            px_long, px_short = await asyncio.gather(
                long_ex.mid_price(symbol),
                short_ex.mid_price(symbol),
            )
            if px_long <= 0 or px_short <= 0:
                logger.error(
                    "SYMBOL_DISABLED | symbol=%s reason=invalid_price %s=%.8f %s=%.8f",
                    symbol,
                    long_ex.name,
                    px_long,
                    short_ex.name,
                    px_short,
                )
                continue
            gap_pct = abs(px_long - px_short) / max((px_long + px_short) / 2.0, 0.0000001) * 100.0
            if gap_pct > max(cfg.max_cross_price_gap_pct, 0.1):
                logger.error(
                    "SYMBOL_DISABLED | symbol=%s reason=cross_price_gap %s=%.8f %s=%.8f gap_pct=%.2f limit_pct=%.2f",
                    symbol,
                    long_ex.name,
                    px_long,
                    short_ex.name,
                    px_short,
                    gap_pct,
                    cfg.max_cross_price_gap_pct,
                )
                continue
            ref_mid = max((px_long + px_short) / 2.0, 0.0000001)
            requested = cfg.position_size_for_symbol(symbol, ref_mid)
            norm_long, norm_short = await asyncio.gather(
                long_ex.normalize_order_size(symbol, requested),
                short_ex.normalize_order_size(symbol, requested),
            )
            if norm_long <= 0 or norm_short <= 0:
                logger.error(
                    "SYMBOL_DISABLED | symbol=%s reason=invalid_norm_size %s=%.8f %s=%.8f",
                    symbol,
                    long_ex.name,
                    norm_long,
                    short_ex.name,
                    norm_short,
                )
                continue
            ratio = min(norm_long, norm_short) / max(norm_long, norm_short)
            # Different step sizes are normal across exchanges. Disable only when
            # mismatch is extreme (likely wrong product mapping), otherwise allow.
            if ratio < 0.10:
                logger.error(
                    "SYMBOL_DISABLED | symbol=%s reason=incompatible_increments %s=%.8f %s=%.8f ratio=%.4f",
                    symbol,
                    long_ex.name,
                    norm_long,
                    short_ex.name,
                    norm_short,
                    ratio,
                )
                continue
            if ratio < 0.90:
                logger.warning(
                    "SYMBOL_INCREMENT_MISMATCH | symbol=%s %s=%.8f %s=%.8f ratio=%.4f",
                    symbol,
                    long_ex.name,
                    norm_long,
                    short_ex.name,
                    norm_short,
                    ratio,
                )
            active.append(symbol)
            logger.info(
                "SYMBOL_ENABLED | symbol=%s %s_px=%.2f %s_px=%.2f requested=%.8f norm_%s=%.8f norm_%s=%.8f",
                symbol,
                long_ex.name,
                px_long,
                short_ex.name,
                px_short,
                requested,
                long_ex.name,
                norm_long,
                short_ex.name,
                norm_short,
            )
        except Exception as e:
            logger.error("SYMBOL_DISABLED | symbol=%s reason=exception err=%s", symbol, repr(e))
    return active


async def _build_adapters(cfg: BotConfig) -> tuple[ExchangeAdapter, ExchangeAdapter, list[Any]]:
    closers: list[Any] = []
    if cfg.execution_mode == "live":
        long_ex: ExchangeAdapter
        long_closer = None
        if cfg.long_exchange == "extended":
            extended = ExtendedLiveExchange(cfg)
            long_ex = extended
            long_closer = extended.close
        else:
            raise RuntimeError("Unsupported LONG_EXCHANGE. Use LONG_EXCHANGE=extended.")
        try:
            nado = NadoLiveExchange(cfg)
        except Exception:
            if long_closer:
                await long_closer()
            raise
        if long_closer:
            closers.append(long_closer)
        closers.append(nado.close)
        return long_ex, nado, closers

    extended = TinyLiveStubExchange("extended", cfg, seed_offset=1)
    nado = TinyLiveStubExchange("nado", cfg, seed_offset=2)
    return extended, nado, closers


async def main() -> None:
    cfg = BotConfig.from_env()
    requested_symbols = list(cfg.symbols)
    if cfg.hold_seconds_min <= 0 or cfg.hold_seconds_max < cfg.hold_seconds_min:
        raise RuntimeError("Invalid hold settings in env")
    if cfg.position_size_tokens <= 0 and cfg.position_notional_usd <= 0 and not cfg.position_size_tokens_by_symbol:
        raise RuntimeError("Set POSITION_SIZE_TOKENS (>0) or legacy POSITION_NOTIONAL_USD (>0)")

    long_ex, short_ex, closers = await _build_adapters(cfg)
    active_symbols = await _validate_symbols(cfg, long_ex, short_ex)
    if not active_symbols:
        for closer in closers:
            try:
                await closer()
            except Exception:
                pass
        raise RuntimeError("No tradable symbols after validation. Check product IDs/markets/size increments.")
    if cfg.require_first_symbol_success and requested_symbols:
        first_requested = requested_symbols[0]
        if first_requested not in active_symbols:
            for closer in closers:
                try:
                    await closer()
                except Exception:
                    pass
            raise RuntimeError(
                f"First symbol gate failed: `{first_requested}` is not tradable now. "
                "Bot stopped to prevent fallback to later symbols."
            )
    if set(active_symbols) != set(cfg.symbols):
        logger.warning("SYMBOL_SET_UPDATED | requested=%s active=%s", ",".join(cfg.symbols), ",".join(active_symbols))
    cfg = dataclasses.replace(cfg, symbols=active_symbols)

    date_key = datetime.now(UTC).date().isoformat()
    daily_pnl = 0.0
    virtual_balance_usd = float(os.getenv("START_BALANCE_USD", "50"))
    exchange_stats: dict[str, dict[str, float]] = {}
    exchange_last_balance: dict[str, float] = {}
    completed_cycles = 0
    failed_cycles = 0

    try:
        start_long_balance, start_short_balance = await asyncio.gather(
            long_ex.account_balance_usd(),
            short_ex.account_balance_usd(),
        )
    except Exception as e:
        logger.warning("START_BALANCE_FETCH_WARN | reason=%s", repr(e))
        start_long_balance, start_short_balance = 0.0, 0.0

    # Use real exchange balances as the primary portfolio baseline for logs.
    # Keep virtual_balance_usd only for internal compatibility with existing daily_pnl flow.
    portfolio_start_balance = start_long_balance + start_short_balance
    virtual_balance_usd = portfolio_start_balance if portfolio_start_balance > 0 else virtual_balance_usd
    exchange_last_balance[long_ex.name] = start_long_balance
    exchange_last_balance[short_ex.name] = start_short_balance

    logger.info(
        "START | date=%s symbols=%s mode=%s dry_run=%s %s_balance=%.2f %s_balance=%.2f",
        date_key,
        ",".join(cfg.symbols),
        cfg.execution_mode,
        cfg.dry_run,
        long_ex.name,
        start_long_balance,
        short_ex.name,
        start_short_balance,
    )
    logger.info(
        "RISK | max_daily_loss=%.2f size_tokens_default=%.8f legacy_notional=%.2f hold=%s..%ss tp=%.3f%% sl=%.3f%% native_tpsl=%s",
        cfg.max_daily_loss_usd,
        cfg.position_size_tokens,
        cfg.position_notional_usd,
        cfg.hold_seconds_min,
        cfg.hold_seconds_max,
        cfg.take_profit_pct,
        cfg.stop_loss_pct,
        cfg.native_tpsl_enabled,
    )

    try:
        for cycle in range(1, cfg.max_cycles + 1):
            if daily_pnl <= -abs(cfg.max_daily_loss_usd):
                logger.error("KILL_SWITCH | daily_pnl=%.4f reached max daily loss", daily_pnl)
                break

            total: float
            legs: list[LegResult]
            last_err: Optional[Exception] = None
            done = False
            for attempt in range(1, max(cfg.cycle_retry_count, 1) + 1):
                try:
                    total, legs = await run_cycle(cfg, long_ex, short_ex, cycle_no=cycle)
                    done = True
                    break
                except Exception as e:
                    last_err = e
                    logger.error(
                        "CYCLE_RETRY | cycle=%s attempt=%s/%s err=%s",
                        cycle,
                        attempt,
                        max(cfg.cycle_retry_count, 1),
                        repr(e),
                    )
                    if cycle == 1 and cfg.require_first_symbol_success:
                        logger.error(
                            "FIRST_SYMBOL_GATE | first cycle failed on required symbol; stopping bot to avoid fallback pairs"
                        )
                        break
                    if isinstance(e, NonRetryableCycleError):
                        break
                    if attempt < max(cfg.cycle_retry_count, 1):
                        await asyncio.sleep(max(cfg.cycle_retry_pause_seconds, 1))
            if not done:
                failed_cycles += 1
                logger.error("CYCLE_ABORTED | cycle=%s err=%s", cycle, repr(last_err))
                if cycle == 1 and cfg.require_first_symbol_success:
                    break
                await asyncio.sleep(cfg.cycle_pause_seconds)
                continue
            daily_pnl += total
            virtual_balance_usd += total
            completed_cycles += 1

            leg1, leg2 = legs
            for leg in legs:
                bucket = exchange_stats.setdefault(
                    leg.exchange,
                    {
                        "volume_usd": 0.0,
                        "realized_pnl_usd": 0.0,
                        "fees_usd": 0.0,
                        "funding_usd": 0.0,
                    },
                )
                bucket["volume_usd"] += leg.traded_volume_usd
                bucket["realized_pnl_usd"] += leg.realized_pnl_usd
                bucket["fees_usd"] += leg.fees_usd
                bucket["funding_usd"] += leg.funding_usd
            b_long, b_short = await asyncio.gather(long_ex.account_balance_usd(), short_ex.account_balance_usd())
            exchange_last_balance[long_ex.name] = b_long
            exchange_last_balance[short_ex.name] = b_short
            portfolio_balance = b_long + b_short
            logger.info(
                "CYCLE %s | total=%.4f portfolio=%.4f daily=%.4f | %s pnl=%.4f fees=%.4f funding=%.4f vol=%.2f bal=%.2f | %s pnl=%.4f fees=%.4f funding=%.4f vol=%.2f bal=%.2f",
                cycle,
                total,
                portfolio_balance,
                daily_pnl,
                leg1.exchange,
                leg1.realized_pnl_usd,
                leg1.fees_usd,
                leg1.funding_usd,
                leg1.traded_volume_usd,
                b_long if leg1.exchange == long_ex.name else b_short,
                leg2.exchange,
                leg2.realized_pnl_usd,
                leg2.fees_usd,
                leg2.funding_usd,
                leg2.traded_volume_usd,
                b_long if leg2.exchange == long_ex.name else b_short,
            )
            await asyncio.sleep(cfg.cycle_pause_seconds)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.warning("SHUTDOWN | stop signal received")
    finally:
        for closer in closers:
            try:
                await closer()
            except Exception:
                pass

    total_volume = 0.0
    logger.info("SUMMARY | cycles_completed=%s cycles_failed=%s", completed_cycles, failed_cycles)
    for exchange, s in sorted(exchange_stats.items()):
        net = s["realized_pnl_usd"] - s["fees_usd"] - s["funding_usd"]
        total_volume += s["volume_usd"]
        logger.info(
            "SUMMARY | exchange=%s volume_usd=%.2f realized_pnl=%.4f fees=%.4f funding=%.4f net=%.4f balance=%.2f",
            exchange,
            s["volume_usd"],
            s["realized_pnl_usd"],
            s["fees_usd"],
            s["funding_usd"],
            net,
            exchange_last_balance.get(exchange, 0.0),
        )
    logger.info("SUMMARY | total_volume_usd=%.2f", total_volume)
    final_long = exchange_last_balance.get(long_ex.name, 0.0)
    final_short = exchange_last_balance.get(short_ex.name, 0.0)
    logger.info(
        "STOP | final_%s_balance=%.4f final_%s_balance=%.4f final_total_balance=%.4f daily_pnl=%.4f",
        long_ex.name,
        final_long,
        short_ex.name,
        final_short,
        final_long + final_short,
        daily_pnl,
    )


if __name__ == "__main__":
    asyncio.run(main())

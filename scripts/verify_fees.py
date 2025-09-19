#!/usr/bin/env python3
"""Compare simulated fees with reference commission ticks.

The script wires together :class:`impl_fees.FeesImpl` and
:class:`impl_quantizer.QuantizerImpl`, samples random trades that satisfy Binance
filters, and checks whether the simulator fee matches the reference value produced by
rounding to the commission step.  Differences larger than a single commission tick are
reported.

Examples
--------
Validate a couple of popular spot pairs with 200 samples each::

    python scripts/verify_fees.py --symbols BTCUSDT ETHUSDT --samples 200

Run the same check using BNB settlement with an explicit conversion rate::

    python scripts/verify_fees.py --symbols BTCUSDT --settlement-mode bnb --bnb-price 280

"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from impl_fees import FeesImpl
from impl_quantizer import QuantizerImpl


LOGGER = logging.getLogger("verify_fees")


@dataclass
class TradeSample:
    symbol: str
    side: str
    liquidity: str
    price: float
    qty: float
    notional: float


@dataclass
class Discrepancy:
    sample: TradeSample
    simulated_fee: float
    reference_fee: float
    commission_step: float


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare simulator fee computation with reference rounding ticks.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to validate. Defaults to BTCUSDT, ETHUSDT, BNBUSDT if omitted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples per symbol (default: 200).",
    )
    parser.add_argument(
        "--settlement-mode",
        choices=["quote", "bnb"],
        default="quote",
        help="Settlement mode to enforce when building the fee model (default: quote).",
    )
    parser.add_argument(
        "--bnb-price",
        type=float,
        default=None,
        help="BNB/quote conversion rate to use when --settlement-mode=bnb.",
    )
    parser.add_argument(
        "--filters",
        type=Path,
        default=None,
        help="Optional path to Binance filters JSON (defaults to data/binance_filters.json).",
    )
    parser.add_argument(
        "--fees-table",
        type=Path,
        default=None,
        help="Optional path to fees table (defaults to data/fees/fees_by_symbol.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def _default_filters_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "binance_filters.json"


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")


def _ceil_to_step(value: float, step: float) -> float:
    if step <= 0.0:
        return float(value)
    units = math.ceil((float(value) - 1e-15) / float(step))
    return float(units * float(step))


def _select_symbols(all_symbols: Sequence[str], requested: Optional[Iterable[str]]) -> List[str]:
    if requested:
        request_list = list(requested)
        if len(request_list) == 1 and request_list[0].strip().upper() == "ALL":
            return list(all_symbols)
        normalized: List[str] = []
        available = {symbol.upper() for symbol in all_symbols}
        for symbol in request_list:
            key = symbol.upper()
            if key in available:
                normalized.append(key)
            else:
                LOGGER.warning("Symbol %s is not available in filters; skipping", symbol)
        return normalized
    if not all_symbols:
        return []
    defaults = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    return [sym for sym in defaults if sym in all_symbols] or list(all_symbols[:3])


def _resolve_rounding_step(details, quantizer, symbol: str) -> float:
    candidates: List[Optional[float]] = []
    if getattr(details, "commission_step", None) is not None:
        candidates.append(float(details.commission_step))
    if getattr(details, "rounding_step", None) is not None:
        candidates.append(float(details.rounding_step))
    if quantizer is not None:
        try:
            step = quantizer.get_commission_step(symbol)
        except Exception:  # pragma: no cover - defensive
            step = None
        if step is not None:
            candidates.append(float(step))
    for step in candidates:
        if step is not None and step > 0.0:
            return float(step)
    return 0.0


def _log_symbol_steps(symbol: str, quantizer, fees_impl: FeesImpl) -> None:
    quant_step = None
    if quantizer is not None:
        try:
            quant_step = float(quantizer.get_commission_step(symbol))
        except Exception:
            quant_step = None
    table = fees_impl.model_payload.get("symbol_fee_table") or {}
    payload = table.get(symbol) or {}
    payload_step = None
    quant_payload_step = None
    if isinstance(payload, dict):
        payload_step = payload.get("commission_step")
        quant_payload = payload.get("quantizer")
        if isinstance(quant_payload, dict):
            quant_payload_step = quant_payload.get("commission_step")
    LOGGER.info(
        "Symbol %s commission steps: quantizer=%s payload=%s quantizer_payload=%s",
        symbol,
        f"{quant_step:.10f}" if quant_step else "n/a",
        f"{float(payload_step):.10f}" if payload_step else "n/a",
        f"{float(quant_payload_step):.10f}" if quant_payload_step else "n/a",
    )


def _resolve_filters(quantizer_impl: QuantizerImpl):
    quantizer = quantizer_impl.quantizer
    if quantizer is None:
        raise RuntimeError("Quantizer is unavailable; ensure filters file exists")
    filters = getattr(quantizer, "_filters", None)
    if not isinstance(filters, dict) or not filters:
        raise RuntimeError("Quantizer has no filters loaded")
    return quantizer, filters


def _sample_trade(
    symbol: str,
    filters,
    quantizer,
    rng: random.Random,
) -> TradeSample:
    sym_filters = filters[symbol]
    for _ in range(1000):
        price_min = max(sym_filters.price_min, sym_filters.price_tick, 1e-8)
        price_max = sym_filters.price_max
        if not math.isfinite(price_max) or price_max <= price_min:
            price_max = price_min * 1000.0
        price = math.exp(rng.uniform(math.log(price_min), math.log(price_max)))
        price = float(quantizer.quantize_price(symbol, price))
        if price <= 0.0:
            continue

        min_notional = max(sym_filters.min_notional, price * sym_filters.qty_min)
        min_notional = max(min_notional, price * sym_filters.qty_step, price * 1e-8)
        target_notional = min_notional * rng.uniform(1.05, 10.0)
        qty_raw = target_notional / price
        qty_min = max(sym_filters.qty_min, sym_filters.qty_step, 1e-8)
        qty = max(qty_raw, qty_min)
        qty = float(quantizer.quantize_qty(symbol, qty))
        if qty <= 0.0:
            continue
        notional = price * qty
        if notional < sym_filters.min_notional:
            adjustment = qty + max(sym_filters.qty_step, qty_min)
            qty = float(quantizer.quantize_qty(symbol, adjustment))
            notional = price * qty
        if qty <= 0.0 or notional <= 0.0:
            continue
        side = "BUY" if rng.random() < 0.5 else "SELL"
        liquidity = "maker" if rng.random() < 0.5 else "taker"
        return TradeSample(
            symbol=symbol,
            side=side,
            liquidity=liquidity,
            price=price,
            qty=qty,
            notional=notional,
        )
    raise RuntimeError(f"Failed to generate valid trade for {symbol}")


def _build_fees_impl(args: argparse.Namespace) -> FeesImpl:
    cfg = {}
    if args.fees_table:
        cfg["path"] = str(args.fees_table)
    if args.settlement_mode == "bnb":
        cfg["settlement"] = {"mode": "bnb", "currency": "BNB"}
    return FeesImpl.from_dict(cfg)


def _attach_quantizer(fees: FeesImpl, quantizer, symbols: Sequence[str]) -> None:
    if quantizer is None:
        return
    from types import SimpleNamespace

    sim = SimpleNamespace(quantizer=quantizer, symbol=None)
    if not symbols:
        fees.attach_to(sim)
        return
    for symbol in symbols:
        sim.symbol = symbol
        fees.attach_to(sim)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    if args.seed is not None:
        random.seed(args.seed)

    filters_path = args.filters or _default_filters_path()
    if not filters_path.exists():
        raise SystemExit(f"Filters file not found: {filters_path}")

    quantizer_impl = QuantizerImpl.from_dict({"path": str(filters_path)})
    quantizer, filters = _resolve_filters(quantizer_impl)

    available_symbols = tuple(sorted(filters))
    symbols = _select_symbols(available_symbols, args.symbols)
    if not symbols:
        raise SystemExit("No symbols resolved for sampling")

    fees_impl = _build_fees_impl(args)
    _attach_quantizer(fees_impl, quantizer, symbols)
    model = fees_impl.model
    if model is None:
        raise SystemExit("Fees model is not available (FeesModel import failed)")

    if args.samples <= 0:
        raise SystemExit("--samples must be a positive integer")
    if args.settlement_mode == "bnb" and args.bnb_price is None:
        LOGGER.warning(
            "BNB settlement selected without conversion rate; fees stay in quote currency",
        )

    LOGGER.info("Sampling %s trades per symbol", args.samples)

    mismatches: List[Discrepancy] = []
    for symbol in symbols:
        _log_symbol_steps(symbol, quantizer, fees_impl)
        for _ in range(args.samples):
            sample = _sample_trade(symbol, filters, quantizer, random)
            kwargs = dict(
                side=sample.side,
                price=sample.price,
                qty=sample.qty,
                liquidity=sample.liquidity,
                symbol=sample.symbol,
                return_details=True,
            )
            if args.settlement_mode == "bnb" and args.bnb_price:
                kwargs["bnb_conversion_rate"] = float(args.bnb_price)
            details = model.compute(**kwargs)
            step = _resolve_rounding_step(details, quantizer, symbol)
            reference = _ceil_to_step(details.fee_before_rounding, step)
            diff = abs(details.fee - reference)
            threshold = step if step > 0.0 else 0.0
            if diff > threshold + 1e-12:
                mismatches.append(
                    Discrepancy(
                        sample=sample,
                        simulated_fee=float(details.fee),
                        reference_fee=float(reference),
                        commission_step=float(step),
                    )
                )
                LOGGER.warning(
                    "Mismatch %s %s %s: sim=%s reference=%s diff=%s step=%s price=%s qty=%s",
                    sample.symbol,
                    sample.side,
                    sample.liquidity,
                    details.fee,
                    reference,
                    diff,
                    step,
                    sample.price,
                    sample.qty,
                )

    if mismatches:
        LOGGER.warning("Detected %s discrepancies above one commission tick", len(mismatches))
        return 1

    LOGGER.info("No discrepancies exceeding one commission tick detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

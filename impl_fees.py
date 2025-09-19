# -*- coding: utf-8 -*-
"""Helpers for plugging the :mod:`fees` module into runtime components."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, Tuple

try:
    from fees import FeesModel
except Exception:  # pragma: no cover
    FeesModel = None  # type: ignore

from services.costs import MakerTakerShareSettings


logger = logging.getLogger(__name__)

_FEE_TABLE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_DEFAULT_FEE_TABLE_PATH = Path("data") / "fees" / "fees_by_symbol.json"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not (num == num) or num in (float("inf"), float("-inf")):
        return default
    return num


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    return num


def _normalise_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        candidate = str(value).strip()
    except Exception:
        return None
    if not candidate:
        return None
    return os.path.expanduser(candidate)


def _plain_mapping(data: Any) -> Dict[str, Any]:
    if isinstance(data, Mapping):
        return {k: v for k, v in data.items()}
    return {}


@dataclass
class FeesConfig:
    """Normalised configuration for :class:`FeesImpl`."""

    enabled: bool = True
    path: Optional[str] = None
    refresh_days: Optional[int] = None
    maker_bps: float = 1.0
    taker_bps: float = 5.0
    use_bnb_discount: bool = False
    maker_discount_mult: Optional[float] = None
    taker_discount_mult: Optional[float] = None
    vip_tier: Optional[int] = None
    fee_rounding_step: Optional[float] = None
    symbol_fee_table: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    maker_taker_share: Optional[Dict[str, Any]] = None
    maker_taker_share_enabled: Optional[bool] = None
    maker_taker_share_mode: Optional[str] = None
    maker_share_default: Optional[float] = None
    spread_cost_maker_bps: Optional[float] = None
    spread_cost_taker_bps: Optional[float] = None
    taker_fee_override_bps: Optional[float] = None

    # filled during normalisation
    maker_taker_share_cfg: Optional[MakerTakerShareSettings] = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.path = _normalise_path(self.path)
        self.refresh_days = _safe_positive_int(self.refresh_days)

        maker_bps = _safe_float(self.maker_bps, 1.0)
        self.maker_bps = maker_bps if maker_bps is not None else 1.0
        taker_bps = _safe_float(self.taker_bps, 5.0)
        self.taker_bps = taker_bps if taker_bps is not None else 5.0

        self.use_bnb_discount = bool(self.use_bnb_discount)
        maker_mult = _safe_float(self.maker_discount_mult)
        taker_mult = _safe_float(self.taker_discount_mult)
        if self.use_bnb_discount:
            if maker_mult is None:
                maker_mult = 0.75
            if taker_mult is None:
                taker_mult = 0.75
        else:
            if maker_mult is None:
                maker_mult = 1.0
            if taker_mult is None:
                taker_mult = 1.0
        self.maker_discount_mult = maker_mult
        self.taker_discount_mult = taker_mult

        self.vip_tier = _safe_positive_int(self.vip_tier)

        step = _safe_float(self.fee_rounding_step)
        if step is not None and step < 0.0:
            step = 0.0
        self.fee_rounding_step = step

        if isinstance(self.symbol_fee_table, Mapping):
            table: Dict[str, Any] = {}
            for symbol, payload in self.symbol_fee_table.items():
                if not isinstance(symbol, str):
                    continue
                if isinstance(payload, Mapping):
                    table[symbol.upper()] = dict(payload)
            self.symbol_fee_table = table
        else:
            self.symbol_fee_table = {}

        if isinstance(self.metadata, Mapping):
            self.metadata = dict(self.metadata)
        else:
            self.metadata = {}

        share_payload: Dict[str, Any] = {}
        if isinstance(self.maker_taker_share, Mapping):
            share_payload.update(self.maker_taker_share)

        overrides = {
            "enabled": self.maker_taker_share_enabled,
            "mode": self.maker_taker_share_mode,
            "maker_share_default": self.maker_share_default,
            "spread_cost_maker_bps": self.spread_cost_maker_bps,
            "spread_cost_taker_bps": self.spread_cost_taker_bps,
            "taker_fee_override_bps": self.taker_fee_override_bps,
        }
        for key, value in overrides.items():
            if value is not None:
                share_payload.setdefault(key, value)

        share_cfg = MakerTakerShareSettings.parse(share_payload)
        self.maker_taker_share_cfg = share_cfg
        if share_cfg is not None:
            share_dict = share_cfg.as_dict()
            self.maker_taker_share = share_dict
            self.maker_taker_share_enabled = bool(share_cfg.enabled)
            self.maker_taker_share_mode = share_cfg.mode
            self.maker_share_default = float(share_cfg.maker_share_default)
            self.spread_cost_maker_bps = float(share_cfg.spread_cost_maker_bps)
            self.spread_cost_taker_bps = float(share_cfg.spread_cost_taker_bps)
            self.taker_fee_override_bps = (
                float(share_cfg.taker_fee_override_bps)
                if share_cfg.taker_fee_override_bps is not None
                else None
            )
        else:
            self.maker_taker_share = None
            if self.maker_taker_share_enabled is None:
                self.maker_taker_share_enabled = False
            if self.maker_taker_share_mode is None:
                self.maker_taker_share_mode = "fixed"
            if self.maker_share_default is None:
                self.maker_share_default = 0.5
            if self.spread_cost_maker_bps is None:
                self.spread_cost_maker_bps = 0.0
            if self.spread_cost_taker_bps is None:
                self.spread_cost_taker_bps = 0.0

        # keep an easily serialisable copy of overrides
        if self.maker_taker_share is None and share_payload:
            cleaned: Dict[str, Any] = {}
            for key, value in share_payload.items():
                if value is None:
                    continue
                cleaned[key] = value
            self.maker_taker_share = cleaned if cleaned else None


class FeesImpl:
    """Wrapper over :class:`fees.FeesModel` with simulator integration helpers."""

    def __init__(self, cfg: FeesConfig) -> None:
        self.cfg = cfg

        self.table_path: Optional[str] = None
        self.table_metadata: Dict[str, Any] = {}
        self.table_age_days: Optional[float] = None
        self.table_stale: bool = False
        self.table_error: Optional[str] = None
        self.symbol_fee_table_raw: Dict[str, Any] = {}
        self.inline_symbol_fee_table: Dict[str, Any] = dict(cfg.symbol_fee_table)
        self.symbol_fee_table: Dict[str, Any] = {}
        self._table_account_overrides: Dict[str, Any] = {}
        self._table_share_raw: Optional[Dict[str, Any]] = None

        table_payload = self._load_symbol_fee_table()
        table_from_file = table_payload.get("table", {}) if table_payload else {}
        if isinstance(table_from_file, Mapping):
            self.symbol_fee_table_raw = {
                str(symbol).upper(): dict(payload)
                for symbol, payload in table_from_file.items()
                if isinstance(symbol, str) and isinstance(payload, Mapping)
            }
        if table_payload:
            account_payload = _plain_mapping(table_payload.get("account"))
            if account_payload:
                self._table_account_overrides = account_payload
            share_payload = table_payload.get("share")
            if isinstance(share_payload, Mapping):
                self._table_share_raw = dict(share_payload)

        if not self.table_stale and self.table_error is None:
            self.symbol_fee_table.update(self.symbol_fee_table_raw)
        if self.inline_symbol_fee_table:
            for symbol, payload in self.inline_symbol_fee_table.items():
                if not isinstance(symbol, str):
                    continue
                if isinstance(payload, Mapping):
                    self.symbol_fee_table[symbol.upper()] = dict(payload)

        self.maker_taker_share_cfg: Optional[MakerTakerShareSettings]
        share_cfg = cfg.maker_taker_share_cfg
        share_raw: Optional[Dict[str, Any]] = None
        if share_cfg is not None:
            share_raw = share_cfg.as_dict()
        elif cfg.maker_taker_share is not None:
            share_raw = dict(cfg.maker_taker_share)
        if share_cfg is None and self._table_share_raw is not None:
            share_cfg = MakerTakerShareSettings.parse(self._table_share_raw)
            if share_cfg is None:
                share_raw = dict(self._table_share_raw)
        self.maker_taker_share_cfg = share_cfg
        self.maker_taker_share_raw = share_raw if share_cfg is None else share_cfg.as_dict()

        self._maker_discount_mult = float(cfg.maker_discount_mult)
        self._taker_discount_mult = float(cfg.taker_discount_mult)
        self._use_bnb_discount = bool(cfg.use_bnb_discount)

        fee_rounding_step = cfg.fee_rounding_step
        if fee_rounding_step is None:
            candidate = _safe_float(self._table_account_overrides.get("fee_rounding_step"))
            if candidate is not None and candidate >= 0.0:
                fee_rounding_step = candidate

        vip_tier = cfg.vip_tier
        if vip_tier is None:
            vip_candidate = _safe_positive_int(
                self._table_account_overrides.get("vip_tier")
            )
            if vip_candidate is not None:
                vip_tier = vip_candidate
        if vip_tier is None:
            vip_tier = 0

        maker_bps = float(cfg.maker_bps)
        taker_bps = float(cfg.taker_bps)

        self.base_fee_bps: Dict[str, float] = {
            "maker_fee_bps": maker_bps * self._maker_discount_mult,
            "taker_fee_bps": taker_bps * self._taker_discount_mult,
        }

        self.maker_taker_share_expected: Optional[Dict[str, float]] = None
        if self.maker_taker_share_cfg is not None:
            self.maker_taker_share_expected = (
                self.maker_taker_share_cfg.expected_fee_breakdown(
                    self.base_fee_bps["maker_fee_bps"],
                    self.base_fee_bps["taker_fee_bps"],
                )
            )

        self.expected_fee_bps: Dict[str, float] = dict(self.base_fee_bps)
        if self.maker_taker_share_expected is not None:
            self.expected_fee_bps.update(self.maker_taker_share_expected)

        symbol_table_payload = (
            {k: dict(v) for k, v in self.symbol_fee_table.items()}
            if self.symbol_fee_table
            else {}
        )

        self.model_payload: Dict[str, Any] = {
            "maker_bps": maker_bps,
            "taker_bps": taker_bps,
            "use_bnb_discount": self._use_bnb_discount,
            "maker_discount_mult": self._maker_discount_mult,
            "taker_discount_mult": self._taker_discount_mult,
            "vip_tier": int(vip_tier),
        }
        if fee_rounding_step is not None:
            self.model_payload["fee_rounding_step"] = float(fee_rounding_step)
        if symbol_table_payload:
            self.model_payload["symbol_fee_table"] = symbol_table_payload

        self._model = (
            FeesModel.from_dict(dict(self.model_payload))
            if FeesModel is not None and cfg.enabled
            else None
        )

        self.metadata = self._build_metadata(
            vip_tier=vip_tier,
            fee_rounding_step=fee_rounding_step,
        )

        self.expected_payload: Dict[str, Any] = self._build_expected_payload(
            vip_tier=vip_tier
        )

    def _build_metadata(
        self, *, vip_tier: int, fee_rounding_step: Optional[float]
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.cfg.metadata:
            meta.update(self.cfg.metadata)
        table_meta = dict(self.table_metadata)
        table_meta.setdefault("path", self.table_path)
        table_meta.setdefault("age_days", self.table_age_days)
        table_meta.setdefault("refresh_days", self.cfg.refresh_days)
        table_meta.setdefault("stale", self.table_stale)
        table_meta.setdefault("error", self.table_error)
        table_meta.setdefault("file_symbol_count", len(self.symbol_fee_table_raw))
        if self._table_account_overrides:
            table_meta.setdefault("account_overrides", self._table_account_overrides)
        if self._table_share_raw is not None:
            table_meta.setdefault("share_from_file", self._table_share_raw)
        meta["table"] = table_meta
        meta["inline_symbol_count"] = len(self.inline_symbol_fee_table)
        meta["symbol_fee_table_used"] = len(self.symbol_fee_table)
        meta["maker_bps"] = float(self.cfg.maker_bps)
        meta["taker_bps"] = float(self.cfg.taker_bps)
        meta["maker_discount_mult"] = self._maker_discount_mult
        meta["taker_discount_mult"] = self._taker_discount_mult
        meta["vip_tier"] = int(vip_tier)
        if fee_rounding_step is not None:
            meta["fee_rounding_step"] = float(fee_rounding_step)
        meta["maker_taker_share"] = (
            dict(self.maker_taker_share_raw)
            if isinstance(self.maker_taker_share_raw, Mapping)
            else None
        )
        meta["enabled"] = bool(self.cfg.enabled)
        meta["table_applied"] = bool(self.symbol_fee_table)
        return meta

    def _build_expected_payload(self, *, vip_tier: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "maker_fee_bps": self.base_fee_bps["maker_fee_bps"],
            "taker_fee_bps": self.base_fee_bps["taker_fee_bps"],
            "maker_discount_mult": self._maker_discount_mult,
            "taker_discount_mult": self._taker_discount_mult,
            "use_bnb_discount": self._use_bnb_discount,
            "vip_tier": int(vip_tier),
        }
        if self.maker_taker_share_expected is not None:
            payload.update(self.maker_taker_share_expected)
        else:
            payload.setdefault("expected_fee_bps", payload["taker_fee_bps"])
        return payload

    @staticmethod
    def _parse_fee_table(raw: Mapping[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        table: Dict[str, Any] = {}
        account: Dict[str, Any] = {}
        share: Optional[Dict[str, Any]] = None

        meta_block = raw.get("meta") or raw.get("metadata")
        if isinstance(meta_block, Mapping):
            meta = {k: v for k, v in meta_block.items()}

        account_block = raw.get("account")
        if isinstance(account_block, Mapping):
            for key, value in account_block.items():
                account[key] = value
        account_keys = {
            "maker_bps",
            "taker_bps",
            "use_bnb_discount",
            "maker_discount_mult",
            "taker_discount_mult",
            "vip_tier",
            "fee_rounding_step",
        }
        for key in account_keys:
            if key in raw and raw[key] is not None and key not in account:
                account[key] = raw[key]

        share_block = raw.get("maker_taker_share")
        if share_block is None and isinstance(account_block, Mapping):
            share_block = account_block.get("maker_taker_share")
        if isinstance(share_block, Mapping):
            share = {k: v for k, v in share_block.items()}

        table_block: Any = None
        for key in ("symbol_fee_table", "symbols", "fees_by_symbol", "data"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping):
                table_block = candidate
                break
        if table_block is None:
            candidate_table: Dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    candidate_table[key] = value
            if candidate_table:
                table_block = candidate_table
        if isinstance(table_block, Mapping):
            for symbol, payload in table_block.items():
                if not isinstance(symbol, str) or not isinstance(payload, Mapping):
                    continue
                table[symbol.upper()] = dict(payload)

        return {"table": table, "meta": meta, "account": account, "share": share}

    @classmethod
    def _read_fee_table(
        cls, path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        abspath = os.path.abspath(path)
        try:
            stat = os.stat(abspath)
        except OSError as exc:
            logger.warning("Fees table %s is not accessible: %s", abspath, exc)
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, None
        mtime = stat.st_mtime
        cached = _FEE_TABLE_CACHE.get(abspath)
        if cached and cached[0] == mtime:
            return cached[1], mtime
        try:
            with open(abspath, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load fees table %s: %s", abspath, exc)
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, mtime
        if not isinstance(raw_payload, Mapping):
            logger.warning(
                "Fees table %s has invalid structure (%s); ignoring",
                abspath,
                type(raw_payload).__name__,
            )
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, mtime
        payload = cls._parse_fee_table(raw_payload)
        _FEE_TABLE_CACHE[abspath] = (mtime, payload)
        return payload, mtime

    def _load_symbol_fee_table(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        path_candidate = self.cfg.path
        if path_candidate is None:
            default_path = _DEFAULT_FEE_TABLE_PATH
            if default_path.exists():
                path_candidate = str(default_path)
        if path_candidate is None:
            return payload
        abspath = os.path.abspath(path_candidate)
        self.table_path = abspath
        data, mtime = self._read_fee_table(abspath)
        if data is None:
            if os.path.exists(abspath):
                logger.warning(
                    "Fees table %s is unusable; falling back to global fees", abspath
                )
                self.table_error = "invalid"
            else:
                if self.cfg.path:
                    logger.warning(
                        "Fees table %s not found; falling back to global fees", abspath
                    )
                self.table_error = "missing"
            self.table_metadata = {
                "path": abspath,
                "age_days": None,
                "refresh_days": self.cfg.refresh_days,
                "stale": False,
                "error": self.table_error,
            }
            return payload

        payload = data
        if mtime is not None:
            age_days = max((time.time() - mtime) / 86400.0, 0.0)
            self.table_age_days = age_days
            refresh_days = self.cfg.refresh_days
            if refresh_days is not None and refresh_days >= 0:
                if age_days > float(refresh_days):
                    self.table_stale = True
                    logger.warning(
                        "Fees table %s is stale (age %.1f days > refresh_days=%s); "
                        "using global rates",
                        abspath,
                        age_days,
                        refresh_days,
                    )
        meta = dict(payload.get("meta", {}))
        meta.update(
            {
                "path": abspath,
                "age_days": self.table_age_days,
                "refresh_days": self.cfg.refresh_days,
                "stale": self.table_stale,
                "error": self.table_error,
            }
        )
        self.table_metadata = meta
        return payload

    @property
    def model(self):
        return self._model

    def get_expected_info(self) -> Dict[str, Any]:
        return {
            "expected": dict(self.expected_payload),
            "metadata": dict(self.metadata),
            "symbol_fee_table": {
                "count": len(self.symbol_fee_table),
                "inline_count": len(self.inline_symbol_fee_table),
                "file_count": len(self.symbol_fee_table_raw),
            },
        }

    def attach_to(self, sim) -> None:
        if self._model is not None:
            setattr(sim, "fees", self._model)
        share_payload = None
        if self.maker_taker_share_cfg is not None:
            share_payload = self.maker_taker_share_cfg.to_sim_payload(
                self.base_fee_bps["maker_fee_bps"],
                self.base_fee_bps["taker_fee_bps"],
            )
        elif isinstance(self.maker_taker_share_raw, Mapping):
            share_payload = dict(self.maker_taker_share_raw)
        setattr(sim, "_maker_taker_share_cfg", share_payload)
        try:
            setattr(sim, "fees_config_payload", dict(self.model_payload))
        except Exception:
            logger.debug("Failed to attach fees_config_payload to simulator", exc_info=True)
        try:
            setattr(sim, "fees_metadata", dict(self.metadata))
        except Exception:
            logger.debug("Failed to attach fees_metadata to simulator", exc_info=True)
        try:
            setattr(sim, "fees_expected_payload", dict(self.expected_payload))
        except Exception:
            logger.debug("Failed to attach fees_expected_payload to simulator", exc_info=True)
        setter = getattr(sim, "set_fees_config", None)
        if callable(setter):
            try:
                setter(
                    dict(self.model_payload),
                    share_payload,
                    dict(self.metadata),
                    dict(self.expected_payload),
                )
            except Exception:
                logger.debug("Simulator set_fees_config call failed", exc_info=True)
        try:
            setattr(sim, "_fees_get_expected_info", self.get_expected_info)
        except Exception:
            logger.debug("Failed to attach _fees_get_expected_info", exc_info=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeesImpl":
        use_bnb = bool(d.get("use_bnb_discount", False))
        maker_mult = d.get("maker_discount_mult")
        taker_mult = d.get("taker_discount_mult")

        share_block = d.get("maker_taker_share")
        share_payload: Optional[Dict[str, Any]] = None
        share_cfg = MakerTakerShareSettings.parse(share_block)
        if share_cfg is not None:
            share_payload = share_cfg.as_dict()
        elif isinstance(share_block, Mapping):
            share_payload = dict(share_block)

        share_enabled = d.get("maker_taker_share_enabled")
        share_mode = d.get("maker_taker_share_mode") or d.get("maker_share_mode")
        share_default = d.get("maker_share_default")
        spread_maker = d.get("spread_cost_maker_bps")
        spread_taker = d.get("spread_cost_taker_bps")
        taker_override = d.get("taker_fee_override_bps")

        symbol_table = None
        for key in ("symbol_fee_table", "symbols", "fees_by_symbol"):
            block = d.get(key)
            if isinstance(block, Mapping):
                symbol_table = dict(block)
                break

        metadata = None
        for key in ("metadata", "meta"):
            block = d.get(key)
            if isinstance(block, Mapping):
                metadata = dict(block)
                break

        path = None
        for key in ("path", "fees_path", "symbol_fee_path"):
            candidate = d.get(key)
            if candidate:
                path = candidate
                break

        refresh_days = d.get("refresh_days")

        vip_tier = d.get("vip_tier")
        fee_rounding_step = d.get("fee_rounding_step")

        return FeesImpl(
            FeesConfig(
                enabled=d.get("enabled", True),
                path=path,
                refresh_days=refresh_days,
                maker_bps=d.get("maker_bps", 1.0),
                taker_bps=d.get("taker_bps", 5.0),
                use_bnb_discount=use_bnb,
                maker_discount_mult=maker_mult,
                taker_discount_mult=taker_mult,
                vip_tier=vip_tier,
                fee_rounding_step=fee_rounding_step,
                symbol_fee_table=symbol_table,
                metadata=metadata,
                maker_taker_share=share_payload,
                maker_taker_share_enabled=share_enabled,
                maker_taker_share_mode=share_mode,
                maker_share_default=share_default,
                spread_cost_maker_bps=spread_maker,
                spread_cost_taker_bps=spread_taker,
                taker_fee_override_bps=taker_override,
            )
        )

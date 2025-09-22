# -*- coding: utf-8 -*-
"""
impl_sim_executor.py
Реализация исполнителя торгов (TradeExecutor) поверх ExecutionSimulator из execution_sim.py.

Контракт:
- Принимает core_models.Order.
- Возвращает core_models.ExecReport(ы) через compat_shims.sim_report_dict_to_core_exec_reports().
- cancel() — no-op (симулятор ордеров «в книге» моделирует через new_order_ids, а не реальную книгу заявок).
- get_open_positions() — строит Position из состояния симулятора (position_qty, _avg_entry_price, realized_pnl_cum, fees_cum).

Важно:
- Все файлы лежат в одной папке; импорты — по именам модулей.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
import logging
import math
from decimal import Decimal
from typing import Dict, Optional, Sequence, Mapping, List, Any

from core_models import Order, ExecReport, Position, as_dict
from core_contracts import TradeExecutor
from compat_shims import sim_report_dict_to_core_exec_reports
from execution_sim import ExecutionSimulator, SimStepReport  # type: ignore
from action_proto import ActionProto, ActionType
from core_config import ExecutionProfile, ExecutionParams, ExecutionEntryMode
from config import DataDegradationConfig

# новые компонентные имплементации
from impl_quantizer import QuantizerImpl, QuantizerConfig
from impl_fees import FeesImpl, FeesConfig
from impl_slippage import SlippageImpl, SlippageCfg, load_calibration_artifact
from impl_latency import LatencyImpl, LatencyCfg
from impl_risk_basic import RiskBasicImpl, RiskBasicCfg


logger = logging.getLogger(__name__)


@dataclass
class _SimCtx:
    symbol: str
    # базовая единица позиции для volume_frac. Если 0, будем интерпретировать quantity как долю 1.0
    max_position_abs_base: float = 1.0


class SimExecutor(TradeExecutor):
    """
    Обёртка над ExecutionSimulator с интерфейсом TradeExecutor.
    """

    @staticmethod
    def _latency_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _execution_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _slippage_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, Mapping):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _dynamic_spread_enabled(cfg: Any) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, Mapping):
            dyn_block = cfg.get("dynamic")
            if dyn_block is None:
                dyn_block = cfg.get("dynamic_spread")
        else:
            dyn_block = getattr(cfg, "dynamic", None)
            if dyn_block is None:
                dyn_block = getattr(cfg, "dynamic_spread", None)
        if dyn_block is None:
            return False
        if isinstance(dyn_block, Mapping):
            enabled_value = dyn_block.get("enabled")
        else:
            enabled_value = getattr(dyn_block, "enabled", None)
        try:
            return bool(enabled_value)
        except Exception:
            return False

    @staticmethod
    def _fees_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, Mapping):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _build_fee_config(
        raw_cfg: Any,
        *,
        run_config: Any | None,
        symbol: str,
    ) -> Dict[str, Any]:
        payload = SimExecutor._fees_dict(raw_cfg)
        if symbol and "symbol" not in payload:
            payload["symbol"] = symbol

        def _get_attr(source: Any, key: str) -> Any:
            if source is None:
                return None
            if isinstance(source, Mapping):
                return source.get(key)
            return getattr(source, key, None)

        share_block = payload.get("maker_taker_share")
        if share_block is None and run_config is not None:
            direct_share = _get_attr(run_config, "maker_taker_share")
            if direct_share is not None:
                if hasattr(direct_share, "dict"):
                    try:
                        share_payload = direct_share.dict(exclude_unset=False)  # type: ignore[call-arg]
                    except Exception:
                        share_payload = None
                    else:
                        if isinstance(share_payload, dict):
                            direct_share = share_payload
                payload["maker_taker_share"] = direct_share

        override_keys = {
            "maker_taker_share_enabled": "maker_taker_share_enabled",
            "maker_taker_share_mode": "maker_taker_share_mode",
            "maker_share_default": "maker_share_default",
            "spread_cost_maker_bps": "spread_cost_maker_bps",
            "spread_cost_taker_bps": "spread_cost_taker_bps",
            "taker_fee_override_bps": "taker_fee_override_bps",
        }
        for attr_name, key in override_keys.items():
            if key in payload:
                continue
            override_value = _get_attr(run_config, attr_name)
            if override_value is None:
                continue
            payload[key] = override_value

        return payload

    @staticmethod
    def _resolve_calibration_path(
        path: Any,
        *,
        run_config: Any | None,
    ) -> Optional[str]:
        candidates: list[str] = []
        env_path = os.getenv("SLIPPAGE_CALIBRATION_PATH")
        if env_path:
            candidates.append(env_path)
        if path:
            candidates.append(str(path))
        if run_config is not None:
            art_dir = getattr(run_config, "artifacts_dir", None)
            if art_dir:
                base_dir = str(art_dir)
                if path:
                    candidates.append(os.path.join(base_dir, str(path)))
                candidates.append(os.path.join(base_dir, "slippage_calibration.json"))
                candidates.append(os.path.join(base_dir, "slippage", "calibration.json"))
        candidates.append("data/slippage/live_slippage_calibration.json")
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            resolved = os.path.expanduser(str(candidate))
            if not os.path.isabs(resolved):
                resolved = os.path.abspath(resolved)
            if resolved in seen:
                continue
            seen.add(resolved)
            if os.path.isfile(resolved):
                return resolved
        return None

    @staticmethod
    def _prepare_slippage_payload(
        raw_cfg: Any,
        *,
        run_config: Any | None,
        symbol: str,
    ) -> Dict[str, Any]:
        payload = SimExecutor._slippage_dict(raw_cfg)

        existing_profiles: Optional[Dict[str, Any]] = None
        existing = payload.get("calibrated_profiles")
        if isinstance(existing, Mapping):
            existing_profiles = dict(existing)
        elif existing is not None:
            try:
                existing_profiles = dict(existing)
            except Exception:
                existing_profiles = None
        if existing_profiles is not None:
            payload["calibrated_profiles"] = existing_profiles

        calibration_enabled = (
            bool(getattr(run_config, "slippage_calibration_enabled", False))
            if run_config is not None
            else False
        )

        if not calibration_enabled:
            if existing_profiles is not None:
                existing_profiles["enabled"] = False
            return payload

        calibration_payload: Optional[Dict[str, Any]] = None
        resolved_path: Optional[str] = None
        calibration_path = (
            getattr(run_config, "slippage_calibration_path", None)
            if run_config is not None
            else None
        )
        resolved_path = SimExecutor._resolve_calibration_path(
            calibration_path,
            run_config=run_config,
        )
        if resolved_path is None:
            logger.warning(
                "Slippage calibration enabled but artifact not found (path=%s)",
                calibration_path,
            )
        else:
            default_symbol = (
                getattr(run_config, "slippage_calibration_default_symbol", None)
                if run_config is not None
                else None
            )
            if not default_symbol:
                default_symbol = symbol
            calibration_payload = load_calibration_artifact(
                resolved_path,
                default_symbol=default_symbol,
                symbols=[symbol],
                enabled=calibration_enabled,
            )
            if calibration_payload is None:
                logger.warning(
                    "Failed to load slippage calibration artifact from %s",
                    resolved_path,
                )

        merged_profiles: Dict[str, Any]
        if calibration_payload:
            merged_profiles = dict(existing_profiles or {})
            calib_symbols = calibration_payload.get("symbols")
            if isinstance(calib_symbols, Mapping):
                existing_symbols: Dict[str, Any] = {}
                symbols_block = merged_profiles.get("symbols")
                if isinstance(symbols_block, Mapping):
                    try:
                        existing_symbols = dict(symbols_block)
                    except Exception:
                        existing_symbols = symbols_block  # type: ignore[assignment]
                if existing_symbols:
                    existing_symbols.update(calib_symbols)  # type: ignore[arg-type]
                else:
                    existing_symbols = dict(calib_symbols)
                merged_profiles["symbols"] = existing_symbols
            for key in ("path", "default_symbol", "last_refresh_ts"):
                value = calibration_payload.get(key)
                if value is not None:
                    merged_profiles[key] = value
            metadata = calibration_payload.get("metadata")
            if isinstance(metadata, Mapping):
                existing_meta = merged_profiles.get("metadata")
                if isinstance(existing_meta, Mapping):
                    meta_dict = dict(existing_meta)
                    meta_dict.update(metadata)
                    merged_profiles["metadata"] = meta_dict
                else:
                    merged_profiles["metadata"] = dict(metadata)
            merged_profiles["enabled"] = bool(
                calibration_payload.get("enabled", calibration_enabled)
            )
            payload["calibrated_profiles"] = merged_profiles
            if resolved_path:
                has_symbols = bool(calibration_payload.get("symbols"))
                log_level = logger.info if has_symbols else logger.warning
                log_level(
                    "Loaded slippage calibration artifact %s for symbol %s%s",
                    resolved_path,
                    symbol,
                    " (no matching profiles)" if not has_symbols else "",
                )
        elif existing_profiles is not None:
            existing_profiles["enabled"] = True

        return payload

    @staticmethod
    def _coerce_execution_profile(
        value: Any,
        default: ExecutionProfile,
    ) -> ExecutionProfile:
        if isinstance(value, ExecutionProfile):
            return value
        if isinstance(value, str):
            text = value.strip().upper().replace("-", "_").replace(" ", "_")
            if text in ExecutionProfile.__members__:
                return ExecutionProfile[text]
        return default

    @staticmethod
    def _bool_or_none(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return bool(int(num))
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            return None
        try:
            return bool(value)
        except Exception:
            return None

    @staticmethod
    def _coerce_execution_entry_mode(
        value: Any,
    ) -> tuple[ExecutionEntryMode | None, ExecutionProfile | None]:
        if value is None:
            return None, None
        if isinstance(value, ExecutionEntryMode):
            return value, None
        if isinstance(value, ExecutionProfile):
            return None, value
        text: Optional[str]
        try:
            text = str(value)
        except Exception:
            text = None
        if not text:
            return None, None
        normalized_text = text.strip()
        if not normalized_text:
            return None, None
        lowered = normalized_text.lower()
        try:
            enum_val = ExecutionEntryMode(lowered)
        except ValueError:
            enum_val = None
        if enum_val is not None:
            return enum_val, None

        alias_map: Dict[str, tuple[ExecutionEntryMode | None, ExecutionProfile | None]] = {
            "market": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "mkt": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "mkt_open_next_h1": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "open": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "open_next": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "next_open": (ExecutionEntryMode.DEFAULT, ExecutionProfile.MKT_OPEN_NEXT_H1),
            "vwap": (ExecutionEntryMode.DEFAULT, ExecutionProfile.VWAP_CURRENT_H1),
            "vwap_current_h1": (ExecutionEntryMode.DEFAULT, ExecutionProfile.VWAP_CURRENT_H1),
            "current_vwap": (ExecutionEntryMode.DEFAULT, ExecutionProfile.VWAP_CURRENT_H1),
            "limit": (ExecutionEntryMode.STRICT, ExecutionProfile.LIMIT_MID_BPS),
            "limit_mid": (ExecutionEntryMode.STRICT, ExecutionProfile.LIMIT_MID_BPS),
            "limit_mid_bps": (ExecutionEntryMode.STRICT, ExecutionProfile.LIMIT_MID_BPS),
            "mid_limit": (ExecutionEntryMode.STRICT, ExecutionProfile.LIMIT_MID_BPS),
            "strict": (ExecutionEntryMode.STRICT, ExecutionProfile.LIMIT_MID_BPS),
            "legacy": (ExecutionEntryMode.DEFAULT, None),
        }
        if lowered in alias_map:
            return alias_map[lowered]

        profile_key = normalized_text.replace("-", "_").replace(" ", "_").upper()
        if profile_key in ExecutionProfile.__members__:
            return None, ExecutionProfile[profile_key]

        return None, None

    @staticmethod
    def resolve_execution_runtime_settings(
        execution_cfg: Any,
        *,
        default_profile: ExecutionProfile | None = None,
    ) -> tuple[ExecutionEntryMode, ExecutionProfile, bool, bool]:
        profile_default = SimExecutor._coerce_execution_profile(
            default_profile or ExecutionProfile.MKT_OPEN_NEXT_H1,
            ExecutionProfile.MKT_OPEN_NEXT_H1,
        )
        entry_mode = ExecutionEntryMode.DEFAULT
        resolved_profile = profile_default
        clip_enabled = True
        strict_open_fill = False

        cfg_obj = execution_cfg
        entry_mode_raw: Any = None
        clip_cfg: Any = None
        if isinstance(cfg_obj, Mapping):
            entry_mode_raw = cfg_obj.get("entry_mode")
            clip_cfg = cfg_obj.get("clip_to_bar")
        else:
            entry_mode_raw = getattr(cfg_obj, "entry_mode", None)
            clip_cfg = getattr(cfg_obj, "clip_to_bar", None)

        mode_candidate, profile_override = SimExecutor._coerce_execution_entry_mode(entry_mode_raw)
        if mode_candidate is not None:
            entry_mode = mode_candidate
        if profile_override is not None:
            resolved_profile = profile_override
        else:
            if isinstance(entry_mode_raw, ExecutionProfile):
                resolved_profile = entry_mode_raw
            elif isinstance(entry_mode_raw, str):
                profile_key = (
                    entry_mode_raw.strip().replace("-", "_").replace(" ", "_").upper()
                )
                if profile_key in ExecutionProfile.__members__:
                    resolved_profile = ExecutionProfile[profile_key]

        clip_enabled_val: bool | None = None
        strict_val: bool | None = None
        if clip_cfg is not None:
            if isinstance(clip_cfg, Mapping):
                clip_enabled_val = SimExecutor._bool_or_none(clip_cfg.get("enabled"))
                strict_val = SimExecutor._bool_or_none(clip_cfg.get("strict_open_fill"))
            else:
                clip_enabled_val = SimExecutor._bool_or_none(
                    getattr(clip_cfg, "enabled", None)
                )
                strict_val = SimExecutor._bool_or_none(
                    getattr(clip_cfg, "strict_open_fill", None)
                )
        if clip_enabled_val is not None:
            clip_enabled = clip_enabled_val
        if strict_val is not None:
            strict_open_fill = strict_val

        return entry_mode, resolved_profile, clip_enabled, strict_open_fill

    @staticmethod
    def _apply_entry_mode_to_sim(
        sim: ExecutionSimulator,
        entry_mode: ExecutionEntryMode | None,
    ) -> None:
        if entry_mode is None:
            return
        value = entry_mode.value if hasattr(entry_mode, "value") else str(entry_mode)
        for attr in ("execution_entry_mode", "_execution_entry_mode"):
            try:
                setattr(sim, attr, value)
            except Exception:
                continue

    @staticmethod
    def _apply_clip_settings_to_sim(
        sim: ExecutionSimulator,
        enabled: bool,
        strict_open_fill: bool,
    ) -> None:
        for attr in ("clip_to_bar_enabled", "_clip_to_bar_enabled"):
            try:
                setattr(sim, attr, bool(enabled))
            except Exception:
                continue
        for attr in (
            "clip_to_bar_strict_open_fill",
            "_clip_to_bar_strict_open_fill",
        ):
            try:
                setattr(sim, attr, bool(strict_open_fill))
            except Exception:
                continue

    @staticmethod
    def configure_simulator_execution(
        sim: ExecutionSimulator,
        execution_cfg: Any,
        *,
        default_profile: ExecutionProfile | None = None,
    ) -> tuple[ExecutionEntryMode, ExecutionProfile, bool, bool]:
        entry_mode, profile, clip_enabled, strict_open_fill = (
            SimExecutor.resolve_execution_runtime_settings(
                execution_cfg, default_profile=default_profile
            )
        )
        SimExecutor._apply_entry_mode_to_sim(sim, entry_mode)
        SimExecutor._apply_clip_settings_to_sim(sim, clip_enabled, strict_open_fill)
        return entry_mode, profile, clip_enabled, strict_open_fill

    @staticmethod
    def _execution_params_dict(params: Any) -> Dict[str, Any]:
        if params is None:
            return {}
        if isinstance(params, Mapping):
            try:
                return dict(params)
            except Exception:
                return {}
        for attr in ("model_dump", "dict"):
            if hasattr(params, attr):
                try:
                    method = getattr(params, attr)
                    payload = method(exclude_unset=False)  # type: ignore[call-arg]
                except TypeError:
                    try:
                        payload = method()  # type: ignore[misc]
                    except Exception:
                        payload = None
                except Exception:
                    payload = None
                if isinstance(payload, Mapping):
                    try:
                        return dict(payload)
                    except Exception:
                        return {}
        try:
            return dict(params)
        except Exception:
            return {}

    @staticmethod
    def _set_execution_profile_on_sim(
        sim: ExecutionSimulator,
        profile: ExecutionProfile,
        params: Any,
    ) -> None:
        payload = SimExecutor._execution_params_dict(params)
        profile_text = str(profile)
        setter = getattr(sim, "set_execution_profile", None)
        if callable(setter):
            try:
                setter(profile_text, payload)
            except Exception:
                pass
        try:
            setattr(sim, "execution_profile", profile_text)
        except Exception:
            pass
        try:
            setattr(sim, "execution_params", payload)
        except Exception:
            pass

    @staticmethod
    def apply_execution_profile(
        sim: ExecutionSimulator,
        profile: ExecutionProfile | str | None,
        params: Any,
    ) -> None:
        coerced = SimExecutor._coerce_execution_profile(
            profile if profile is not None else ExecutionProfile.MKT_OPEN_NEXT_H1,
            ExecutionProfile.MKT_OPEN_NEXT_H1,
        )
        SimExecutor._set_execution_profile_on_sim(sim, coerced, params)

    @staticmethod
    def _attach_quantizer_via_api(
        sim: ExecutionSimulator,
        quantizer: QuantizerImpl,
    ) -> bool:
        attach_api = getattr(sim, "attach_quantizer", None)
        if not callable(attach_api):
            return False

        metadata_view = getattr(quantizer, "filters_metadata", None)
        metadata_payload: Dict[str, Any] = {}
        if isinstance(metadata_view, Mapping):
            try:
                metadata_payload = dict(metadata_view)
            except Exception:
                metadata_payload = {}

        try:
            attach_api(
                impl=quantizer,
                metadata=metadata_payload or None,
            )
        except TypeError:
            logger.debug(
                "Simulator %s.attach_quantizer signature mismatch; falling back to legacy attachment",
                type(sim).__name__,
            )
            return False
        except Exception as exc:
            logger.warning(
                "Simulator %s.attach_quantizer failed: %s; falling back to legacy attachment",
                type(sim).__name__,
                exc,
            )
            return False

        return True

    def _maybe_register_slippage_regime_listener(self) -> None:
        if not getattr(self, "_slippage_regime_updates_enabled", True):
            return
        slippage = getattr(self, "_slippage_impl", None)
        if slippage is None:
            return
        register_fn = getattr(self._sim, "register_market_regime_listener", None)
        if callable(register_fn):
            try:
                register_fn(slippage.set_market_regime)
                self._slippage_regime_listener_registered = True
            except Exception:
                logger.debug(
                    "Failed to register market regime listener", exc_info=True
                )
        if not getattr(self, "_slippage_regime_listener_registered", False):
            regime = getattr(self._sim, "_last_market_regime", None)
            if regime is not None:
                try:
                    slippage.set_market_regime(regime)
                except Exception:
                    logger.debug(
                        "Failed to seed slippage regime state", exc_info=True
                    )

    def __init__(
        self,
        sim: ExecutionSimulator,
        *,
        symbol: str,
        max_position_abs_base: float = 1.0,
        quantizer: QuantizerImpl | None = None,
        risk: RiskBasicImpl | None = None,
        latency: LatencyImpl | None = None,
        slippage: SlippageImpl | None = None,
        fees: FeesImpl | None = None,
        data_degradation: DataDegradationConfig | None = None,
        run_config: Any | None = None,
    ) -> None:
        """Создать исполнителя поверх :class:`ExecutionSimulator`.

        Параметры ``quantizer``, ``risk``, ``latency``, ``slippage`` и ``fees`` могут
        быть переданы явно как соответствующие реализации. Если они отсутствуют,
        попытка построения происходит из блоков ``run_config``: ``quantizer``,
        ``fees``, ``slippage``, ``latency``, ``risk`` и ``no_trade``. При отсутствии
        этих блоков используются значения по умолчанию.
        """

        self._sim = sim
        self._run_id = str(getattr(run_config, "run_id", "sim") or "sim")
        self._ctx = _SimCtx(symbol=str(symbol), max_position_abs_base=float(max_position_abs_base))
        self._slippage_impl: SlippageImpl | None = None
        self._slippage_regime_updates_enabled = (
            bool(getattr(run_config, "slippage_regime_updates", True))
            if run_config is not None
            else True
        )
        self._slippage_regime_listener_registered = False

        close_lag_value: int | None = None
        if run_config is not None:
            timing_cfg = getattr(run_config, "timing", None)
            if timing_cfg is not None:
                candidate = getattr(timing_cfg, "close_lag_ms", None)
                if candidate is None and isinstance(timing_cfg, Mapping):
                    candidate = timing_cfg.get("close_lag_ms")
                try:
                    if candidate is not None:
                        close_lag_value = int(candidate)
                except (TypeError, ValueError):
                    close_lag_value = None
        if close_lag_value is not None and close_lag_value < 0:
            close_lag_value = 0
        if close_lag_value is not None:
            self._close_lag_ms = close_lag_value
            for attr in ("close_lag_ms", "_timing_close_lag_ms"):
                try:
                    setattr(self._sim, attr, int(self._close_lag_ms))
                except Exception:
                    continue
        else:
            existing_close_lag = getattr(self._sim, "close_lag_ms", None)
            try:
                self._close_lag_ms = int(existing_close_lag)
            except (TypeError, ValueError):
                self._close_lag_ms = 0
            if self._close_lag_ms < 0:
                self._close_lag_ms = 0

        rc_quantizer = getattr(run_config, "quantizer", {}) if run_config else {}
        rc_risk = getattr(run_config, "risk", None) if run_config else None
        rc_latency = getattr(run_config, "latency", None) if run_config else None
        rc_slippage = SimExecutor._prepare_slippage_payload(
            getattr(run_config, "slippage", {}) if run_config else {},
            run_config=run_config,
            symbol=str(symbol),
        )
        rc_fees = getattr(run_config, "fees", {}) if run_config else {}
        rc_degradation = getattr(run_config, "data_degradation", {}) if run_config else {}
        self._no_trade_cfg = getattr(run_config, "no_trade", {}) if run_config else {}
        raw_exec_profile = (
            getattr(run_config, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)
            if run_config is not None
            else ExecutionProfile.MKT_OPEN_NEXT_H1
        )
        self._exec_profile: ExecutionProfile = SimExecutor._coerce_execution_profile(
            raw_exec_profile,
            ExecutionProfile.MKT_OPEN_NEXT_H1,
        )
        self._exec_params: ExecutionParams = (
            getattr(run_config, "execution_params", ExecutionParams())
            if run_config is not None
            else ExecutionParams()
        )
        self._execution_cfg = getattr(run_config, "execution", None) if run_config else None
        exec_cfg_payload: Dict[str, Any] = {}
        if run_config is not None:
            exec_cfg_payload = self._execution_dict(self._execution_cfg)
        if exec_cfg_payload:
            try:
                setattr(self._sim, "_execution_intrabar_cfg", dict(exec_cfg_payload))
            except Exception:
                pass
            bridge_payload = exec_cfg_payload.get("bridge")
            if isinstance(bridge_payload, dict):
                try:
                    setattr(self._sim, "_execution_bridge_cfg", dict(bridge_payload))
                except Exception:
                    pass
        if run_config is not None:
            try:
                setattr(self._sim, "_execution_runtime_cfg", self._execution_cfg)
            except Exception:
                pass

        (
            self._entry_mode,
            self._exec_profile,
            self._clip_to_bar_enabled,
            self._strict_open_fill,
        ) = SimExecutor.configure_simulator_execution(
            self._sim,
            self._execution_cfg,
            default_profile=self._exec_profile,
        )
        if data_degradation is None:
            data_degradation = (
                DataDegradationConfig.from_dict(rc_degradation)
                if rc_degradation
                else DataDegradationConfig.default()
            )
        self._data_degradation = data_degradation

        if quantizer is None:
            quantizer = QuantizerImpl.from_dict(rc_quantizer)
        self._quantizer_impl: QuantizerImpl | None = quantizer
        if risk is None:
            risk = RiskBasicImpl.from_dict(rc_risk)
        if latency is None:
            cfg_lat = self._latency_dict(rc_latency)
            if run_config is not None:
                lat_path = getattr(run_config, "latency_seasonality_path", None)
                if lat_path and not cfg_lat.get("latency_seasonality_path"):
                    cfg_lat.setdefault("latency_seasonality_path", lat_path)
            sim_lat_cfg = getattr(sim, "latency_config_payload", None)
            if sim_lat_cfg:
                sim_lat_dict = self._latency_dict(sim_lat_cfg)
                for key, value in sim_lat_dict.items():
                    cfg_lat.setdefault(key, value)
            cfg_lat.setdefault("symbol", symbol)
            latency = LatencyImpl.from_dict(cfg_lat)
        if slippage is None:
            slippage = SlippageImpl.from_dict(rc_slippage, run_config=run_config)
        self._slippage_impl = slippage
        if fees is None:
            fee_cfg_payload = self._build_fee_config(
                rc_fees,
                run_config=run_config,
                symbol=str(symbol),
            )
            fees = FeesImpl.from_dict(fee_cfg_payload)

        dyn_cfg_source: Any = None
        if run_config is not None:
            dyn_cfg_source = getattr(run_config, "slippage", None)
        if dyn_cfg_source is None:
            dyn_cfg_source = rc_slippage
        dyn_spread_enabled = self._dynamic_spread_enabled(dyn_cfg_source)

        # последовательное подключение компонентов к симулятору
        if quantizer is not None:
            attached = self._attach_quantizer_via_api(self._sim, quantizer)
            if not attached:
                quantizer.attach_to(
                    self._sim,
                    strict=quantizer.cfg.strict,
                    enforce_percent_price_by_side=quantizer.cfg.enforce_percent_price_by_side,
                )
        if risk is not None:
            risk.attach_to(self._sim)
        if latency is not None:
            latency.attach_to(self._sim)
        if slippage is not None:
            slippage.attach_to(self._sim)
            if dyn_spread_enabled:
                profile = getattr(slippage, "dynamic_profile", None)
                if profile is not None:
                    try:
                        setattr(self._sim, "slippage_dynamic_profile", profile)
                    except Exception:
                        pass
        if fees is not None:
            fees.attach_to(self._sim)

        self._maybe_register_slippage_regime_listener()

        SimExecutor.apply_execution_profile(
            self._sim,
            self._exec_profile,
            self._exec_params,
        )

    def update_market_regime(self, regime: Any) -> None:
        if not getattr(self, "_slippage_regime_updates_enabled", True):
            return
        slippage = getattr(self, "_slippage_impl", None)
        if slippage is not None:
            setter = getattr(slippage, "set_market_regime", None)
            if callable(setter):
                try:
                    setter(regime)
                except Exception:
                    logger.debug(
                        "Failed to forward market regime to slippage", exc_info=True
                    )
        hint_fn = getattr(self._sim, "set_market_regime_hint", None)
        if callable(hint_fn):
            try:
                hint_fn(regime)
            except Exception:
                logger.debug(
                    "Failed to update simulator regime hint", exc_info=True
                )
        else:
            try:
                setattr(self._sim, "_last_market_regime", regime)
            except Exception:
                pass

    @staticmethod
    def from_config(
        *,
        symbol: str,
        max_position_abs_base: float = 1.0,
        sim: ExecutionSimulator,
        run_config: Any | None = None,
    ) -> "SimExecutor":
        """Сконструировать :class:`SimExecutor` из ``run_config``.

        Извлекает блоки ``quantizer``, ``fees``, ``slippage``, ``latency``, ``risk`` и
        ``no_trade`` из ``run_config`` и создаёт соответствующие реализации.
        Значения по умолчанию используются, если блок отсутствует.
        """

        q_impl = QuantizerImpl.from_dict(getattr(run_config, "quantizer", {}) or {})
        fee_cfg_payload = SimExecutor._build_fee_config(
            getattr(run_config, "fees", {}) or {},
            run_config=run_config,
            symbol=str(symbol),
        )
        f_impl = FeesImpl.from_dict(fee_cfg_payload)
        s_impl = SlippageImpl.from_dict(
            getattr(run_config, "slippage", {}) or {}, run_config=run_config
        )
        l_cfg = SimExecutor._latency_dict(getattr(run_config, "latency", None))
        lat_path = getattr(run_config, "latency_seasonality_path", None)
        if lat_path and not l_cfg.get("latency_seasonality_path"):
            l_cfg.setdefault("latency_seasonality_path", lat_path)
        sim_lat_cfg = getattr(sim, "latency_config_payload", None)
        if sim_lat_cfg:
            sim_lat_dict = SimExecutor._latency_dict(sim_lat_cfg)
            for key, value in sim_lat_dict.items():
                l_cfg.setdefault(key, value)
        l_cfg.setdefault("symbol", symbol)
        l_impl = LatencyImpl.from_dict(l_cfg)
        r_impl = RiskBasicImpl.from_dict(getattr(run_config, "risk", None))
        d_impl = DataDegradationConfig.from_dict(
            getattr(run_config, "data_degradation", {}) or {}
        )

        execution_cfg = getattr(run_config, "execution", None)
        exec_cfg_payload = SimExecutor._execution_dict(execution_cfg)
        if exec_cfg_payload:
            try:
                setattr(sim, "_execution_intrabar_cfg", dict(exec_cfg_payload))
            except Exception:
                pass
            bridge_payload = exec_cfg_payload.get("bridge")
            if isinstance(bridge_payload, dict):
                try:
                    setattr(sim, "_execution_bridge_cfg", dict(bridge_payload))
                except Exception:
                    pass
        if run_config is not None:
            try:
                setattr(sim, "_execution_runtime_cfg", execution_cfg)
            except Exception:
                pass

        default_profile = SimExecutor._coerce_execution_profile(
            getattr(run_config, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)
            if run_config is not None
            else ExecutionProfile.MKT_OPEN_NEXT_H1,
            ExecutionProfile.MKT_OPEN_NEXT_H1,
        )
        _, resolved_profile, _, _ = SimExecutor.configure_simulator_execution(
            sim,
            execution_cfg,
            default_profile=default_profile,
        )
        SimExecutor.apply_execution_profile(
            sim,
            resolved_profile,
            getattr(run_config, "execution_params", None) if run_config is not None else None,
        )

        if q_impl is not None:
            attached = SimExecutor._attach_quantizer_via_api(sim, q_impl)
            if not attached:
                q_impl.attach_to(
                    sim,
                    strict=q_impl.cfg.strict,
                    enforce_percent_price_by_side=q_impl.cfg.enforce_percent_price_by_side,
                )
        if r_impl is not None:
            r_impl.attach_to(sim)
        if l_impl is not None:
            l_impl.attach_to(sim)
        if s_impl is not None:
            s_impl.attach_to(sim)
        if f_impl is not None:
            f_impl.attach_to(sim)

        return SimExecutor(
            sim,
            symbol=symbol,
            max_position_abs_base=max_position_abs_base,
            quantizer=q_impl,
            risk=r_impl,
            latency=l_impl,
            slippage=s_impl,
            fees=f_impl,
            data_degradation=d_impl,
            run_config=run_config,
        )

    # ---- вспомогательное: Order -> (ActionType, ActionProto) ----
    def _order_to_action(self, order: Order) -> tuple[int, object]:
        """
        Преобразует core_models.Order к (ActionType, ActionProto) симулятора.
        Интерпретация:
        - MARKET: volume_frac = sign(quantity) * min(1.0, abs(quantity) / max_position_abs_base)
        - LIMIT:  volume_frac аналогично; если price задан — кладём в proto.abs_price
        """
        qty = float(order.quantity)
        base = float(self._ctx.max_position_abs_base) if self._ctx.max_position_abs_base > 0 else 1.0
        if base <= 0:
            base = 1.0
        vol_frac = max(0.0, abs(qty) / base)
        if str(order.side).upper().endswith("SELL"):
            vol_frac = -vol_frac

        tif = str(self._exec_params.tif)
        ttl_steps = int(self._exec_params.ttl_steps)

        profile = getattr(self, "_exec_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)

        client_tag = getattr(order, "client_order_id", "") or ""
        if profile == ExecutionProfile.LIMIT_MID_BPS:
            proto_kwargs = dict(
                action_type=ActionType.LIMIT,
                volume_frac=float(vol_frac),
                ttl_steps=ttl_steps,
                tif=tif,
                client_tag=client_tag,
            )
            price = getattr(order, "price", None)
            if price is None:
                mid = getattr(self._sim, "_last_ref_price", None)
                if mid is None:
                    bid = getattr(self._sim, "_last_bid", None)
                    ask = getattr(self._sim, "_last_ask", None)
                    if bid is not None and ask is not None:
                        mid = (float(bid) + float(ask)) / 2.0
                if mid is not None:
                    off = float(self._exec_params.limit_offset_bps) / 1e4
                    if vol_frac > 0:
                        price = mid * (1 - off)
                    else:
                        price = mid * (1 + off)
            if price is not None:
                proto_kwargs["abs_price"] = float(price)
            proto = ActionProto(**proto_kwargs)
            return ActionType.LIMIT, proto

        proto = ActionProto(
            action_type=ActionType.MARKET,
            volume_frac=float(vol_frac),
            ttl_steps=ttl_steps,
            tif=tif,
            client_tag=client_tag,
        )
        return ActionType.MARKET, proto

    def _quantizer_precheck_enabled(self) -> bool:
        quantizer = self._quantizer_impl
        if quantizer is None:
            return False
        cfg = getattr(quantizer, "cfg", None)
        if cfg is None:
            return False
        strict = bool(getattr(cfg, "strict", False))
        enforce_ppbs = bool(getattr(cfg, "enforce_percent_price_by_side", False))
        return strict or enforce_ppbs

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

    def _build_precheck_rejection(
        self,
        order: Order,
        *,
        reason_code: Optional[str],
        reason_message: Optional[str],
        details: Optional[Mapping[str, Any]],
        quantized_price: float,
        quantized_qty: float,
        price: float | None,
        ref_price: float | None,
        signed_qty: float,
    ) -> ExecReport:
        report = SimStepReport()
        report.status = "REJECTED_BY_FILTER"
        report.exec_status = "REJECTED"
        report.execution_profile = str(self._exec_profile)
        price_hint = price if price is not None else ref_price

        def _safe_float(value: Any) -> float:
            try:
                if value is None:
                    return 0.0
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        report.position_qty = _safe_float(getattr(self._sim, "position_qty", 0.0))
        report.realized_pnl = _safe_float(getattr(self._sim, "realized_pnl_cum", 0.0))
        report.unrealized_pnl = _safe_float(getattr(self._sim, "unrealized_pnl", 0.0))
        report.equity = _safe_float(getattr(self._sim, "equity", 0.0))
        report.mark_price = _safe_float(price_hint)
        report.mtm_price = report.mark_price
        report.bid = _safe_float(getattr(self._sim, "_last_bid", 0.0))
        report.ask = _safe_float(getattr(self._sim, "_last_ask", 0.0))
        report.latency_p50_ms = _safe_float(getattr(self._sim, "latency_p50_ms", 0.0))
        report.latency_p95_ms = _safe_float(getattr(self._sim, "latency_p95_ms", 0.0))
        report.latency_timeout_ratio = _safe_float(
            getattr(self._sim, "latency_timeout_ratio", 0.0)
        )
        report.vol_factor = getattr(self._sim, "_last_vol_factor", None)
        report.liquidity = getattr(self._sim, "_last_liquidity", None)

        detail_payload: Dict[str, Any] = {
            "code": str(reason_code or "FILTER"),
            "price": quantized_price,
            "qty": quantized_qty,
            "original_price": price_hint,
            "original_qty": abs(float(signed_qty)),
            "side": str(order.side),
        }
        if reason_message:
            detail_payload["message"] = str(reason_message)
        if details:
            try:
                detail_payload["constraint"] = dict(details)
            except Exception:
                detail_payload["constraint"] = details
        if ref_price is not None:
            detail_payload["ref_price"] = ref_price
        entry_extra: Dict[str, Any] = {
            "order_type": str(order.order_type),
            "source": "quantizer_precheck",
        }
        client_id = getattr(order, "client_order_id", None)
        if client_id:
            entry_extra["client_order_id"] = str(client_id)
        rejection_entry = ExecutionSimulator._build_reason_payload(
            str(reason_code or "FILTER"),
            details=detail_payload,
            extra=entry_extra,
        )
        entries = [rejection_entry]
        counts = ExecutionSimulator._summarize_rejection_counts(entries)
        extra_payload = {"counts": counts} if counts else None
        report.reason = ExecutionSimulator._build_reason_payload(
            "FILTER_REJECTION",
            details={"rejections": entries},
            extra=extra_payload,
        )

        payload = report.to_dict()
        core_reports = sim_report_dict_to_core_exec_reports(
            payload,
            symbol=self._ctx.symbol,
            run_id=self._run_id,
            client_order_id=str(client_id or ""),
        )
        if core_reports:
            return core_reports[0]

        return ExecReport.from_dict(
            {
                "ts": int(order.ts),
                "symbol": self._ctx.symbol,
                "side": "BUY"
                if str(order.side).upper().endswith("BUY")
                else "SELL",
                "order_type": str(order.order_type),
                "price": float(price_hint or 0.0),
                "quantity": 0.0,
                "fee": 0.0,
                "fee_asset": None,
                "pnl": 0.0,
                "exec_status": "REJECTED",
                "liquidity": "UNKNOWN",
                "client_order_id": str(client_id or ""),
                "order_id": None,
                "meta": {
                    "filter_rejection": report.reason,
                    "execution_profile": str(self._exec_profile),
                },
                "execution_profile": str(self._exec_profile),
                "run_id": self._run_id,
            }
        )

    # ---- интерфейс TradeExecutor ----
    def execute(self, order: Order) -> ExecReport:
        """
        Синхронно исполняет ордер через ExecutionSimulator и возвращает первый ExecReport.
        Если сделок не было — возвращает ExecReport с нулевым qty и статусом 'NONE'
        (совместимый fallback). Отказы фильтров отражаются как ExecStatus.REJECTED.
        """
        symbol = self._ctx.symbol
        side_str = str(order.side)
        order_type_str = str(order.order_type)
        qty_val = float(order.quantity)
        signed_qty = abs(qty_val)
        if side_str.upper().endswith("SELL"):
            signed_qty = -signed_qty

        explicit_price = self._float_or_none(getattr(order, "price", None))
        last_ref_raw = getattr(self._sim, "_last_ref_price", None)
        last_ref_price = self._float_or_none(last_ref_raw)
        price_for_check = explicit_price if explicit_price is not None else last_ref_price
        ref_for_check = last_ref_price if last_ref_price is not None else price_for_check

        validation_result: Any | None = None
        quantizer = self._quantizer_impl
        if (
            quantizer is not None
            and hasattr(quantizer, "validate_order")
            and self._quantizer_precheck_enabled()
            and price_for_check is not None
            and ref_for_check is not None
        ):
            cfg = getattr(quantizer, "cfg", None)
            enforce_ppbs = bool(getattr(cfg, "enforce_percent_price_by_side", False)) if cfg is not None else False
            try:
                validation_result = quantizer.validate_order(
                    symbol,
                    side_str,
                    float(price_for_check),
                    float(signed_qty),
                    ref_price=float(ref_for_check),
                    enforce_ppbs=enforce_ppbs,
                )
            except Exception as exc:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Quantizer pre-check failed for order %s (%s): %s",
                        getattr(order, "client_order_id", ""),
                        symbol,
                        exc,
                        exc_info=True,
                    )

        if validation_result is not None:
            accepted = bool(getattr(validation_result, "accepted", True))
            quantized_price = self._float_or_none(getattr(validation_result, "price", price_for_check))
            if quantized_price is None:
                quantized_price = float(price_for_check or 0.0)
            quantized_qty = self._float_or_none(getattr(validation_result, "qty", abs(signed_qty)))
            if quantized_qty is None:
                quantized_qty = abs(float(signed_qty))
            if not accepted:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Quantizer pre-check rejected order %s (%s %s): code=%s reason=%s",
                        getattr(order, "client_order_id", ""),
                        side_str,
                        order_type_str,
                        getattr(validation_result, "reason_code", None),
                        getattr(validation_result, "reason", None),
                    )
                return self._build_precheck_rejection(
                    order,
                    reason_code=getattr(validation_result, "reason_code", None),
                    reason_message=getattr(validation_result, "reason", None),
                    details=getattr(validation_result, "details", None),
                    quantized_price=float(quantized_price),
                    quantized_qty=float(quantized_qty),
                    price=price_for_check,
                    ref_price=ref_for_check,
                    signed_qty=float(signed_qty),
                )

            price_changed = (
                price_for_check is not None
                and quantized_price is not None
                and not math.isclose(
                    float(quantized_price),
                    float(price_for_check),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            qty_changed = not math.isclose(
                float(quantized_qty),
                float(abs(signed_qty)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            if (price_changed or qty_changed) and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Quantizer pre-check adjusted order %s: price %s -> %s, qty %s -> %s",
                    getattr(order, "client_order_id", ""),
                    price_for_check,
                    quantized_price,
                    abs(signed_qty),
                    quantized_qty,
                )

        atype, proto = self._order_to_action(order)

        # обновим символ и реф. цену в симуляторе, если нужно
        try:
            self._sim.set_symbol(self._ctx.symbol)
        except Exception:
            pass

        if getattr(order, "price", None) is not None:
            try:
                ref_price_arg = float(order.price)
            except (TypeError, ValueError):
                ref_price_arg = explicit_price
        else:
            ref_price_arg = last_ref_raw

        # прогон шага симуляции с одним действием
        rep: SimStepReport = self._sim.run_step(
            ts=int(order.ts),
            ref_price=ref_price_arg,
            bid=None,
            ask=None,
            vol_factor=None,
            liquidity=None,
            actions=[(atype, proto)],
        )  # type: ignore

        d = rep.to_dict()
        core_reports: List[ExecReport] = sim_report_dict_to_core_exec_reports(
            d,
            symbol=self._ctx.symbol,
            run_id=self._run_id,
            client_order_id=str(getattr(order, "client_order_id", "") or ""),
        )
        if core_reports:
            return core_reports[0]

        status_val = str(getattr(rep, "status", "") or d.get("status") or "").upper()
        if status_val == "REJECTED_BY_FILTER":
            # compat-шлюз строит REJECTED-заглушку, если сделки отсутствуют
            reject_reports = sim_report_dict_to_core_exec_reports(
                d,
                symbol=self._ctx.symbol,
                run_id=self._run_id,
                client_order_id=str(getattr(order, "client_order_id", "") or ""),
            )
            if reject_reports:
                return reject_reports[0]

        # Возвращаем первый отчёт; при необходимости вызывающая сторона может получить остальные из d.
        # Для остальных случаев сохраняем прежний NONE-fallback.
        return ExecReport.from_dict({
            "ts": int(order.ts),
            "run_id": self._run_id,
            "symbol": self._ctx.symbol,
            "side": "BUY" if side_str.upper().endswith("BUY") else "SELL",
            "order_type": "MARKET" if order_type_str.upper().endswith("MARKET") else "LIMIT",
            "price": float(order.price) if getattr(order, "price", None) is not None else 0.0,
            "quantity": 0.0,
            "fee": 0.0,
            "fee_asset": None,
            "pnl": 0.0,
            "exec_status": "NEW",
            "liquidity": "UNKNOWN",
            "client_order_id": str(getattr(order, "client_order_id", "") or ""),
            "order_id": None,
            "meta": {},
            "execution_profile": str(self._exec_profile),
        })

    def cancel(self, client_order_id: str) -> None:
        """
        В простом симуляторе отмена моделируется через new_order_ids/new_order_pos или LOB-заглушку.
        Здесь — no-op.
        """
        return None

    def get_open_positions(self, symbols: Optional[Sequence[str]] = None) -> Mapping[str, Position]:
        """
        Строит Position из внутренних полей симулятора.
        """
        sym = self._ctx.symbol
        if symbols is not None and len(symbols) > 0 and sym not in set(map(str, symbols)):
            return {}

        qty = float(getattr(self._sim, "position_qty", 0.0))
        avg = float(getattr(self._sim, "_avg_entry_price", 0.0) or 0.0)
        realized_pnl = float(getattr(self._sim, "realized_pnl_cum", 0.0) or 0.0)
        fee_paid = float(getattr(self._sim, "fees_cum", 0.0) or 0.0)

        pos = Position(
            symbol=str(sym),
            qty=Decimal(str(qty)),
            avg_entry_price=Decimal(str(avg if qty != 0 else 0.0)),
            realized_pnl=Decimal(str(realized_pnl)),
            fee_paid=Decimal(str(fee_paid)),
            ts=None,
            meta={},
        )
        return {sym: pos}

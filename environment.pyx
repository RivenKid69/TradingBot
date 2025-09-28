# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport log, tanh

import numpy as np

from api.config import EnvConfig
from coreworkspace cimport SimulationWorkspace
from fast_lob cimport CythonLOB
from lob_state_cython cimport EnvState, CyMicrostructureGenerator


cdef class TradingEnv:
    """Light-weight trading environment with optional full LOB simulation."""

    cdef object config
    cdef EnvState state
    cdef SimulationWorkspace workspace
    cdef CythonLOB lob
    cdef CyMicrostructureGenerator micro_gen
    cdef bint use_full_lob
    cdef double prev_net_worth
    cdef double prev_units
    cdef double last_fill_ratio
    cdef double last_price
    cdef object pending_order
    cdef double tick_size

    def __init__(self, config=None):
        if config is None:
            config = EnvConfig.default()
        self.config = config
        self.use_full_lob = config.execution_mode.upper() == "FULL_LOB"
        self.tick_size = 1.0 / config.market.price_scale
        self._initialize_environment()

    def _initialize_environment(self):
        self.state = EnvState()
        self.state.cash = self.config.market.initial_balance
        self.state.units = 0.0
        self.state.net_worth = self.config.market.initial_balance
        self.state.prev_net_worth = self.config.market.initial_balance
        self.state.peak_value = self.config.market.initial_balance
        self.state._position_value = 0.0
        self.state.realized_pnl_cum = 0.0
        self.state.step_idx = 0
        self.state.is_bankrupt = False
        self.state.next_order_id = 1

        self.state.taker_fee = self.config.execution.taker_fee
        self.state.maker_fee = self.config.execution.maker_fee
        self.state.spot_cost_taker_fee_bps = max(0.0, float(self.config.execution.taker_fee) * 10000.0)
        self.state.spot_cost_half_spread_bps = max(0.0, float(getattr(self.config.execution, "half_spread_bps", 0.0)))
        self.state.spot_cost_impact_coeff = max(0.0, float(getattr(self.config.execution, "impact_coefficient", 0.0)))
        impact_exp = float(getattr(self.config.execution, "impact_exponent", 1.0))
        if impact_exp <= 0.0:
            impact_exp = 1.0
        self.state.spot_cost_impact_exponent = impact_exp
        self.state.spot_cost_adv_quote = max(0.0, float(getattr(self.config.execution, "adv_quote", 0.0)))
        self.state.profit_close_bonus = self.config.reward.profit_close_bonus
        self.state.loss_close_penalty = self.config.reward.loss_close_penalty
        self.state.bankruptcy_threshold = self.config.risk.bankruptcy_threshold
        self.state.max_drawdown = self.config.risk.max_drawdown
        self.state.trade_frequency_penalty = self.config.reward.trade_frequency_penalty
        self.state.turnover_penalty_coef = self.config.reward.turnover_penalty_coef
        self.state.use_potential_shaping = self.config.reward.use_potential_shaping
        self.state.use_legacy_log_reward = self.config.reward.use_legacy_log_reward
        self.state.gamma = self.config.reward.gamma
        self.state.last_potential = 0.0
        self.state.potential_shaping_coef = self.config.reward.potential_shaping_coef
        self.state.risk_aversion_variance = self.config.reward.risk_aversion_variance
        self.state.risk_aversion_drawdown = self.config.reward.risk_aversion_drawdown
        self.state.use_dynamic_risk = self.config.risk.use_dynamic_risk
        self.state.risk_off_level = self.config.risk.risk_off_level
        self.state.risk_on_level = self.config.risk.risk_on_level
        self.state.max_position_risk_off = self.config.risk.max_position_risk_off
        self.state.max_position_risk_on = self.config.risk.max_position_risk_on
        self.state.price_scale = <long long>self.config.market.price_scale
        self.state.last_agent_fill_ratio = 1.0
        self.state.last_executed_notional = 0.0
        self.state.last_bar_atr = self.config.market.initial_atr

        self.prev_net_worth = self.state.net_worth
        self.prev_units = self.state.units
        self.state.prev_net_worth = self.prev_net_worth
        self.last_fill_ratio = 1.0
        self.last_price = self.config.market.initial_price
        self.pending_order = None

        self.workspace = SimulationWorkspace(64)

        if self.use_full_lob:
            self.lob = CythonLOB()
            self.lob.set_fee_model(self.state.maker_fee, self.state.taker_fee, 0.0)
            self.micro_gen = CyMicrostructureGenerator()
        else:
            self.lob = None
            self.micro_gen = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._initialize_environment()
        obs = self._get_observation()
        info = {}
        return obs, info

    cpdef tuple step(self, object action):
        self.state.step_idx += 1
        self.prev_net_worth = self.state.net_worth
        self.prev_units = self.state.units
        self.workspace.clear_step()
        self.state.last_executed_notional = 0.0

        cdef bint done = False
        cdef dict info = {}
        cdef double reward = 0.0
        cdef double target_fraction = 0.0
        cdef bint is_limit_order = False
        cdef double max_frac = 0.0
        cdef double fg = 0.0
        cdef double ratio = 0.0
        cdef double actual_fill_ratio = 1.0
        cdef double prev_price = self.last_price
        cdef double current_price = prev_price
        cdef double vol_imbalance = 0.0
        cdef int total_trades = 0
        cdef int agent_trade_count = 0
        cdef str closed_reason = ""
        cdef double expected_vol = 0.0
        cdef double filled_vol = 0.0
        cdef double tick_size = self.tick_size
        cdef double current_fraction = 0.0
        cdef double target_units = 0.0
        cdef double needed = 0.0
        cdef double exec_price = 0.0
        cdef double trade_value = 0.0
        cdef double threshold_value = 0.0
        cdef double drawdown_frac = 0.0
        cdef double base_reward = 0.0
        cdef double current_atr = self.config.market.initial_atr
        cdef double open_risk = 0.0
        cdef double drawdown = 0.0
        cdef double penalty_value = 0.0
        cdef double potential = 0.0
        cdef double shaping_reward = 0.0
        cdef double realized_spread = 0.0
        cdef double tick = tick_size
        cdef int i = 0

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) >= 2:
                target_fraction = <double>action[0]
                is_limit_order = bool(int(action[1]) != 0)
            elif len(action) == 1:
                target_fraction = <double>action[0]
        elif isinstance(action, (int, float)):
            target_fraction = <double>action
        else:
            raise ValueError("Unsupported action type")

        if target_fraction > 1.0:
            target_fraction = 1.0
        if target_fraction < -1.0:
            target_fraction = -1.0

        if self.config.risk.use_dynamic_risk:
            fg = self.config.risk.fear_greed_value
            if fg <= self.config.risk.risk_off_level:
                max_frac = self.config.risk.max_position_risk_off
            elif fg >= self.config.risk.risk_on_level:
                max_frac = self.config.risk.max_position_risk_on
            else:
                ratio = (fg - self.config.risk.risk_off_level) / (
                    self.config.risk.risk_on_level - self.config.risk.risk_off_level + 1e-9
                )
                if ratio < 0.0:
                    ratio = 0.0
                elif ratio > 1.0:
                    ratio = 1.0
                max_frac = self.config.risk.max_position_risk_off + ratio * (
                    self.config.risk.max_position_risk_on - self.config.risk.max_position_risk_off
                )
            if abs(target_fraction) > max_frac:
                target_fraction = (target_fraction / abs(target_fraction)) * max_frac

        if not self.use_full_lob and self.state.step_idx == 1:
            prev_price = self.config.market.initial_price
            current_price = prev_price
            self.last_price = current_price

        current_fraction = 0.0
        if self.state.net_worth > 1e-9:
            current_fraction = (self.state.units * prev_price) / self.state.net_worth

        if not is_limit_order:
            target_units = target_fraction * self.state.net_worth / (prev_price if prev_price > 0 else 1.0)
            needed = target_units - self.state.units
            if needed > 1e-9:
                exec_price = prev_price + 0.5 * tick_size
                trade_value = needed * exec_price
                self.state.cash -= trade_value
                self.state.units += needed
                self.state.cash -= trade_value * self.state.taker_fee
                current_price = exec_price
                agent_trade_count = 1
                self.workspace.push_trade(exec_price, needed, <char>1, <char>0, <long long>self.state.step_idx)
                self.state.last_executed_notional += trade_value
            elif needed < -1e-9:
                needed = -needed
                exec_price = prev_price - 0.5 * tick_size
                trade_value = needed * exec_price
                self.state.cash += trade_value
                self.state.units -= needed
                self.state.cash -= trade_value * self.state.taker_fee
                current_price = exec_price
                agent_trade_count = 1
                self.workspace.push_trade(exec_price, needed, <char>-1, <char>0, <long long>self.state.step_idx)
                self.state.last_executed_notional += trade_value
        else:
            self.pending_order = {
                "fraction": target_fraction,
                "side": 1 if target_fraction > current_fraction else -1,
            }
            agent_trade_count = 0
            current_price = prev_price

        total_trades = self.workspace.trade_count
        if total_trades > 0:
            current_price = self.workspace.trade_prices[total_trades - 1]
            vol_imbalance = 0.0
            filled_vol = 0.0
            expected_vol = abs(target_fraction * self.prev_net_worth / (prev_price if prev_price > 0 else 1.0) - self.prev_units)
            for i in range(total_trades):
                if self.workspace.trade_sides[i] > 0:
                    vol_imbalance += self.workspace.trade_qtys[i]
                    filled_vol += self.workspace.trade_qtys[i]
                else:
                    vol_imbalance -= self.workspace.trade_qtys[i]
                    filled_vol += self.workspace.trade_qtys[i]
            if expected_vol > 1e-9:
                actual_fill_ratio = filled_vol / expected_vol
        else:
            current_price = prev_price

        self.state._position_value = self.state.units * current_price
        self.state.net_worth = self.state.cash + self.state._position_value
        if self.state.net_worth > self.state.peak_value:
            self.state.peak_value = self.state.net_worth

        if self.config.risk.use_atr_stop or self.config.risk.use_trailing_stop or self.config.risk.tp_atr_mult > 0:
            if self.prev_units != 0 and self.state.units == 0:
                if self.config.risk.use_atr_stop:
                    closed_reason = "atr_sl_long" if self.prev_units > 0 else "atr_sl_short"
                elif self.config.risk.use_trailing_stop:
                    closed_reason = "trailing_sl_long" if self.prev_units > 0 else "trailing_sl_short"
                elif self.config.risk.tp_atr_mult > 0:
                    closed_reason = "static_tp_long" if self.prev_units > 0 else "static_tp_short"
                if self.config.risk.terminate_on_sl_tp:
                    done = True

        threshold_value = self.config.risk.bankruptcy_threshold * self.config.market.initial_balance
        if self.state.net_worth <= threshold_value + 1e-9:
            self.state.is_bankrupt = True
            closed_reason = "bankrupt"
            done = True
            self.state.cash = 0.0
            self.state.units = 0.0
            self.state._position_value = 0.0

        if self.config.risk.max_drawdown < 1.0:
            drawdown_frac = 0.0
            if self.state.peak_value > 1e-9:
                drawdown_frac = (self.state.peak_value - self.state.net_worth) / self.state.peak_value
            if drawdown_frac >= self.config.risk.max_drawdown - 1e-9:
                closed_reason = "max_drawdown"
                done = True

        ratio = 1.0
        if self.prev_net_worth > 1e-9:
            ratio = self.state.net_worth / self.prev_net_worth
        if ratio < 1e-4:
            ratio = 1e-4
        elif ratio > 10.0:
            ratio = 10.0

        base_reward = log(ratio)
        reward = base_reward
        if self.config.reward.use_potential_shaping:
            current_atr = self.config.market.initial_atr
            self.state.last_bar_atr = current_atr
            if self.state.net_worth > 1e-9:
                open_risk = (abs(self.state.units) * current_atr) / self.state.net_worth
            else:
                open_risk = 0.0
            drawdown = 0.0
            if self.state.peak_value > 1e-9:
                drawdown = (self.state.peak_value - self.state.net_worth) / self.state.peak_value
            penalty_value = self.config.reward.risk_aversion_variance * open_risk + self.config.reward.risk_aversion_drawdown * drawdown
            potential = -tanh(penalty_value) * self.config.reward.potential_shaping_coef
            shaping_reward = self.config.reward.gamma * potential - self.state.last_potential
            reward += shaping_reward
            self.state.last_potential = potential

        if self.config.reward.trade_frequency_penalty > 1e-9:
            reward -= self.config.reward.trade_frequency_penalty * agent_trade_count
        if self.prev_units != 0 and self.state.units == 0:
            if self.state.net_worth > self.prev_net_worth and self.config.reward.profit_close_bonus > 1e-9:
                reward += self.config.reward.profit_close_bonus
            elif self.state.net_worth < self.prev_net_worth and self.config.reward.loss_close_penalty > 1e-9:
                reward -= self.config.reward.loss_close_penalty

        info["vol_imbalance"] = float(vol_imbalance)
        info["trade_intensity"] = int(total_trades)

        if total_trades > 0:
            realized_spread = tick_size / 2.0
        else:
            realized_spread = tick_size / 2.0
        info["realized_spread"] = float(realized_spread)

        if agent_trade_count > 0:
            self.last_fill_ratio = actual_fill_ratio if actual_fill_ratio <= 1.0 else 1.0
        self.state.last_agent_fill_ratio = self.last_fill_ratio
        info["agent_fill_ratio"] = float(self.last_fill_ratio)
        info["closed"] = closed_reason if closed_reason else None

        self.last_price = current_price

        return self._get_observation(), float(reward), bool(done), info

    def _get_observation(self):
        cdef list obs_features = []
        cdef double cash_frac = 0.0
        cdef double pos_frac = 0.0
        if self.state.net_worth > 1e-9:
            cash_frac = self.state.cash / self.state.net_worth
            pos_frac = self.state._position_value / self.state.net_worth
        obs_features.append(float(tanh(cash_frac)))
        obs_features.append(float(tanh(pos_frac)))
        obs_features.append(float(self.last_fill_ratio))
        return np.array(obs_features, dtype=np.float32)

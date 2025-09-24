# distutils: language = c++
cython: boundscheck=False, wraparound=False

from libc.math cimport log, tanh

 

from api.config import EnvConfig
cimport api.config # for potential cdef access if needed (EnvConfig is pure Python class)
from core cimport EnvState, SimulationWorkspace
from exec.engine cimport ExecutionEngine, AgentOrderTracker
from exec.lob_book cimport OrderBook
from micro.generator cimport MicrostructureGenerator
from core_constants cimport PRICE_SCALE, MarketRegime

 

import numpy as np

 

cdef class TradingEnv:
    """
    Торговое окружение с поддержкой режимов FULL_LOB (детальная симуляция стакана)
    и FAST (упрощенный режим). Реализует интерфейс Gym-подобного окружения (reset, step).
    """
    cdef EnvConfig config
    cdef EnvState state
    cdef SimulationWorkspace workspace
    cdef ExecutionEngine engine
    cdef OrderBook lob
    cdef AgentOrderTracker order_tracker
    cdef MicrostructureGenerator micro_gen
    cdef bint use_full_lob
    cdef double prev_net_worth
    cdef double prev_units
    cdef double last_fill_ratio
    cdef double last_price # последний известный рыночный курс для оценки позиции (цена последней сделки или mid)
    cdef object pending_order # для FAST: незавершенный лимитный ордер

    def __init__(self, config: EnvConfig = None):
        if config is None:
            config = EnvConfig.default()
        self.config = config
        self.use_full_lob = (config.execution_mode.upper() == "FULL_LOB")
        self._initialize_environment()

    def _initialize_environment(self):
        """Инициализирует состояние и объекты симуляции."""
        # State initialization
        self.state = EnvState()
        self.state.cash = self.config.market.initial_balance
        self.state.units = 0.0
        self.state.net_worth = self.config.market.initial_balance
        self.state.prev_net_worth = self.config.market.initial_balance
        self.state.peak_value = self.config.market.initial_balance
        self.state._position_value = 0.0
        self.state.step_idx = 0
        self.state.is_bankrupt = False
        self.state.next_order_id = 1
        # Load config parameters into state
        self.state.taker_fee = self.config.execution.taker_fee
        self.state.maker_fee = self.config.execution.maker_fee
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
        self.state.price_scale = self.config.market.price_scale
        # Internal tracking
        self.prev_net_worth = self.state.net_worth
        self.prev_units = self.state.units
        self.last_fill_ratio = 1.0
        self.last_price = self.config.market.initial_price
        self.pending_order = None
        if self.use_full_lob:
            self.lob = OrderBook()
            self.order_tracker = AgentOrderTracker()
            self.micro_gen = MicrostructureGenerator(self.config.micro.events_per_step,
                                                     self.config.micro.p_limit_order,
                                                     self.config.micro.p_market_order,
                                                     self.config.micro.p_cancel_order)
            self.workspace = SimulationWorkspace(100)
            self.engine = ExecutionEngine(self.state, self.lob, self.order_tracker, self.micro_gen, self.workspace)
        else:
            self.lob = None
            self.order_tracker = None
            self.micro_gen = None
            self.workspace = None
            self.engine = None

    def reset(self, seed: int = None):
        """
        Сбрасывает состояние окружения. Если указан seed, задаёт зерно генератора случайностей.
        Возвращает стартовое наблюдение и info.
        """
        if seed is not None:
            np.random.seed(seed)
        self._initialize_environment()
        obs = self._get_observation()
        info = {}
        return obs, info

    cpdef tuple step(self, object action):
        """
        Выполняет шаг среды с заданным действием агента.
        Возвращает (observation, reward, done, info).
        """
        self.state.step_idx += 1
        self.prev_net_worth = self.state.net_worth
        self.prev_units = self.state.units

        cdef bint done = False
        cdef dict info = {}
        cdef double reward = 0.0
        cdef double target_fraction = 0.0
        cdef bint is_limit_order = False
        cdef double max_frac = 0.0
        cdef double fg = 0.0
        cdef double ratio = 0.0
        cdef double actual_fill_ratio = 1.0
        cdef double prev_price = 0.0
        cdef double current_price = 0.0
        cdef double vol_imbalance = 0.0
        cdef int total_trades = 0
        cdef int agent_trade_count = 0
        cdef int i = 0
        cdef str closed_reason = ""
        cdef double expected_vol = 0.0
        cdef double filled_vol = 0.0
        cdef double tick_size = 0.0
        cdef double current_fraction = 0.0
        cdef double target_units = 0.0
        cdef double needed = 0.0
        cdef double exec_price = 0.0
        cdef double trade_value = 0.0
        cdef double threshold_value = 0.0
        cdef double drawdown_frac = 0.0
        cdef double base_reward = 0.0
        cdef double current_atr = 0.0
        cdef double open_risk = 0.0
        cdef double drawdown = 0.0
        cdef double penalty_value = 0.0
        cdef double potential = 0.0
        cdef double shaping_reward = 0.0
        cdef double realized_spread = 0.0
        cdef double tick = 0.0

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) >= 2:
                target_fraction = <double> action[0]
                is_limit_order = bool(int(action[1]) != 0)
            elif len(action) == 1:
                target_fraction = <double> action[0]
        elif isinstance(action, (int, float)):
            target_fraction = <double> action
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
                ratio = (fg - self.config.risk.risk_off_level) / (self.config.risk.risk_on_level - self.config.risk.risk_off_level + 1e-9)
                if ratio < 0.0:
                    ratio = 0.0
                elif ratio > 1.0:
                    ratio = 1.0
                max_frac = self.config.risk.max_position_risk_off + ratio * (self.config.risk.max_position_risk_on - self.config.risk.max_position_risk_off)
            if abs(target_fraction) > max_frac:
                target_fraction = (target_fraction / abs(target_fraction)) * max_frac

        prev_price = self.last_price
        current_price = prev_price

        if self.use_full_lob:
            self.engine.step(target_fraction, not is_limit_order)
            total_trades = self.workspace.n_trades
            vol_imbalance = 0.0
            agent_trade_count = 0

            for i in range(total_trades):
                if self.workspace.agent_taker[i] or self.workspace.agent_maker[i]:
                    agent_trade_count += 1
                    if self.workspace.agent_taker[i]:
                        if self.workspace.trade_side[i] > 0:
                            vol_imbalance += self.workspace.trade_volume[i]
                        elif self.workspace.trade_side[i] < 0:
                            vol_imbalance -= self.workspace.trade_volume[i]

            if agent_trade_count > 0 and vol_imbalance != 0.0:
                expected_vol = 0.0
                if self.prev_net_worth > 1e-9:
                    expected_vol = abs(target_fraction * self.prev_net_worth / (prev_price if prev_price > 0 else 1.0) - self.prev_units)
                filled_vol = abs(self.state.units - self.prev_units)
                if expected_vol > 1e-9:
                    actual_fill_ratio = filled_vol / expected_vol

            if total_trades > 0:
                current_price = self.workspace.trade_price[total_trades - 1] / self.state.price_scale
            elif self.lob.best_ask > 0 and self.lob.best_bid > 0:
                current_price = (self.lob.best_ask + self.lob.best_bid) / (2.0 * self.state.price_scale)
            else:
                current_price = prev_price
            self.last_price = current_price
        else:
            if self.state.step_idx == 1:
                current_price = self.config.market.initial_price
                prev_price = current_price
                self.last_price = current_price
            else:
                current_price = prev_price

            if not is_limit_order:
                tick_size = 1.0 / self.config.market.price_scale
                current_fraction = 0.0
                if self.state.net_worth > 1e-9:
                    current_fraction = (self.state.units * current_price) / self.state.net_worth
                if target_fraction > current_fraction:
                    target_units = target_fraction * self.state.net_worth / (current_price if current_price > 0 else 1.0)
                    needed = target_units - self.state.units
                    if needed > 1e-9:
                        exec_price = current_price + 0.5 * tick_size
                        trade_value = needed * exec_price
                        self.state.cash -= trade_value
                        self.state.units += needed
                        self.state.cash -= trade_value * self.config.execution.taker_fee
                        agent_trade_count = 1
                elif target_fraction < current_fraction:
                    target_units = target_fraction * self.state.net_worth / (current_price if current_price > 0 else 1.0)
                    needed = self.state.units - target_units
                    if needed > 1e-9:
                        exec_price = current_price - 0.5 * tick_size
                        trade_value = needed * exec_price
                        self.state.cash += trade_value
                        self.state.units -= needed
                        self.state.cash -= trade_value * self.config.execution.taker_fee
                        agent_trade_count = 1
            else:
                self.pending_order = {
                    "fraction": target_fraction,
                    "side": (1 if target_fraction > ((self.state.units * current_price) / self.state.net_worth if self.state.net_worth > 0 else 0) else -1),
                }
                agent_trade_count = 0

            self.state.net_worth = self.state.cash + self.state.units * current_price
            vol_imbalance = 0.0
            total_trades = agent_trade_count
            actual_fill_ratio = 1.0
            self.last_price = current_price

        self.state._position_value = self.state.units * current_price
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
            open_risk = 0.0
            if self.state.net_worth > 1e-9:
                open_risk = (abs(self.state.units) * current_atr) / self.state.net_worth
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

        realized_spread = 0.0
        if self.use_full_lob:
            if self.lob.best_ask > 0 and self.lob.best_bid > 0:
                realized_spread = (self.lob.best_ask - self.lob.best_bid) / (2.0 * self.state.price_scale)
        else:
            tick = 1.0 / self.config.market.price_scale
            realized_spread = tick / 2.0
        info["realized_spread"] = float(realized_spread)

        if agent_trade_count > 0:
            self.last_fill_ratio = actual_fill_ratio if actual_fill_ratio <= 1.0 else 1.0
        info["agent_fill_ratio"] = float(self.last_fill_ratio)
        info["closed"] = closed_reason if closed_reason else None

        return self._get_observation(), float(reward), bool(done), info

    def _get_observation(self):
        """
        Формирует вектор наблюдения на основе текущего состояния.
        (Упрощенная реализация; для полного набора признаков следует использовать модуль obs.)
        """
        cdef list obs_features = []
        # Cash and position fractions (with tanh clipping)
        cdef double cash_frac = 0.0
        cdef double pos_frac = 0.0
        if self.state.net_worth > 1e-9:
            cash_frac = self.state.cash / self.state.net_worth
            pos_frac = self.state._position_value / self.state.net_worth
        obs_features.append(float(tanh(cash_frac)))
        obs_features.append(float(tanh(pos_frac)))
        # Last agent fill ratio
        obs_features.append(float(self.last_fill_ratio))
        # (Additional features like indicators, microstructure proxies, etc., are omitted for brevity)
        return np.array(obs_features, dtype=np.float32)

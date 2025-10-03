# cython: language_level=3, language=c++, c_string_type=str, c_string_encoding=utf-8, boundscheck=False, wraparound=False
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from array_specializations cimport ArrayDouble4, ArrayDouble168

from cython cimport Py_ssize_t
from libc.stddef cimport size_t
from libc.stdlib cimport rand
from libc.math cimport log, tanh, fmax, fmin, log1p, fabs, isnan
from libc.stdint cimport uint64_t

cimport cython
cimport numpy as np
import numpy as np

import core_constants as consts
from core_constants cimport MarketRegime, NORMAL, CHOPPY_FLAT, STRONG_TREND, ILLIQUID
from coreworkspace cimport SimulationWorkspace
from fast_lob cimport CythonLOB, OrderBook
from obs_builder cimport build_observation_vector_c

# Инициализируем NumPy C-API
np.import_array()

# вычисляем число признаков для observation_space
# создаём временный буфер достаточной длины
cdef public int N_FEATURES
@cython.cfunc
cdef void _shuffle_events(vector[MicroEvent]& events, vector[unsigned char]& sources):
    cdef size_t n = events.size()
    cdef size_t i
    cdef size_t j
    cdef MicroEvent tmp
    cdef unsigned char tmp_source

    if n <= 1:
        return

    if sources.size() != n:
        return

    i = n
    while i > 1:
        i -= 1
        j = <size_t>(rand() % (<int>i + 1))
        tmp = events[i]
        events[i] = events[j]
        events[j] = tmp
        tmp_source = sources[i]
        sources[i] = sources[j]
        sources[j] = tmp_source

def _compute_n_features() -> int:
    """Вспомогательная функция для подсчёта длины вектора признаков."""
    cdef int max_tokens = 1      # максимальное число токенов (подгоните при необходимости)
    cdef int num_tokens = 1
    norm_cols = np.zeros(0, dtype=np.float32)
    # выделяем буфер заведомо большей длины
    buf = np.empty(256, dtype=np.float32)
    buf.fill(np.nan)
    # вызываем функцию построения наблюдений с фиктивными значениями
    build_observation_vector_c(
        0.0, 0.0, 0.0, 0.0,  # price, prev_price, log_volume_norm, rel_volume
        0.0, 0.0, 0.0, 0.0, 0.0,  # ma5, ma20, rsi14, macd, macd_signal
        0.0, 0.0, 0.0, 0.0,  # momentum, atr, cci, obv
        0.0, 0.0,  # bb_lower, bb_upper
        0.0, 0.0,  # is_high_importance, time_since_event
        0.0, False, False,  # fear_greed_value, has_fear_greed, risk_off_flag
        0.0, 0.0,  # cash, units
        0.0, 0.0,  # last_vol_imbalance, last_trade_intensity
        0.0, 0.0,  # last_realized_spread, last_agent_fill_ratio
        0, max_tokens,  # token_id, max_num_tokens
        num_tokens,
        norm_cols,
        buf
    )
    # определяем индекс последнего заполненного элемента
    cdef Py_ssize_t i
    for i in range(buf.shape[0]):
        if np.isnan(buf[i]):
            return <int>i
    return <int>buf.shape[0]

# вычисляем N_FEATURES один раз при инициализации модуля (без служебных флагов)
N_FEATURES = _compute_n_features()

# Максимальное ожидаемое количество сделок за один шаг симуляции.
# Используется для предварительного выделения памяти под NumPy массивы.
DEF MAX_TRADES_PER_STEP = 10000
DEF MAX_GENERATED_EVENTS_PER_TYPE = 5000

cdef extern from "MarketSimulator.h":

    cdef cppclass MarketSimulator:
        MarketSimulator(
            double* price,
            double* open,
            double* high,
            double* low,
            double* volume_usd,
            size_t n_steps,
            uint64_t seed
        ) except +
        double step(size_t i, double black_swan_probability, bint is_training_mode)
        void set_regime_distribution(const ArrayDouble4& probs)
        void enable_random_shocks(bint enable, double probability_per_step)
        void force_market_regime(MarketRegime regime, size_t start, size_t duration)
        void set_liquidity_seasonality(const ArrayDouble168& multipliers)
        int shock_triggered(size_t i) const
        double get_ma5(size_t i) const
        double get_ma20(size_t i) const
        double get_atr(size_t i) const
        double get_rsi(size_t i) const
        double get_macd(size_t i) const
        double get_macd_signal(size_t i) const
        double get_momentum(size_t i) const
        double get_cci(size_t i) const
        double get_obv(size_t i) const
        double get_bb_lower(size_t i) const
        double get_bb_upper(size_t i) const
        void set_seed(uint64_t seed)



# --- ИСПРАВЛЕННАЯ CYTHON ОБЕРТКА ДЛЯ C++ КЛАССА ---
cdef class MarketSimulatorWrapper:
    """
    Cython-обертка для C++ класса MarketSimulator.
    Управляет жизненным циклом C++ объекта и предоставляет Python-интерфейс.
    """
    cdef MarketSimulator* thisptr  # Указатель на C++ объект
    cdef public object _price_arr_ref, _open_arr_ref, _high_arr_ref, _low_arr_ref, _volume_usd_arr_ref
    cdef double* _price_ptr
    cdef size_t _n_steps
    cdef size_t _last_step_idx
    cdef double _last_price
    cdef object _regime_distribution_cache
    cdef bint _random_shocks_enabled
    cdef double _random_shock_probability

    def __cinit__(self,
                  object price_arr not None,
                  object open_arr not None,
                  object high_arr not None,
                  object low_arr not None,
                  object volume_usd_arr not None,
                  unsigned long long seed=0):

        self.thisptr = NULL
        self._price_ptr = <double*>NULL
        self._n_steps = <size_t>0
        self._last_step_idx = <size_t>0
        self._last_price = 0.0

        cdef np.ndarray[np.float64_t, ndim=1] price = np.ascontiguousarray(price_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] open_ = np.ascontiguousarray(open_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] high = np.ascontiguousarray(high_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] low = np.ascontiguousarray(low_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] volume = np.ascontiguousarray(volume_usd_arr, dtype=np.float64)

        if price.shape[0] != open_.shape[0] or price.shape[0] != high.shape[0] or \
           price.shape[0] != low.shape[0] or price.shape[0] != volume.shape[0]:
            raise ValueError("All OHLCV arrays must have the same length")
        if price.shape[0] == 0:
            raise ValueError("Market simulator requires at least one timestep")

        self._price_arr_ref = price
        self._open_arr_ref = open_
        self._high_arr_ref = high
        self._low_arr_ref = low
        self._volume_usd_arr_ref = volume

        self._n_steps = <size_t>price.shape[0]
        self._last_step_idx = <size_t>0
        self._price_ptr = &price[0]
        self._last_price = price[0]
        self._regime_distribution_cache = None
        self._random_shocks_enabled = False
        self._random_shock_probability = 0.0

        cdef uint64_t c_seed = <uint64_t>seed

        self.thisptr = new MarketSimulator(
            &price[0],
            &open_[0],
            &high[0],
            &low[0],
            &volume[0],
            self._n_steps,
            c_seed
        )
        if not self.thisptr:
            raise MemoryError("Failed to allocate MarketSimulator")

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL
        self._price_ptr = <double*>NULL
        self._n_steps = <size_t>0
        self._last_step_idx = <size_t>0
        self._last_price = 0.0

    cpdef double step(self, int current_step_idx, double black_swan_probability, bint is_training_mode):
        """
        Вызывает C++ метод step для выполнения одного шага симуляции.
        """
        if self.thisptr is NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        if current_step_idx < 0:
            raise ValueError("current_step_idx must be non-negative")

        self._last_step_idx = <size_t>current_step_idx
        self._last_price = self.thisptr.step(self._last_step_idx, black_swan_probability, is_training_mode)
        return self._last_price

    cpdef set_regime_distribution(self, object probabilities):
        """Validate, normalise and forward regime probabilities to the simulator."""
        cdef np.ndarray[np.float64_t, ndim=1] prob_array
        cdef ArrayDouble4 c_probs
        cdef double total
        cdef Py_ssize_t i

        if self.thisptr is NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        if probabilities is None:
            raise ValueError("probabilities must not be None")

        prob_array = np.ascontiguousarray(probabilities, dtype=np.float64)
        if prob_array.ndim != 1 or prob_array.shape[0] != 4:
            raise ValueError("probabilities must be a 1D array of length 4")

        total = 0.0
        for i in range(4):
            if prob_array[i] < 0.0:
                raise ValueError("probabilities must be non-negative")
            total += prob_array[i]

        if total <= 0.0:
            raise ValueError("probabilities must sum to a positive value")

        prob_array = np.ascontiguousarray(prob_array / total, dtype=np.float64)
        for i in range(4):
            c_probs[i] = prob_array[i]

        self.thisptr.set_regime_distribution(c_probs)
        self._regime_distribution_cache = np.array(prob_array, dtype=np.float64, copy=True)

    cpdef enable_random_shocks(self, bint enable, object probability_per_step=None):
        """Enable or disable random shocks after validating the provided probability."""
        cdef double probability
        cdef np.ndarray[np.float64_t, ndim=1] prob_array

        if self.thisptr is NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        if probability_per_step is None:
            probability = 0.0
        elif np.isscalar(probability_per_step):
            probability = float(probability_per_step)
        else:
            prob_array = np.asarray(probability_per_step, dtype=np.float64)
            if prob_array.ndim > 1 or prob_array.size != 1:
                raise ValueError("probability_per_step must be a scalar value")
            probability = float(prob_array.ravel()[0])

        if probability < 0.0:
            raise ValueError("probability_per_step must be non-negative")

        self.thisptr.enable_random_shocks(enable, probability)
        self._random_shocks_enabled = enable
        self._random_shock_probability = probability

    property regime_distribution:
        def __get__(self):
            return self.get_regime_distribution()

    cpdef np.ndarray[np.float64_t, ndim=1] get_regime_distribution(self):
        """Return a copy of the cached regime probabilities."""
        if self._regime_distribution_cache is None:
            raise AttributeError("Regime distribution has not been set")
        return np.array(self._regime_distribution_cache, dtype=np.float64, copy=True)
    cpdef double get_last_price(self):
        if self._price_ptr != <double*>NULL and self._n_steps > 0:
            if self._last_step_idx < self._n_steps:
                return self._price_ptr[self._last_step_idx]
            return self._price_ptr[self._n_steps - 1]
        return self._last_price
    cpdef double get_ma5(self, size_t idx):
        return self.thisptr.get_ma5(idx)
    cpdef double get_ma20(self, size_t idx):
        return self.thisptr.get_ma20(idx)
    cpdef double get_atr(self, size_t idx):
        return self.thisptr.get_atr(idx)
    cpdef double get_rsi(self, size_t idx):
        return self.thisptr.get_rsi(idx)
    cpdef double get_macd(self, size_t idx):
        return self.thisptr.get_macd(idx)
    cpdef double get_macd_signal(self, size_t idx):
        return self.thisptr.get_macd_signal(idx)
    cpdef double get_momentum(self, size_t idx):
        return self.thisptr.get_momentum(idx)
    cpdef double get_cci(self, size_t idx):
        return self.thisptr.get_cci(idx)
    cpdef double get_obv(self, size_t idx):
        return self.thisptr.get_obv(idx)
    cpdef double get_bb_lower(self, size_t idx):
        return self.thisptr.get_bb_lower(idx)
    cpdef double get_bb_upper(self, size_t idx):
        return self.thisptr.get_bb_upper(idx)

    cpdef force_market_regime(self, object regime_name, size_t start_idx, size_t duration):
        """Force a regime via MarketRegime, an integer code, or a string alias."""
        cdef MarketRegime regime

        if isinstance(regime_name, consts.MarketRegime):
            regime = <MarketRegime><int>regime_name
        elif isinstance(regime_name, int):
            if regime_name == consts.MarketRegime.NORMAL:
                regime = NORMAL
            elif regime_name == consts.MarketRegime.CHOPPY_FLAT:
                regime = CHOPPY_FLAT
            elif regime_name == consts.MarketRegime.STRONG_TREND:
                regime = STRONG_TREND
            elif regime_name == consts.MarketRegime.ILLIQUID:
                regime = ILLIQUID
            else:
                raise ValueError("Unknown market regime code: %s" % (regime_name,))
        else:
            regime_str = str(regime_name).strip().lower()
            if regime_str in ("choppy_flat", "flat"):
                regime = CHOPPY_FLAT
            elif regime_str in ("strong_trend", "liquidity_shock", "trend"):
                regime = STRONG_TREND
            elif regime_str in ("illiquid", "illiquidity"):
                regime = ILLIQUID
            else:
                regime = NORMAL

        if self.thisptr is NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self.thisptr.force_market_regime(regime, start_idx, duration)




# ---------- Состояние Среды (НОВЫЙ КЛАСС) ----------
cdef class EnvState:
    """
    Высокопроизводительный контейнер для всех переменных состояния торговой среды.
    Все поля объявлены явно для максимальной производительности и безопасности.
    """
    def __cinit__(self):
        # Этот метод вызывается для инициализации C/C++ полей ПЕРЕД __init__
        self.agent_orders_ptr = new AgentOrderTracker()
        self._position_value = 0.0 # Инициализация нулём
        self.price_scale = consts.PRICE_SCALE
        self._entry_price = -1.0
        self._atr_at_entry = -1.0
        self._initial_sl = -1.0
        self._initial_tp = -1.0
        self._max_price_since_entry = -1.0
        self._min_price_since_entry = -1.0
        self._high_extremum = -1.0
        self._low_extremum = -1.0
        self.fear_greed_value = 0.0
        self.trailing_stop_trigger_count = 0
        self.atr_stop_trigger_count = 0
        self.tp_trigger_count = 0
        self.last_agent_fill_ratio = 0.0
        self.last_event_importance = 0.0
        self.time_since_event = 0.0
        self.last_event_step = -1
        self.token_index = 0
        self.last_realized_spread = 0.0
        self.last_executed_notional = 0.0
        self.last_bar_atr = 0.0
        self.spot_cost_taker_fee_bps = 0.0
        self.spot_cost_half_spread_bps = 0.0
        self.spot_cost_impact_coeff = 0.0
        self.spot_cost_impact_exponent = 1.0
        self.spot_cost_adv_quote = 0.0
        self.lob = None
        self.realized_pnl_cum = 0.0
        if self.agent_orders_ptr is NULL:
            raise MemoryError("Failed to allocate AgentOrderTracker")

    def __dealloc__(self):
        if self.agent_orders_ptr is not NULL:
            del self.agent_orders_ptr
            self.agent_orders_ptr = NULL

    
# ---------- Microstructure Generator (C++ Wrapper) ----------
cdef class CyMicrostructureGenerator:
    """
    Cython-обертка для C++ генератора событий.
    """
    def __cinit__(self,
            base_order_imbalance_ratio=0.8,
            base_cancel_ratio=0.2,
            momentum_factor=0.3,
            mean_reversion_factor=0.5,
            adversarial_factor=0.6):

        self.base_order_imbalance_ratio = base_order_imbalance_ratio
        self.base_cancel_ratio = base_cancel_ratio
        self.thisptr = new CppMicrostructureGenerator()
        if self.thisptr is NULL:
            raise MemoryError("Failed to allocate CppMicrostructureGenerator")

        cdef HawkesParams hp = HawkesParams()
        cdef int i, j
        cdef double base_limit_mu = 0.2
        cdef double buy_multiplier = base_order_imbalance_ratio if base_order_imbalance_ratio > 0 else 1.0

        for i in range(CH_K):
            hp.mu[i] = base_limit_mu

        hp.mu[CH_LIM_BUY] = base_limit_mu * buy_multiplier
        hp.mu[CH_LIM_SELL] = base_limit_mu
        hp.mu[CH_MKT_BUY] = 0.12
        hp.mu[CH_MKT_SELL] = 0.12
        hp.mu[CH_CAN_BUY] = 0.10
        hp.mu[CH_CAN_SELL] = 0.10

        for i in range(CH_K):
            for j in range(CH_K):
                hp.alpha[i][j] = 0.15 if i == j else 0.03
                hp.beta[i][j] = 1.20

        self.thisptr.set_hawkes_params(hp)
        if base_cancel_ratio < 0.0:
            base_cancel_ratio = 0.0
        self.thisptr.set_cancel_rate(base_cancel_ratio)

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    cpdef void set_seed(self, unsigned long long seed):
        """Set the random seed for the underlying generator."""
        if self.thisptr is NULL:
            raise ValueError("Generator is not initialized")
        self.thisptr.set_seed(seed)

    cpdef unsigned long long generate_public_events_cy(self,
            vector[MicroEvent]& out_events,
            CythonLOB lob,
            int timestamp,
            int max_events=MAX_GENERATED_EVENTS_PER_TYPE):

        if self.thisptr is NULL:
            raise ValueError("Generator is not initialized")
        if max_events <= 0:
            return self.thisptr.last_order_id() + 1

        cdef vector[MicroEvent] buffer = vector[MicroEvent]()
        buffer.resize(max_events)

        cdef Py_ssize_t idx
        for idx in range(max_events):
            buffer[idx].timestamp = -1
            buffer[idx].type = MicroEventType.CANCEL
            buffer[idx].is_buy = False
            buffer[idx].price_ticks = 0
            buffer[idx].size = 0.0
            buffer[idx].order_id = 0

        cdef OrderBook* lob_ptr = <OrderBook*>lob.raw_ptr()
        if lob_ptr == NULL:
            raise ValueError("CythonLOB is not initialized")

        self.thisptr.step(lob_ptr[0], timestamp, &buffer[0], max_events)

        for idx in range(max_events):
            if buffer[idx].timestamp == -1:
                break
            out_events.push_back(buffer[idx])

        return self.thisptr.last_order_id() + 1

# ==============================================================================
# ====== НАЧАЛО РЕФАКТОРИНГА: НОВАЯ СТРУКТУРА ДЛЯ АТОМАРНЫХ ИЗМЕНЕНИЙ ======
# ==============================================================================

# C-структура для хранения всех предлагаемых изменений состояния.
# Это ядро паттерна "Propose-and-Commit".
cdef struct StateUpdateDelta:
    # Дельты для основных численных значений счета
    double cash_delta
    double units_delta
    double position_value_delta
    double realized_pnl_delta

    # Полные значения, которые вычисляются в конце и заменяют старые
    double final_net_worth
    double final_peak_value
    double final_last_potential
    double final_last_pos
    double executed_notional

    # Изменения в ордерах агента
    vector[long long] agent_orders_to_remove
    vector[pair[long long, AgentOrderInfo]] new_agent_orders_to_add
    bint clear_all_agent_orders     # Флаг для полной очистки ордеров

    # Изменения в состоянии отслеживания позиции (SL/TP)
    double entry_price
    double atr_at_entry
    double initial_sl
    double initial_tp
    double max_price_since_entry
    double min_price_since_entry
    bint trailing_active

    # Флаги для управления логикой коммита
    bint pos_was_closed         # Позиция была закрыта, нужно сбросить SL/TP
    bint new_pos_opened         # Открыта новая позиция, нужно установить SL/TP
    bint is_bankrupt            # Флаг банкротства
    bint sl_tp_triggered        # Флаг принудительной ликвидации по SL/TP

# ==============================================================================
# ====== ФИНАЛЬНАЯ ФУНКЦИЯ ЯДРА СИМУЛЯЦИИ (ОТРЕФАКТОРЕНА) ===================
# ==============================================================================

@cython.profile(True)
cpdef tuple run_full_step_logic_cython(
    SimulationWorkspace workspace,
    CythonLOB lob, 
    CyMicrostructureGenerator generator,
    float bar_price,
    float bar_open,
    float bar_volume_usd,
    float bar_taker_buy_volume,
    float bar_atr,
    float long_term_atr,
    int   bar_trade_count,
    float bar_fear_greed,
    np.ndarray action,
    EnvState state
    
):
    if action.shape[0] < 2:
        raise ValueError("Action array must have at least 2 elements")

    # --- Объявление переменных ---
    cdef int total_trades_count = 0
    cdef int total_fully_executed_count = 0
    cdef double event_reward = 0.0, reward
    cdef bint done = False
    cdef dict info = {}
    
    cdef double current_max_pos, target_pos_ratio, order_type_signal, current_pos_ratio, delta_ratio, volume_to_trade, offset, price, avg_slippage, fear_greed, ratio, current_price, marketable_volume, remaining_volume, agent_net_taker_flow
    cdef double base_offset, ideal_price, tick_size, price_threshold, tick_size_rand
    cdef str side, order_id_str, reason_str
    cdef long long order_id
    cdef int trades_made, i
    cdef int dynamic_hysteresis_ticks, dynamic_offset_range, offset_in_ticks
    cdef double best_bid, best_ask, best_bid_scaled, best_ask_scaled
    # --- ИСПРАВЛЕНИЕ: Объявляем переменные здесь, чтобы они были доступны во всей функции ---
    cdef double final_price, volatility_factor, old_units_for_commit, prev_net_worth_before_step, step_pnl
    cdef double temp_units, temp_pos_value, vol, fee, old_units, old_value, old_avg_price, realized_pnl, fee_total_event
    cdef double final_cash, final_units, sl_to_check, tp_to_check, atr_for_trail
    cdef double base_stop_loss_price, tick_size_sl, price_offset
    cdef object order_id_to_remove, order_info_dict, reason_obj
    cdef tuple generated_events
    cdef np.ndarray limit_sides_bool, limit_prices, limit_sizes, cancel_sides, public_market_sides_bool, public_market_sizes
    cdef bint is_buy, should_replace_order, is_buy_for_ideal_price, is_agent_taker, is_agent_event
    cdef bint is_taker, is_maker
    cdef int cancelled_order_count = 0
    cdef int trades_start_idx, j, agent_trades_count, dynamic_sl_offset_range, random_ticks_offset, sl_range, trades_this_step
    cdef size_t size_before, size_after
    # Новые переменные для событийной модели
    cdef vector[MicroEvent] all_events
    cdef vector[unsigned char] event_sources
    cdef vector[long long] ids_to_cancel
    cdef pair[long long, long long] closest_order
    cdef MicroEvent agent_event, cancel_ev, current_event
    cdef CythonLOB lob_clone, public_lob
    cdef long long closest_order_price
    cdef unsigned long long agent_order_id
    cdef const AgentOrderInfo* info_ptr
    cdef int trades_made_this_event, executed_count_this_event
    cdef AgentOrderInfo order_info
    
    
    
    # --- Переменные для отслеживания действий агента ---
    cdef double agent_taker_buy_vol_this_step = 0.0
    cdef double agent_taker_sell_vol_this_step = 0.0
    cdef double agent_limit_buy_vol_this_step = 0.0
    cdef double agent_limit_sell_vol_this_step = 0.0
    
    cdef double[::1] prices_all_arr = workspace.prices_all_arr
    cdef double[::1] volumes_all_arr = workspace.volumes_all_arr
    cdef unsigned long long[::1] maker_ids_all_arr = workspace.maker_ids_all_arr
    cdef char[::1] maker_is_agent_all_arr = workspace.maker_is_agent_all_arr
    cdef long long[::1] timestamps_all_arr = workspace.timestamps_all_arr
    cdef char[::1] is_buy_side_all_arr = workspace.is_buy_side_all_arr
    cdef char[::1] taker_is_agent_all_arr = workspace.taker_is_agent_all_arr
    cdef long long[::1] fully_executed_ids_all_arr = workspace.fully_executed_ids_all_arr
    
    assert prices_all_arr.c_contiguous, "Workspace prices_all_arr must be C-contiguous"
    assert volumes_all_arr.c_contiguous, "Workspace volumes_all_arr must be C-contiguous"
    assert maker_ids_all_arr.c_contiguous, "Workspace maker_ids_all_arr must be C-contiguous"
    assert maker_is_agent_all_arr.c_contiguous, "Workspace maker_is_agent_all_arr must be C-contiguous"
    assert timestamps_all_arr.c_contiguous, "Workspace timestamps_all_arr must be C-contiguous"
    assert is_buy_side_all_arr.c_contiguous, "Workspace is_buy_side_all_arr must be C-contiguous"
    assert taker_is_agent_all_arr.c_contiguous, "Workspace taker_is_agent_all_arr must be C-contiguous"
    assert fully_executed_ids_all_arr.c_contiguous, "Workspace fully_executed_ids_all_arr must be C-contiguous"
    
    # ==============================================================
    # 1. ФАЗА ПРЕДЛОЖЕНИЯ (PROPOSE)
    # ==============================================================
    # Все изменения сначала накапливаются в структуре 'delta'
    # Оригинальный объект 'state' в этой фазе не меняется (read-only)
    
    cdef StateUpdateDelta delta
    # Инициализация дельты
    delta.cash_delta = 0.0
    delta.units_delta = 0.0
    delta.position_value_delta = 0.0
    delta.realized_pnl_delta = 0.0
    delta.executed_notional = 0.0
    
    delta.clear_all_agent_orders = False
    delta.pos_was_closed = False
    delta.new_pos_opened = False
    delta.is_bankrupt = False
    delta.sl_tp_triggered = False
    old_units_for_commit = state.units # <--- Сохраняем состояние юнитов до всех изменений
    prev_net_worth_before_step = state.prev_net_worth

    try:
        lob_clone = lob.clone()
        # --- 1.1. Логика действий Агента ---
        current_price = bar_price
        volatility_factor = bar_atr / (current_price * 0.001 + 1e-9)
        
        current_max_pos = 1.0
        if state.use_dynamic_risk:
            fear_greed = bar_fear_greed
            if fear_greed <= state.risk_off_level: current_max_pos = state.max_position_risk_off
            elif fear_greed >= state.risk_on_level: current_max_pos = state.max_position_risk_on
            else:
                ratio = (fear_greed - state.risk_off_level) / (state.risk_on_level - state.risk_off_level)
                current_max_pos = state.max_position_risk_off + ratio * (state.max_position_risk_on - state.max_position_risk_off)
        
        target_pos_ratio, order_type_signal = action[0] * current_max_pos, action[1]
        current_pos_ratio = (state.units * current_price) / (state.prev_net_worth + 1e-8)
        delta_ratio = target_pos_ratio - current_pos_ratio
        
        if abs(delta_ratio) > 0.01:
            should_replace_order = True
            if not state.agent_orders_ptr.is_empty():
                is_buy_for_ideal_price = (delta_ratio > 0)
                base_offset = current_price * 0.001
                ideal_price = current_price - base_offset if is_buy_for_ideal_price else current_price + base_offset
                dynamic_hysteresis_ticks = 3 + int(volatility_factor * 5)
                tick_size = current_price * 0.0001
                price_threshold = dynamic_hysteresis_ticks * tick_size
                # ИЗМЕНЕНО: Масштабируем ideal_price перед поиском ближайшего ордера
                closest_order = state.agent_orders_ptr.find_closest_order(<long long>(ideal_price * 10000))
                if closest_order.first != -1: # Проверяем, что ордер найден
                    closest_order_price = closest_order.second
                    if abs(ideal_price - (closest_order_price / 10000.0)) <= price_threshold:
                        should_replace_order = False
            if should_replace_order:

                delta.clear_all_agent_orders = True # Сохраняем флаг для StateUpdateDelta
                ids_to_cancel = state.agent_orders_ptr.get_all_ids()
                cancelled_order_count = ids_to_cancel.size()
                for i in range(cancelled_order_count):
                    cancel_ev.type = MicroEventType.CANCEL
                    cancel_ev.order_id = ids_to_cancel[i]
                    cancel_ev.timestamp = state.step_idx
                    info_ptr = state.agent_orders_ptr.get_info(ids_to_cancel[i])
                    if info_ptr is not NULL:
                        cancel_ev.is_buy = info_ptr.is_buy_side
                        cancel_ev.price_ticks = info_ptr.price
                    else:
                        cancel_ev.is_buy = True
                        cancel_ev.price_ticks = 0
                    cancel_ev.size = 0.0
                    all_events.push_back(cancel_ev)
                    event_sources.push_back(1)

                is_buy = (delta_ratio > 0)
                volume_to_trade = abs(delta_ratio) * state.prev_net_worth / current_price

                if volume_to_trade * current_price >= 10:
                    # РЕЗЕРВИРУЕМ ID ДЛЯ ОРДЕРА АГЕНТА ЗАРАНЕЕ
                    agent_order_id = 0
                    if order_type_signal > 0.5:
                        agent_order_id = state.next_order_id
                        state.next_order_id += 1

                    # ПРАВИЛЬНАЯ ЛОГИКА IF/ELSE
                    if order_type_signal > 0.5: # Limit order
                        # Восстанавливаем вычисление цены для лимитного ордера
                        offset = current_price * 0.001
                        price = current_price - offset if is_buy else current_price + offset
                        dynamic_offset_range = 2 + int(volatility_factor * 4)
                        tick_size_rand = current_price * 0.0001
                        offset_in_ticks = 0
                        if dynamic_offset_range > 0:
                            offset_in_ticks = -dynamic_offset_range + (rand() % (2 * dynamic_offset_range + 1))
                        price += offset_in_ticks * tick_size_rand

                        # СОЗДАЕМ СОБЫТИЕ ЛИМИТНОГО ОРДЕРА АГЕНТА
                        agent_event.type = MicroEventType.LIMIT
                        agent_event.is_buy = is_buy
                        agent_event.price_ticks = <long long>(price * state.price_scale)
                        agent_event.size = volume_to_trade
                        agent_event.order_id = agent_order_id
                        agent_event.timestamp = state.step_idx
                        all_events.push_back(agent_event)
                        event_sources.push_back(1)
                        # Запоминаем намерение для расчета потока
                        if is_buy: agent_limit_buy_vol_this_step += volume_to_trade
                        else: agent_limit_sell_vol_this_step += volume_to_trade

                    else: # Market order
                        # СОЗДАЕМ СОБЫТИЕ РЫНОЧНОГО ОРДЕРА АГЕНТА
                        agent_event.type = MicroEventType.MARKET
                        agent_event.is_buy = is_buy
                        agent_event.size = volume_to_trade
                        agent_event.price_ticks = 0
                        agent_event.order_id = 0
                        agent_event.timestamp = state.step_idx
                        all_events.push_back(agent_event)
                        event_sources.push_back(1)
                        # Запоминаем намерение для расчета потока
                        if is_buy: agent_taker_buy_vol_this_step += volume_to_trade
                        else: agent_taker_sell_vol_this_step += volume_to_trade

        # --- 1.2. Генерация публичных событий ---
        agent_net_taker_flow = agent_taker_buy_vol_this_step - agent_taker_sell_vol_this_step
        size_before = all_events.size()
        public_lob = lob_clone.clone()
        state.next_order_id = generator.generate_public_events_cy(
            all_events,
            public_lob,
            state.step_idx,
            MAX_GENERATED_EVENTS_PER_TYPE
        )
        size_after = all_events.size()
        for i in range(size_before, size_after):
            event_sources.push_back(0)

        # --- 1.3. Перемешивание всех событий ---
        if all_events.size() > 1:
            _shuffle_events(all_events, event_sources)

        # --- 1.4. НОВЫЙ ЦИКЛ: Исполнение перемешанных событий ---
        for i in range(all_events.size()):
            current_event = all_events[i]
            trades_made_this_event = 0
            executed_count_this_event = 0
            is_agent_event = (event_sources[i] != 0)

            # Используем клон LOB для всех операций в цикле
            if current_event.type == MicroEventType.LIMIT:
                lob_clone.add_limit_order_with_id(
                    current_event.is_buy,
                    current_event.price_ticks,
                    current_event.size,
                    current_event.order_id,
                    state.step_idx,
                    is_agent_event,
                )

            elif current_event.type == MicroEventType.MARKET:
                is_agent_taker = is_agent_event
                trades_made_this_event, fee_total_event = lob_clone.match_market_order_cy(
                    current_event.is_buy, current_event.size, state.step_idx, is_agent_taker,
                    prices_all_arr, volumes_all_arr, maker_ids_all_arr,
                    maker_is_agent_all_arr, timestamps_all_arr,
                    fully_executed_ids_all_arr,
                    total_trades_count, total_fully_executed_count
                )
                if trades_made_this_event > 0:
                    is_buy_side_all_arr[total_trades_count : total_trades_count + trades_made_this_event] = current_event.is_buy
                    taker_is_agent_all_arr[total_trades_count : total_trades_count + trades_made_this_event] = is_agent_taker

            elif current_event.type == MicroEventType.CANCEL:
                if is_agent_event:
                    info_ptr = state.agent_orders_ptr.get_info(current_event.order_id)
                    if info_ptr is not NULL:
                        lob_clone.remove_order(info_ptr.is_buy_side, info_ptr.price, current_event.order_id)
                else:
                    lob_clone.remove_order(current_event.is_buy, current_event.price_ticks, current_event.order_id)

            # --- КРИТИЧЕСКИ ВАЖНО: ОБРАБОТКА PNL ВНУТРИ ЦИКЛА ---
            if trades_made_this_event > 0:
                temp_units = state.units + delta.units_delta
                temp_pos_value = state._position_value + delta.position_value_delta
                trades_start_idx = total_trades_count

                for j in range(trades_start_idx, trades_start_idx + trades_made_this_event):
                    is_taker = taker_is_agent_all_arr[j]
                    is_maker = maker_is_agent_all_arr[j]
                    if not (is_taker or is_maker): continue

                    price = prices_all_arr[j] / state.price_scale
                    vol = volumes_all_arr[j]
                    fee = state.taker_fee if is_taker else state.maker_fee
                    d_units = vol if is_buy_side_all_arr[j] else -vol

                    delta.executed_notional += fabs(vol * price)

                    delta.cash_delta -= d_units * price
                    delta.cash_delta -= vol * price * fee

                    old_units = temp_units
                    old_value = temp_pos_value
                    temp_units += d_units

                    if old_units * temp_units >= 0.0:
                        if abs(temp_units) > abs(old_units):
                            temp_pos_value += d_units * price
                        else:
                            old_avg_price = old_value / old_units if abs(old_units) > 1e-8 else 0.0
                            temp_pos_value += d_units * old_avg_price
                    else: # Разворот
                        old_avg_price = old_value / old_units if abs(old_units) > 1e-8 else 0.0
                        realized_pnl = old_units * (price - old_avg_price)
                        delta.realized_pnl_delta += realized_pnl
                        temp_pos_value = temp_units * price

                delta.units_delta = temp_units - state.units
                delta.position_value_delta = temp_pos_value - state._position_value

                total_trades_count += trades_made_this_event
                total_fully_executed_count += executed_count_this_event

                if total_trades_count > MAX_TRADES_PER_STEP or total_fully_executed_count > MAX_TRADES_PER_STEP:
                    raise MemoryError("Workspace buffer overflow during event processing loop.")

        

        # --- 1.4. Финальные расчеты и обновление состояния ---
        best_bid_scaled = lob_clone.get_best_bid()
        best_ask_scaled = lob_clone.get_best_ask()

        if best_bid_scaled > 0 and best_ask_scaled > 0:
            # ИСПРАВЛЕНО: Используем динамический state.price_scale
            final_price = (best_bid_scaled + best_ask_scaled) / (2.0 * state.price_scale)
        else:
            final_price = current_price
        
        units_after_trades = state.units + delta.units_delta

        # SL/TP Logic - все записи идут в 'delta'
        if state.last_pos != 0 and abs(units_after_trades) < 1e-6: # Position closed
            delta.pos_was_closed = True
        
        delta.final_last_pos = units_after_trades

        if state._entry_price < 0 and abs(units_after_trades) > 1e-6: # New position opened
            delta.new_pos_opened = True
            delta.entry_price = (state.prev_net_worth - (state.cash + delta.cash_delta)) / (units_after_trades + 1e-9)
            delta.atr_at_entry = max(bar_atr, long_term_atr)
            side = 'BUY' if units_after_trades > 0 else 'SELL'
            
            tick_size_sl = current_price * 0.0001
            dynamic_sl_offset_range = 6 + int(volatility_factor * 10)
            sl_range = max(1, dynamic_sl_offset_range - 1)
            random_ticks_offset = 1 + (rand() % sl_range)
            price_offset = random_ticks_offset * tick_size_sl

            if side == 'BUY':
                base_stop_loss_price = delta.entry_price - state.atr_multiplier * delta.atr_at_entry
                delta.initial_sl = base_stop_loss_price - price_offset
                delta.initial_tp = delta.entry_price + state.tp_atr_mult * delta.atr_at_entry
                if not state._trailing_active: delta.max_price_since_entry = delta.entry_price
            else: # 'SELL'
                base_stop_loss_price = delta.entry_price + state.atr_multiplier * delta.atr_at_entry
                delta.initial_sl = base_stop_loss_price + price_offset
                delta.initial_tp = delta.entry_price - state.tp_atr_mult * delta.atr_at_entry
                if not state._trailing_active: delta.min_price_since_entry = delta.entry_price

        # Check for SL/TP triggers
        sl_to_check = state._initial_sl
        tp_to_check = state._initial_tp

        # Если позиция была только что открыта, используем новые уровни из delta
        if delta.new_pos_opened:
            sl_to_check = delta.initial_sl
            tp_to_check = delta.initial_tp

        atr_for_trail = 0.0
        if units_after_trades > 0: # Long position checks
            if state.use_trailing_stop:
                if not state._trailing_active and state._entry_price > 0 and state._atr_at_entry > 0 and final_price >= state._entry_price + state._atr_at_entry * 1.5:
                    delta.trailing_active = True
                    delta.max_price_since_entry = final_price
                elif state._trailing_active and state._max_price_since_entry > 0:
                    delta.max_price_since_entry = max(state._max_price_since_entry, final_price)
                    atr_for_trail = max(state._atr_at_entry, bar_atr)
                    if final_price <= delta.max_price_since_entry - state.trailing_atr_mult * atr_for_trail:
                        info, done = {"closed": "trailing_sl_long"}, state.terminate_on_sl_tp
            if "closed" not in info and state.use_atr_stop and sl_to_check > 0 and final_price <= sl_to_check:
                info, done = {"closed": "atr_sl_long"}, state.terminate_on_sl_tp
            if "closed" not in info and not state._trailing_active and tp_to_check > 0 and final_price >= tp_to_check:
                info, done = {"closed": "static_tp_long"}, state.terminate_on_sl_tp
        elif units_after_trades < 0: # Short position checks
            if state.use_trailing_stop:
                if not state._trailing_active and state._entry_price > 0 and state._atr_at_entry > 0 and final_price <= state._entry_price - state._atr_at_entry * 1.5:
                    delta.trailing_active = True
                    delta.min_price_since_entry = final_price
                elif state._trailing_active and state._min_price_since_entry > 0:
                    delta.min_price_since_entry = min(state._min_price_since_entry, final_price)
                    atr_for_trail = max(state._atr_at_entry, bar_atr)
                    if final_price >= delta.min_price_since_entry + state.trailing_atr_mult * atr_for_trail:
                        info, done = {"closed": "trailing_sl_short"}, state.terminate_on_sl_tp
            if "closed" not in info and state.use_atr_stop and sl_to_check > 0 and final_price >= sl_to_check:
                info, done = {"closed": "atr_sl_short"}, state.terminate_on_sl_tp
            if "closed" not in info and not state._trailing_active and tp_to_check > 0 and final_price <= tp_to_check:
                info, done = {"closed": "static_tp_short"}, state.terminate_on_sl_tp

        
            
            

        # Final state value calculations
        final_cash = state.cash + delta.cash_delta
        final_units = state.units + delta.units_delta
        delta.final_net_worth = final_cash + final_units * final_price

        agent_trades_count = 0
        for i in range(total_trades_count):
            if taker_is_agent_all_arr[i]: # char 1 (True) или 0 (False)
                agent_trades_count += 1

        trades_this_step = agent_trades_count + delta.agent_orders_to_remove.size() + cancelled_order_count
        
        
        delta.final_peak_value = max(state.peak_value, delta.final_net_worth)
        reward, delta.final_last_potential = _compute_reward_cython(
            delta.final_net_worth, prev_net_worth_before_step, event_reward,
            state.use_legacy_log_reward, state.use_potential_shaping,
            state.gamma, state.last_potential, state.potential_shaping_coef, final_units, bar_atr,
            state.risk_aversion_variance, delta.final_peak_value, state.risk_aversion_drawdown,
            trades_this_step, state.trade_frequency_penalty,
            delta.executed_notional, state.turnover_penalty_coef
        )

        step_pnl = delta.final_net_worth - prev_net_worth_before_step
        info['step_pnl'] = step_pnl
        info['turnover'] = <float>delta.executed_notional

        # Risk termination handled in Mediator/RiskGuard — do not set `done` or penalties here.
        # (bankruptcy_threshold / max_drawdown checks removed to avoid double counting)
        pass

    except Exception:
        # Если в фазе вычислений произошла ошибка, 'state' не был изменен.
        # Просто выходим, пробрасывая исключение дальше.
        raise
        

    # ==============================================================
    # 2. ФАЗА СОХРАНЕНИЯ (COMMIT)
    # ==============================================================
    # Этот блок выполняется, только если в 'try' не было исключений.
    # Он атомарно применяет все накопленные изменения к 'state'.
    lob.swap(lob_clone)
    
    # Применяем дельты
    state.cash += delta.cash_delta
    state._position_value += delta.position_value_delta
    state.units += delta.units_delta
    state.realized_pnl_cum += delta.realized_pnl_delta

    # Обновляем ордера агента
    if delta.clear_all_agent_orders:
        state.agent_orders_ptr.clear()
    if not delta.agent_orders_to_remove.empty():
        for i in range(delta.agent_orders_to_remove.size()):
            state.agent_orders_ptr.remove(delta.agent_orders_to_remove[i])
    if not delta.new_agent_orders_to_add.empty():
        for i in range(delta.new_agent_orders_to_add.size()):
            order_id = delta.new_agent_orders_to_add[i].first
            order_info = delta.new_agent_orders_to_add[i].second
            state.agent_orders_ptr.add(order_id, order_info.price, order_info.is_buy_side)

    # --- НАЧАЛО НОВОГО, ЕДИНОГО БЛОКА ОБНОВЛЕНИЯ ЦЕНЫ ВХОДА И SL/TP ---

    

    # Шаг 1: Всегда пересчитываем среднюю цену или сбрасываем состояние, если позиция закрыта.
    if abs(state.units) > 1e-8:
        # Позиция открыта или существует, вычисляем точную средневзвешенную цену.
        state._entry_price = state._position_value / state.units
    else:
        # Позиция только что закрылась или уже была закрыта. Сбрасываем всё.
        state._entry_price = -1.0
        state._position_value = 0.0
        state._atr_at_entry = -1.0
        state._initial_sl = -1.0
        state._initial_tp = -1.0
        state._max_price_since_entry = -1.0
        state._min_price_since_entry = -1.0
        state._high_extremum = -1.0
        state._low_extremum = -1.0
        state._trailing_active = False

    # Шаг 2: Проверяем, не была ли только что открыта НОВАЯ позиция (переход с нуля).
    if abs(old_units_for_commit) < 1e-8 and abs(state.units) > 1e-8:
        # Устанавливаем параметры для новой позиции.
        state._atr_at_entry = max(bar_atr, long_term_atr)
        state._trailing_active = False # Трейлинг всегда выключен вначале.

        volatility_factor = bar_atr / (bar_price * 0.001 + 1e-9)
        tick_size_sl = bar_price * 0.0001
        dynamic_sl_offset_range = 6 + int(volatility_factor * 10)
        sl_range = max(1, dynamic_sl_offset_range - 1)
        random_ticks_offset = 1 + (rand() % sl_range)
        price_offset = random_ticks_offset * tick_size_sl
        
        # Устанавливаем начальные SL/TP и экстремумы цены.
        if state.units > 0: # Long
            state._initial_sl = (state._entry_price - state.atr_multiplier * state._atr_at_entry) - price_offset
            state._initial_tp = state._entry_price + state.tp_atr_mult * state._atr_at_entry
            state._max_price_since_entry = final_price # Начальный максимум = текущая цена
            state._min_price_since_entry = -1.0
            state._high_extremum = final_price
            state._low_extremum = final_price
        else: # Short
            state._initial_sl = (state._entry_price + state.atr_multiplier * state._atr_at_entry) + price_offset
            state._initial_tp = state._entry_price - state.tp_atr_mult * state._atr_at_entry
            state._min_price_since_entry = final_price # Начальный минимум = текущая цена
            state._max_price_since_entry = -1.0
            state._high_extremum = final_price
            state._low_extremum = final_price

    # Шаг 3: Обновляем состояние трейлинг-стопа.
    if delta.trailing_active and not state._trailing_active:
        # Активация трейлинга произошла на этом шаге.
        state._trailing_active = True
    
    # Обновляем пик/дно цены, если трейлинг уже активен.
    if state._trailing_active:
        if state.units > 0:
            state._max_price_since_entry = max(state._max_price_since_entry, final_price)
            state._high_extremum = max(state._high_extremum, state._max_price_since_entry)
            if state._low_extremum < 0:
                state._low_extremum = final_price
            else:
                state._low_extremum = min(state._low_extremum, final_price)
        elif state.units < 0:
            state._min_price_since_entry = min(state._min_price_since_entry, final_price)
            if state._low_extremum < 0:
                state._low_extremum = state._min_price_since_entry
            else:
                state._low_extremum = min(state._low_extremum, state._min_price_since_entry)
    elif abs(state.units) > 1e-8:
        # Если трейлинг ещё не активен, экстремумы отслеживают текущую цену
        if state.units > 0:
            state._high_extremum = max(state._high_extremum, final_price)
            if state._low_extremum < 0:
                state._low_extremum = final_price
            else:
                state._low_extremum = min(state._low_extremum, final_price)
        else:
            if state._high_extremum < 0:
                state._high_extremum = final_price
            else:
                state._high_extremum = max(state._high_extremum, final_price)
            if state._low_extremum >= 0:
                state._low_extremum = min(state._low_extremum, final_price)
            else:
                state._low_extremum = final_price
    
    state.last_pos = delta.final_last_pos
    state.last_executed_notional = delta.executed_notional
    state.last_bar_atr = bar_atr

    # Применяем финальные вычисленные значения
    state.net_worth = delta.final_net_worth
    state.peak_value = delta.final_peak_value
    state.last_potential = delta.final_last_potential
    state.fear_greed_value = bar_fear_greed

    if "closed" in info:
        reason_obj = info["closed"]
        try:
            reason_str = <str>reason_obj
        except Exception:
            reason_str = ""
        if reason_str.startswith("trailing_sl"):
            state.trailing_stop_trigger_count += 1
        elif reason_str.startswith("atr_sl"):
            state.atr_stop_trigger_count += 1
        elif reason_str.startswith("static_tp"):
            state.tp_trigger_count += 1

    # Обработка банкротства
    if delta.is_bankrupt:
        state.is_bankrupt = True
        state.cash, state.units, state.net_worth = 0.0, 0.0, 0.0
        state._position_value = 0.0        
        state._entry_price = -1.0
        state.agent_orders_ptr.clear()

    # --- Добавляем метрики микроструктуры в info для Python ---
    
    # 1. Дисбаланс объема тейкер-ордеров агента
    cdef double vol_imbalance = agent_taker_buy_vol_this_step - agent_taker_sell_vol_this_step
    info['vol_imbalance'] = vol_imbalance

    # 2. Интенсивность торговли (общее число сделок в стакане за шаг)
    # ПРИМЕЧАНИЕ: total_trades_count уже включает и сделки агента, и публичные
    info['trade_intensity'] = <float>total_trades_count

    # 3. Реализованный спред (упрощенная версия - спред на момент окончания шага)
    # Для более точного расчета потребовался бы BBO в момент каждой сделки.
    if lob_clone.get_best_bid() > 0 and lob_clone.get_best_ask() > 0:
        info['realized_spread'] = (lob_clone.get_best_ask() - lob_clone.get_best_bid()) / (2.0 * state.price_scale)
    else:
        info['realized_spread'] = 0.0

    # 4. Коэффициент исполнения тейкер-ордеров агента (fill ratio)
    cdef double agent_intended_taker_volume = 0.0
    if abs(delta_ratio) > 0.01: # Если агент вообще хотел торговать
        agent_intended_taker_volume = abs(delta_ratio) * prev_net_worth_before_step / current_price
    
    cdef double agent_actual_taker_volume = agent_taker_buy_vol_this_step + agent_taker_sell_vol_this_step
    
    if agent_intended_taker_volume > 1e-8:
        info['agent_fill_ratio'] = agent_actual_taker_volume / agent_intended_taker_volume
    else:
        info['agent_fill_ratio'] = 0.0

    state.prev_net_worth = state.net_worth

    # Возвращаем результат
    return reward, done, info


# ==============================================================================
# ====== ВОССТАНОВЛЕННАЯ И ОПТИМИЗИРОВАННАЯ ФУНКЦИЯ ВОЗНАГРАЖДЕНИЯ ===========
# ==============================================================================
cdef inline tuple _compute_reward_cython(
    float net_worth, float prev_net_worth, float event_reward,
    bint use_legacy_log_reward, bint use_potential_shaping,
    float gamma, float last_potential, float potential_shaping_coef,
    float units, float atr, float risk_aversion_variance,
    float peak_value, float risk_aversion_drawdown,
    int trades_this_step, float trade_frequency_penalty,
    double executed_notional, double turnover_penalty_coef
):
    # Вознаграждение с базовым сигналом ΔPnL и опциональным наследуемым логарифмическим компонентом.
    cdef double reward = net_worth - prev_net_worth
    cdef double current_potential = 0.0
    cdef double clipped_ratio, risk_penalty, dd_penalty

    if use_legacy_log_reward:
        clipped_ratio = fmax(0.1, fmin(net_worth / (prev_net_worth + 1e-9), 10.0))
        reward += log(clipped_ratio)

        if use_potential_shaping:
            risk_penalty = 0.0
            dd_penalty = 0.0

            if net_worth > 1e-9 and units != 0 and atr > 0:
                risk_penalty = -risk_aversion_variance * abs(units) * atr / (abs(net_worth) + 1e-9)

            if peak_value > 1e-9:
                dd_penalty = -risk_aversion_drawdown * (peak_value - net_worth) / peak_value

            current_potential = potential_shaping_coef * tanh(risk_penalty + dd_penalty)
            reward += gamma * current_potential - last_potential

    reward -= trades_this_step * trade_frequency_penalty

    if turnover_penalty_coef > 0.0 and executed_notional > 0.0:
        reward -= turnover_penalty_coef * executed_notional

    reward += event_reward

    return reward, current_potential

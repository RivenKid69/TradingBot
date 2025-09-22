# # cython: language_level=3, c_string_type=str, c_string_encoding=utf-8, boundscheck=False, wraparound=False
cdef extern from "cpp_microstructure_generator.h":
    cdef enum MarketEventType:
        NO_EVENT,
        PUBLIC_LIMIT_ADD,
        PUBLIC_MARKET_MATCH,
        PUBLIC_CANCEL_RANDOM,
        AGENT_LIMIT_ADD,
        AGENT_MARKET_MATCH,
        AGENT_CANCEL_SPECIFIC

    cdef struct MarketEvent:
        MarketEventType type
        bint is_buy
        long long price
        double size
        unsigned long long order_id
        int buy_cancel_count
        int sell_cancel_count

    cdef cppclass CppMicrostructureGenerator:
        CppMicrostructureGenerator(double, double, double) except +
        void generate_public_events(
            double, double, double, int, double, double, double, double,
            double, double, int,
            vector[MarketEvent]&,
            long long&
        )
import uuid
from cython cimport Py_ssize_t
from libc.stdlib cimport rand, RAND_MAX, srand, time
from libc.string cimport memset
from libc.math cimport sin, sqrt, log, abs as c_abs, fmax as c_fmax
import numpy as np
cimport numpy as np
from libc.stddef cimport size_t
cimport cython
import random
import pandas as pd
from fast_lob cimport CythonLOB
# Инициализируем NumPy C-API
np.import_array()
# вычисляем число признаков для observation_space
# создаём временный буфер достаточной длины
import numpy as _np
cdef public int N_FEATURES

def _compute_n_features() -> int:
    """Вспомогательная функция для подсчёта длины вектора признаков."""
    cdef int max_tokens = 1      # максимальное число токенов (подгоните при необходимости)
    cdef int num_tokens = 1
    cdef _np.ndarray[float, ndim=1] norm_cols = _np.zeros(0, dtype=_np.float32)
    # выделяем буфер заведомо большей длины
    cdef _np.ndarray[float, ndim=1] buf = _np.zeros(256, dtype=_np.float32)
    # вызываем функцию построения наблюдений с фиктивными значениями
    build_observation_vector_c(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        0.0, False,
        False,       # risk_off_flag
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0, max_tokens,
        num_tokens,
        norm_cols,
        buf
    )
    # определяем индекс последнего заполненного элемента
    # NB: функция использует feature_idx как счётчик; здесь длина равна количеству ненулевых значений
    return buf.shape[0] if False else  # оставьте строку, чтобы mypy/flake не ругались

# вычисляем N_FEATURES один раз при инициализации модуля (без служебных флагов)
N_FEATURES = _compute_n_features()
from libcpp.algorithm cimport random_shuffle

from core_constants cimport PRICE_SCALE, MarketRegime

# Импортируем нашу новую Cython-обертку вместо старых классов
from fast_lob cimport CythonLOB
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.math cimport isnan
from libc.math cimport log, tanh, fmax, fmin, log1p

cdef extern from "AgentOrderTracker.h":
    cdef struct AgentOrderInfo:
        long long price
        bint is_buy_side
    cdef cppclass AgentOrderTracker:
        AgentOrderTracker() except +; void add(long long, long long, bint); void remove(long long); bint contains(long long); const AgentOrderInfo* get_info(long long); void clear(); bint is_empty(); vector[long long] get_all_ids(); const pair[const long long, AgentOrderInfo]* get_first_order_info(); pair[long long, long long] find_closest_order(long long);

# Инициализируем NumPy C-API
np.import_array()

# Максимальное ожидаемое количество сделок за один шаг симуляции.
# Используется для предварительного выделения памяти под NumPy массивы.
DEF MAX_TRADES_PER_STEP = 10000
DEF MAX_GENERATED_EVENTS_PER_TYPE = 5000
cdef class SimulationWorkspace:
    """
    Контейнер для предварительно выделенных NumPy массивов,
    чтобы избежать их пересоздания на каждом шаге симуляции.
    """
    # Объявляем публичные атрибуты для быстрого доступа из Cython
    cdef public double[::1] prices_all_arr
    cdef public double[::1] volumes_all_arr
    cdef public unsigned long long[::1] maker_ids_all_arr
    cdef public char[::1] maker_is_agent_all_arr
    cdef public int[::1] timestamps_all_arr
    cdef public char[::1] is_buy_side_all_arr
    cdef public char[::1] taker_is_agent_all_arr
    cdef public unsigned long long[::1] fully_executed_ids_all_arr

    def __cinit__(self):
        # Выделяем память один раз при создании объекта
        self.prices_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.float64, order="C")
        self.volumes_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.float64, order="C")
        self.maker_ids_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.uint64, order="C")
        self.maker_is_agent_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.int8, order="C")
        self.timestamps_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.int32, order="C")
        self.is_buy_side_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.int8, order="C")
        self.taker_is_agent_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.int8, order="C")
        self.fully_executed_ids_all_arr = np.empty(MAX_TRADES_PER_STEP, dtype=np.uint64, order="C")

        # Закрепляем массивы, чтобы GC не освободил их
        self._buf_refs = [
            self.prices_all_arr, self.volumes_all_arr, self.maker_ids_all_arr,
            self.maker_is_agent_all_arr, self.timestamps_all_arr,
            self.is_buy_side_all_arr, self.taker_is_agent_all_arr,
            self.fully_executed_ids_all_arr
        ]

# --- ИСПРАВЛЕННЫЕ C++ ДЕКЛАРАЦИИ ---
# Объявляем, что мы будем использовать C++ класс MarketSimulator с полным набором указателей
# PATCH‑ID:P15_LOBSTATE_comment  (no code change – for traceability)
cdef extern from "MarketSimulator.h":
    
    cdef cppclass MarketSimulator:
        MarketSimulator(
            float* price_arr, float* open_arr, float* high_arr, float* low_arr,
            float* volume_usd_arr,
            size_t n_steps, int window_size,
            int ma5_window, int ma20_window, int atr_window, int rsi_window,
            int macd_fast, int macd_slow, int macd_signal, int momentum_window,
            int cci_window, int bb_window, double bb_std_dev, int obv_ma_window
        ) except +
        double step(size_t, double, bint)
        double get_last_price() const
        double get_ma5(size_t)
        double get_ma20(size_t)
        double get_atr(size_t)
        double get_rsi(size_t)
        double get_macd(size_t)
        double get_macd_signal(size_t)
        double get_momentum(size_t)
        double get_cci(size_t)
        double get_obv(size_t)
        double get_bb_lower(size_t)
        double get_bb_upper(size_t)
        void force_market_regime(MarketRegime, size_t, size_t)



# --- ИСПРАВЛЕННАЯ CYTHON ОБЕРТКА ДЛЯ C++ КЛАССА ---
cdef class MarketSimulatorWrapper:
    """
    Cython-обертка для C++ класса MarketSimulator.
    Управляет жизненным циклом C++ объекта и предоставляет Python-интерфейс.
    """
    cdef MarketSimulator* thisptr  # Указатель на C++ объект
    # НОВОЕ: Добавляем члены класса для хранения ссылок на массивы
    cdef public object _price_arr_ref, _open_arr_ref, _high_arr_ref, _low_arr_ref, _volume_usd_arr_ref

    def __cinit__(self,
                  # Основные сырые данные
                  np.ndarray[float, ndim=1, mode="c"] price_arr,
                  np.ndarray[float, ndim=1, mode="c"] open_arr,
                  np.ndarray[float, ndim=1, mode="c"] high_arr,
                  np.ndarray[float, ndim=1, mode="c"] low_arr,
                  np.ndarray[float, ndim=1, mode="c"] volume_usd_arr,
                  # Параметры индикаторов и симуляции
                  int rsi_window, int macd_fast, int macd_slow, int macd_signal,
                  int momentum_window, int cci_window, int bb_window, double bb_std_dev,
                  int ma5_window, int ma20_window, int atr_window, int window_size, int obv_ma_window):
        
        # НОВОЕ: Сохраняем ссылки. Это увеличивает счетчик ссылок,
        # и GC не тронет массивы, пока жив этот объект.
        self._price_arr_ref = price_arr
        self._open_arr_ref = open_arr
        self._high_arr_ref = high_arr
        self._low_arr_ref = low_arr
        self._volume_usd_arr_ref = volume_usd_arr
        cdef size_t n_steps = price_arr.shape[0]
        self.thisptr = new MarketSimulator(
            &price_arr[0], &open_arr[0], &high_arr[0], &low_arr[0],
            &volume_usd_arr[0],
            n_steps, window_size,
            # --- Аргументы должны идти в этом порядке ---
            ma5_window, 
            ma20_window, 
            atr_window, 
            rsi_window, 
            macd_fast,
            macd_slow, 
            macd_signal, 
            momentum_window, 
            cci_window, 
            bb_window, 
            bb_std_dev,
            obv_ma_window
        )
        if not self.thisptr:
            raise MemoryError("Failed to allocate MarketSimulator")

    def __dealloc__(self):
        # Освобождение C++ объекта требует GIL
        with gil:
            if self.thisptr:
                del self.thisptr
                self.thisptr = NULL

    def double step(self, int current_step_idx, double black_swan_probability, bint is_training_mode):
        """
        Вызывает C++ метод step для выполнения одного шага симуляции.
        """
        with nogil:
            self.thisptr.step(current_step_idx, black_swan_probability, is_training_mode)
    cpdef double get_last_price(self):
        return self.thisptr.get_last_price()
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

    cpdef force_market_regime(self, str regime_name, size_t start_idx, size_t duration):
        cdef MarketRegime regime
        if regime_name == "choppy_flat" or regime_name == "flat":
            regime = CHOPPY_FLAT
        elif regime_name in ("strong_trend", "liquidity_shock", "trend"):
            regime = STRONG_TREND
        else:
            regime = NORMAL
        
        with nogil:
            self.thisptr.force_market_regime(regime, start_idx, duration)



# НОВАЯ И ЕДИНСТВЕННАЯ ВЕРСИЯ ФУНКЦИИ ПОСТРОЕНИЯ НАБЛЮДЕНИЙ
cpdef void build_observation_vector_c(
    
    # Аргументы-признаки (все остаются как есть)
    float price, float  prev_price, float  log_volume_norm, float  rel_volume,
    float ma5, float  ma20, float  rsi14, float  macd, float  macd_signal,
    float momentum, float  atr, float  cci, float  obv,
    float bb_lower, float  bb_upper,
    float is_high_importance, float  time_since_event,
    float fear_greed_value, bint has_fear_greed, bint risk_off_flag,
    # Аргументы состояния агента
    float cash, float  units,
    float last_vol_imbalance, float  last_trade_intensity,
    float last_realized_spread, float  last_agent_fill_ratio,
    # Аргументы для нормализованных колонок
    int token_id,  int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    # ВЫХОДНОЙ ПАРАМЕТР: Указатель на заранее выделенный массив
    float[::1] out_features
) nogil:
    # `nogil` здесь возможен, так как мы убрали все Python-операции,
    # кроме чтения из `norm_cols_values`, что безопасно.

    cdef int feature_idx = 0
    cdef float feature_val # Временная переменная для вычислений

    # --- Базовые признаки ---
    out_features[feature_idx] = <float>price; feature_idx += 1
    out_features[feature_idx] = <float>log_volume_norm; feature_idx += 1
    out_features[feature_idx] = <float>rel_volume; feature_idx += 1
    cdef bint ma5_valid = not isnan(ma5)
    out_features[feature_idx] = <float>ma5 if ma5_valid else 0.0; feature_idx += 1
    out_features[feature_idx] = <float>ma5_valid; feature_idx += 1
    cdef bint ma20_valid = not isnan(ma20)
    out_features[feature_idx] = <float>ma20 if ma20_valid else 0.0; feature_idx += 1
    out_features[feature_idx] = <float>ma20_valid; feature_idx += 1
    out_features[feature_idx] = <float>rsi14; feature_idx += 1
    out_features[feature_idx] = <float>macd; feature_idx += 1
    out_features[feature_idx] = <float>macd_signal; feature_idx += 1
    out_features[feature_idx] = <float>momentum; feature_idx += 1
    out_features[feature_idx] = <float>atr; feature_idx += 1
    out_features[feature_idx] = <float>cci; feature_idx += 1
    out_features[feature_idx] = <float>obv; feature_idx += 1
    

    # --- Производные признаки (с заменой np.* на libc.math.*) ---
    cdef double ret_1h = tanh((price - prev_price) / (prev_price + 1e-8))
    out_features[feature_idx] = <float>ret_1h; feature_idx += 1

    # Используем log1p для большей точности при малых значениях x в log(1+x)
    cdef double vol_24h = tanh(log1p(atr / (price + 1e-8)))
    out_features[feature_idx] = <float>vol_24h; feature_idx += 1

    # --- Признаки состояния агента ---
    cdef double position_value = units * price
    cdef double total_worth = cash + position_value
    
    feature_val = 1.0 if total_worth <= 1e-8 else cash / total_worth
    out_features[feature_idx] = <float>fmax(0.0, fmin(feature_val, 1.0)); feature_idx += 1 # clip(liquidity, 0, 1)

    feature_val = 0.0 if total_worth <= 1e-8 else position_value / total_worth
    out_features[feature_idx] = <float>tanh(feature_val); feature_idx += 1 # tanh(position_ratio)
    
    out_features[feature_idx] = <float>tanh(last_vol_imbalance); feature_idx += 1
    out_features[feature_idx] = <float>tanh(last_trade_intensity); feature_idx += 1

    feature_val = fmax(-0.1, fmin(last_realized_spread, 0.1)) # clip(spread, -0.1, 0.1)
    out_features[feature_idx] = <float>feature_val; feature_idx += 1
    out_features[feature_idx] = <float>last_agent_fill_ratio; feature_idx += 1
    # --- Microstructure: OFI / QueueImbalance / Microprice (proxies) ---

    # OFI-проxi: знак изменения mid * интенсивность объёма
    cdef double mid_ret     = tanh((price - prev_price) / (prev_price + 1e-8))
    cdef double vol_int     = tanh(rel_volume)
    cdef double ofi_proxy   = mid_ret * vol_int
    if FIDX_OFI < 0: FIDX_OFI = feature_idx
    out_features[feature_idx] = <float>ofi_proxy; feature_idx += 1

    # Queue-imbalance: уже передан last_vol_imbalance → нормализуем tanh
    cdef double qimb = tanh(last_vol_imbalance)
    if FIDX_QIMB < 0: FIDX_QIMB = feature_idx
    out_features[feature_idx] = <float>qimb; feature_idx += 1

    # Microprice proxy: смещение от mid пропорционально половине спрэда и дисбалансу
    cdef double micro_dev = 0.5 * last_realized_spread * qimb
    if FIDX_MICRO < 0: FIDX_MICRO = feature_idx
    out_features[feature_idx] = <float>micro_dev; feature_idx += 1

    # --- Признаки Bollinger Bands ---
    cdef double bb_width = bb_upper - bb_lower
    cdef bint bb_valid = not isnan(bb_lower) # Проверки одного достаточно
    cdef double min_bb_width = price * 0.0001
    feature_val = 0.5 if not bb_valid or bb_width <= min_bb_width else (price - bb_lower) / (bb_width + 1e-9)
    out_features[feature_idx] = <float>fmax(-1.0, fmin(feature_val, 2.0)); feature_idx += 1 # clip(bb_pos, -1, 2)
    
    feature_val = 0.0 if not bb_valid else (bb_width / (price + 1e-8))
    out_features[feature_idx] = <float>bb_valid; feature_idx += 1

    # --- Прочие признаки ---
    out_features[feature_idx] = <float>is_high_importance; feature_idx += 1
    out_features[feature_idx] = <float>tanh(time_since_event / 24.0); feature_idx += 1

    # --- Нормализованные колонки ---
    # Этот цикл все еще работает с Python-списком, но без создания новых объектов.
    # Это компромисс, чтобы не менять сигнатуру для `norm_cols_values`.
    cdef Py_ssize_t i
    for i in range(norm_cols_values.shape[0]):
        feature_val = fmax(-3.0, fmin(norm_cols_values[i], 3.0)) # clip
        out_features[feature_idx] = feature_val
        feature_idx += 1

    # --- Индекс страха и жадности ---
    if has_fear_greed:
        feature_val = fmax(-3.0, fmin(fear_greed_value / 100.0, 3.0)) # clip
        out_features[feature_idx] = <float>feature_val
        feature_idx += 1

    # --- One-Hot Encoding для токенов (с паддингом) ---
    if max_num_tokens > 1:
        cdef int start_idx = feature_idx
        # ПРАВИЛЬНО: Всегда проходим по МАКСИМАЛЬНОМУ числу токенов, заполняя нулями
        for i in range(max_num_tokens):
            out_features[start_idx + i] = 0.0
    
        # ПРАВИЛЬНО: Ставим единицу только если токен валиден для ТЕКУЩЕГО ассета
        if 0 <= token_id < max_num_tokens:
            out_features[start_idx + token_id] = 1.0
        
        # ПРАВИЛЬНО: Сдвигаем индекс всегда на МАКСИМАЛЬНОЕ число токенов
        feature_idx += max_num_tokens


# ---------- Состояние Среды (НОВЫЙ КЛАСС) ----------
cdef class EnvState:
    """
    Высокопроизводительный контейнер для всех переменных состояния торговой среды.
    Все поля объявлены явно для максимальной производительности и безопасности.
    """
    # Состояние счета и позиции
    cdef public float cash, units, net_worth, prev_net_worth, peak_value
    cdef public double _position_value      # накопленная стоимость позиции (может быть <0)

    # Состояние симуляции
    cdef public int step_idx
    cdef public bint is_bankrupt
    cdef public AgentOrderTracker* agent_orders_ptr
    cdef public unsigned long long next_order_id

    # Параметры Stop-Loss / Take-Profit
    cdef public bint use_atr_stop, use_trailing_stop, terminate_on_sl_tp, _trailing_active
    cdef public double _entry_price, _atr_at_entry, _initial_sl_level, _initial_tp_level
    cdef public double _max_price_since_entry, _min_price_since_entry
    cdef public double atr_multiplier, trailing_atr_mult, tp_atr_mult
    cdef public double last_pos

    # Параметры вознаграждения и комиссий
    cdef public double profit_close_bonus, loss_close_penalty, taker_fee, maker_fee
    cdef public double bankruptcy_threshold, bankruptcy_penalty, max_drawdown

    # Параметры Shaping-а и риска
    cdef public bint use_potential_shaping, use_dynamic_risk, use_legacy_log_reward
    cdef public double gamma, last_potential, potential_shaping_coef
    cdef public double risk_aversion_variance, risk_aversion_drawdown
    cdef public double trade_frequency_penalty, turnover_penalty_coef, risk_off_level, risk_on_level
    cdef public double max_position_risk_off, max_position_risk_on, market_impact_k
    cdef public long long price_scale # <-- ДОБАВЛЕНО ПОЛЕ ДЛЯ МАСШТАБА

    def __cinit__(self):
        # Этот метод вызывается для инициализации C/C++ полей ПЕРЕД __init__
        self.agent_orders_ptr = new AgentOrderTracker()
        self._position_value = 0.0 # Инициализация нулём
        self.price_scale = PRICE_SCALE
        if self.agent_orders_ptr is NULL:
            raise MemoryError("Failed to allocate AgentOrderTracker")

    def __dealloc__(self):
        with gil:
            if self.agent_orders_ptr is not NULL:
                del self.agent_orders_ptr
                self.agent_orders_ptr = NULL

    
# ---------- Microstructure Generator (C++ Wrapper) ----------
cdef class CyMicrostructureGenerator:
    """
    Cython-обертка для C++ генератора событий.
    """
    cdef CppMicrostructureGenerator* thisptr
    cdef public double base_order_imbalance_ratio
    cdef public double base_cancel_ratio

    def __cinit__(self,
            base_order_imbalance_ratio=0.8,
            base_cancel_ratio=0.2,
            momentum_factor=0.3,
            mean_reversion_factor=0.5,
            adversarial_factor=0.6):

        self.base_order_imbalance_ratio = base_order_imbalance_ratio
        self.base_cancel_ratio = base_cancel_ratio
        self.thisptr = new CppMicrostructureGenerator(
            momentum_factor, mean_reversion_factor, adversarial_factor)

    def __dealloc__(self):
        with gil:
            if self.thisptr:
                del self.thisptr
                self.thisptr = NULL

    cpdef long long generate_public_events_cy(self,
            vector[MarketEvent]& out_events,
            unsigned long long next_public_order_id,
            double bar_price, double bar_open, double bar_volume_usd,
            int bar_trade_count, double bar_taker_buy_volume,
            double agent_net_taker_flow, double agent_limit_buy_vol,
            double agent_limit_sell_vol, int timestamp):

        cdef long long next_id_ref = next_public_order_id
        self.thisptr.generate_public_events(
            bar_price, bar_open, bar_volume_usd, bar_trade_count, bar_taker_buy_volume,
            agent_net_taker_flow, agent_limit_buy_vol, agent_limit_sell_vol,
            self.base_order_imbalance_ratio,
            self.base_cancel_ratio,
            timestamp,
            out_events,
            next_id_ref
        )
        return next_id_ref

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
    double initial_sl_level
    double initial_tp_level
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
    assert action.shape[0] >= 2, f"Action array must have at least 2 elements, but got shape {action.shape}"

    # --- Объявление переменных ---
    cdef int total_trades_count = 0
    cdef int total_fully_executed_count = 0
    cdef double event_reward = 0.0, reward
    cdef bint done = False
    cdef dict info = {}
    
    cdef double current_max_pos, target_pos_ratio, order_type_signal, current_pos_ratio, delta_ratio, volume_to_trade, offset, price, avg_slippage, fear_greed, ratio, current_price, marketable_volume, remaining_volume, agent_net_taker_flow
    cdef str side, order_id_str
    cdef long long order_id
    cdef int trades_made, i
    cdef double best_bid, best_ask
    # --- ИСПРАВЛЕНИЕ: Объявляем переменные здесь, чтобы они были доступны во всей функции ---
    cdef double final_price, volatility_factor, old_units_for_commit, prev_net_worth_before_step, step_pnl
    cdef object order_id_to_remove, order_info_dict
    cdef tuple generated_events
    cdef np.ndarray limit_sides_bool, limit_prices, limit_sizes, cancel_sides, public_market_sides_bool, public_market_sizes
    cdef bint is_buy
    cdef int cancelled_order_count = 0
    # Новые переменные для событийной модели
    cdef vector[MarketEvent] all_events
    cdef MarketEvent agent_event
    
    
    
    # --- Переменные для отслеживания действий агента ---
    cdef double agent_taker_buy_vol_this_step = 0.0
    cdef double agent_taker_sell_vol_this_step = 0.0
    cdef double agent_limit_buy_vol_this_step = 0.0
    cdef double agent_limit_sell_vol_this_step = 0.0
    
    cdef double[::1] prices_all_arr = workspace.prices_all_arr
    cdef double[::1] volumes_all_arr = workspace.volumes_all_arr
    cdef long long[::1] maker_ids_all_arr = workspace.maker_ids_all_arr
    cdef char[::1] maker_is_agent_all_arr = workspace.maker_is_agent_all_arr
    cdef int[::1] timestamps_all_arr = workspace.timestamps_all_arr
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
    delta.executed_notional = 0.0
    
    delta.clear_all_agent_orders = False
    delta.pos_was_closed = False
    delta.new_pos_opened = False
    delta.is_bankrupt = False
    delta.sl_tp_triggered = False
    old_units_for_commit = state.units # <--- Сохраняем состояние юнитов до всех изменений
    prev_net_worth_before_step = state.prev_net_worth

    try:
        cdef CythonLOB lob_clone = lob.clone()
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
            cdef bint should_replace_order = True
            if not state.agent_orders_ptr.is_empty():
                is_buy_for_ideal_price = (delta_ratio > 0)
                base_offset = current_price * 0.001
                ideal_price = current_price - base_offset if is_buy_for_ideal_price else current_price + base_offset
                cdef int dynamic_hysteresis_ticks = 3 + int(volatility_factor * 5)
                cdef double tick_size = current_price * 0.0001
                cdef double price_threshold = dynamic_hysteresis_ticks * tick_size
                # ИЗМЕНЕНО: Масштабируем ideal_price перед поиском ближайшего ордера
                cdef pair[long long, long long] closest_order = state.agent_orders_ptr.find_closest_order(<long long>(ideal_price * 10000))
                if closest_order.first != -1: # Проверяем, что ордер найден
                    cdef long long closest_order_price = closest_order.second
                    if abs(ideal_price - (closest_order_price / 10000.0)) <= price_threshold:
                        should_replace_order = False
            if should_replace_order:
                
                delta.clear_all_agent_orders = True # Сохраняем флаг для StateUpdateDelta
                cdef vector[long long] ids_to_cancel = state.agent_orders_ptr.get_all_ids()
                cancelled_order_count = ids_to_cancel.size()
                for i in range(cancelled_order_count):
                    cdef MarketEvent cancel_ev
                    cancel_ev.type = AGENT_CANCEL_SPECIFIC
                    cancel_ev.order_id = ids_to_cancel[i]
                    all_events.push_back(cancel_ev)
                
                is_buy = (delta_ratio > 0)
                volume_to_trade = abs(delta_ratio) * state.prev_net_worth / current_price
                
                if volume_to_trade * current_price >= 10:
                    # РЕЗЕРВИРУЕМ ID ДЛЯ ОРДЕРА АГЕНТА ЗАРАНЕЕ
                    cdef unsigned long long agent_order_id = 0
                    if order_type_signal > 0.5:
                        agent_order_id = state.next_order_id
                        state.next_order_id += 1

                    # ПРАВИЛЬНАЯ ЛОГИКА IF/ELSE
                    if order_type_signal > 0.5: # Limit order
                        # Восстанавливаем вычисление цены для лимитного ордера
                        offset = current_price * 0.001
                        price = current_price - offset if is_buy else current_price + offset
                        cdef int dynamic_offset_range = 2 + int(volatility_factor * 4)
                        cdef double tick_size_rand = current_price * 0.0001
                        cdef int offset_in_ticks = 0
                        if dynamic_offset_range > 0:
                            offset_in_ticks = -dynamic_offset_range + (rand() % (2 * dynamic_offset_range + 1))
                        price += offset_in_ticks * tick_size_rand

                        # СОЗДАЕМ СОБЫТИЕ ЛИМИТНОГО ОРДЕРА АГЕНТА
                        agent_event.type = AGENT_LIMIT_ADD
                        agent_event.is_buy = is_buy
                        agent_event.price = <long long>(price * state.price_scale)
                        agent_event.size = volume_to_trade
                        agent_event.order_id = agent_order_id
                        all_events.push_back(agent_event)
                        # Запоминаем намерение для расчета потока
                        if is_buy: agent_limit_buy_vol_this_step += volume_to_trade
                        else: agent_limit_sell_vol_this_step += volume_to_trade

                    else: # Market order
                        # СОЗДАЕМ СОБЫТИЕ РЫНОЧНОГО ОРДЕРА АГЕНТА
                        agent_event.type = AGENT_MARKET_MATCH
                        agent_event.is_buy = is_buy
                        agent_event.size = volume_to_trade
                        all_events.push_back(agent_event)
                        # Запоминаем намерение для расчета потока
                        if is_buy: agent_taker_buy_vol_this_step += volume_to_trade
                        else: agent_taker_sell_vol_this_step += volume_to_trade

        # --- 1.2. Генерация публичных событий ---
        agent_net_taker_flow = agent_taker_buy_vol_this_step - agent_taker_sell_vol_this_step
        state.next_order_id = generator.generate_public_events_cy(
            all_events, # Передаем вектор для добавления публичных событий
            state.next_order_id,
            bar_price, bar_open, bar_volume_usd, bar_trade_count,
            bar_taker_buy_volume, agent_net_taker_flow,
            agent_limit_buy_vol_this_step, agent_limit_sell_vol_this_step,
            state.step_idx
        )

        # --- 1.3. Перемешивание всех событий ---
        if all_events.size() > 1:
            random_shuffle(all_events.begin(), all_events.end())

        # --- 1.4. НОВЫЙ ЦИКЛ: Исполнение перемешанных событий ---
        cdef MarketEvent current_event
        cdef int trades_made_this_event, executed_count_this_event

        for i in range(all_events.size()):
            current_event = all_events[i]
            trades_made_this_event = 0
            executed_count_this_event = 0

            # Используем клон LOB для всех операций в цикле
            if current_event.type == AGENT_LIMIT_ADD or current_event.type == PUBLIC_LIMIT_ADD:
                lob_clone.add_limit_order(current_event.is_buy, current_event.price, current_event.size, current_event.order_id, (current_event.type == AGENT_LIMIT_ADD), state.step_idx)

            elif current_event.type == AGENT_MARKET_MATCH or current_event.type == PUBLIC_MARKET_MATCH:
                cdef bint is_agent_taker = (current_event.type == AGENT_MARKET_MATCH)
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

            elif current_event.type == AGENT_CANCEL_SPECIFIC:
                cdef const AgentOrderInfo* info_ptr = state.agent_orders_ptr.get_info(current_event.order_id)
                if info_ptr is not NULL:
                    lob_clone.remove_order(info_ptr.is_buy_side, info_ptr.price, current_event.order_id)

            elif current_event.type == PUBLIC_CANCEL_RANDOM:
                if current_event.buy_cancel_count > 0:
                    lob_clone.cancel_random_public_orders(True, current_event.buy_cancel_count)
                if current_event.sell_cancel_count > 0:
                    lob_clone.cancel_random_public_orders(False, current_event.sell_cancel_count)

            # --- КРИТИЧЕСКИ ВАЖНО: ОБРАБОТКА PNL ВНУТРИ ЦИКЛА ---
            if trades_made_this_event > 0:
                cdef double temp_units = state.units + delta.units_delta
                cdef double temp_pos_value = state._position_value + delta.position_value_delta
                cdef int trades_start_idx = total_trades_count

                for j in range(trades_start_idx, trades_start_idx + trades_made_this_event):
                    is_taker = taker_is_agent_all_arr[j]
                    is_maker = maker_is_agent_all_arr[j]
                    if not (is_taker or is_maker): continue

                    price = prices_all_arr[j] / state.price_scale
                    vol = volumes_all_arr[j]
                    fee = state.taker_fee if is_taker else state.maker_fee
                    d_units = vol if is_buy_side_all_arr[j] else -vol

                    delta.executed_notional += c_abs(vol * price)

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
                        delta.cash_delta += realized_pnl
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
        
        cdef double units_after_trades = state.units + delta.units_delta

        # SL/TP Logic - все записи идут в 'delta'
        if state.last_pos != 0 and abs(units_after_trades) < 1e-6: # Position closed
            delta.pos_was_closed = True
        
        delta.final_last_pos = units_after_trades

        if state._entry_price < 0 and abs(units_after_trades) > 1e-6: # New position opened
            delta.new_pos_opened = True
            delta.entry_price = (state.prev_net_worth - (state.cash + delta.cash_delta)) / (units_after_trades + 1e-9)
            delta.atr_at_entry = max(bar_atr, long_term_atr)
            side = 'BUY' if units_after_trades > 0 else 'SELL'
            
            cdef double base_stop_loss_price, tick_size_sl, price_offset
            cdef int dynamic_sl_offset_range, random_ticks_offset
            tick_size_sl = current_price * 0.0001
            dynamic_sl_offset_range = 6 + int(volatility_factor * 10)
            cdef int sl_range = max(1, dynamic_sl_offset_range - 1)
            random_ticks_offset = 1 + (rand() % sl_range)
            price_offset = random_ticks_offset * tick_size_sl

            if side == 'BUY':
                base_stop_loss_price = delta.entry_price - state.atr_multiplier * delta.atr_at_entry
                delta.initial_sl_level = base_stop_loss_price - price_offset
                delta.initial_tp_level = delta.entry_price + state.tp_atr_mult * delta.atr_at_entry
                if not state._trailing_active: delta.max_price_since_entry = delta.entry_price
            else: # 'SELL'
                base_stop_loss_price = delta.entry_price + state.atr_multiplier * delta.atr_at_entry
                delta.initial_sl_level = base_stop_loss_price + price_offset
                delta.initial_tp_level = delta.entry_price - state.tp_atr_mult * delta.atr_at_entry
                if not state._trailing_active: delta.min_price_since_entry = delta.entry_price
        
        # Check for SL/TP triggers
        cdef double sl_to_check = state._initial_sl_level
        cdef double tp_to_check = state._initial_tp_level

        # Если позиция была только что открыта, используем новые уровни из delta
        if delta.new_pos_opened:
            sl_to_check = delta.initial_sl_level
            tp_to_check = delta.initial_tp_level

        cdef double atr_for_trail = 0.0
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
        cdef double final_cash = state.cash + delta.cash_delta
        cdef double final_units = state.units + delta.units_delta
        delta.final_net_worth = final_cash + final_units * final_price
        
        cdef int agent_trades_count = 0
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
    state.cash += delta.cash_delta # PnL от разворотов уже учтён внутри cash_delta
    state._position_value += delta.position_value_delta
    state.units += delta.units_delta

    # Обновляем ордера агента
    if delta.clear_all_agent_orders:
        state.agent_orders_ptr.clear()
    if not delta.agent_orders_to_remove.empty():
        for i in range(delta.agent_orders_to_remove.size()):
            state.agent_orders_ptr.remove(delta.agent_orders_to_remove[i])
    if not delta.new_agent_orders_to_add.empty():
        for i in range(delta.new_agent_orders_to_add.size()):
            order_id = delta.new_agent_orders_to_add[i].first
            cdef AgentOrderInfo info = delta.new_agent_orders_to_add[i].second
            state.agent_orders_ptr.add(order_id, info.price, info.is_buy_side)

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
        state._initial_sl_level = -1.0
        state._initial_tp_level = -1.0
        state._max_price_since_entry = -1.0
        state._min_price_since_entry = -1.0
        state._trailing_active = False

    # Шаг 2: Проверяем, не была ли только что открыта НОВАЯ позиция (переход с нуля).
    if abs(old_units_for_commit) < 1e-8 and abs(state.units) > 1e-8:
        # Устанавливаем параметры для новой позиции.
        state._atr_at_entry = max(bar_atr, long_term_atr)
        state._trailing_active = False # Трейлинг всегда выключен вначале.

        cdef double volatility_factor = bar_atr / (bar_price * 0.001 + 1e-9)
        cdef double tick_size_sl = bar_price * 0.0001
        cdef int dynamic_sl_offset_range = 6 + int(volatility_factor * 10)
        cdef int sl_range = max(1, dynamic_sl_offset_range - 1)
        cdef int random_ticks_offset = 1 + (rand() % sl_range)
        cdef double price_offset = random_ticks_offset * tick_size_sl
        
        # Устанавливаем начальные SL/TP и экстремумы цены.
        if state.units > 0: # Long
            state._initial_sl_level = (state._entry_price - state.atr_multiplier * state._atr_at_entry) - price_offset
            state._initial_tp_level = state._entry_price + state.tp_atr_mult * state._atr_at_entry
            state._max_price_since_entry = final_price # Начальный максимум = текущая цена
            state._min_price_since_entry = -1.0
        else: # Short
            state._initial_sl_level = (state._entry_price + state.atr_multiplier * state._atr_at_entry) + price_offset
            state._initial_tp_level = state._entry_price - state.tp_atr_mult * state._atr_at_entry
            state._min_price_since_entry = final_price # Начальный минимум = текущая цена
            state._max_price_since_entry = -1.0

    # Шаг 3: Обновляем состояние трейлинг-стопа.
    if delta.trailing_active and not state._trailing_active:
        # Активация трейлинга произошла на этом шаге.
        state._trailing_active = True
    
    # Обновляем пик/дно цены, если трейлинг уже активен.
    if state._trailing_active:
        if state.units > 0:
            state._max_price_since_entry = max(state._max_price_since_entry, final_price)
        elif state.units < 0:
            state._min_price_since_entry = min(state._min_price_since_entry, final_price)
    
    state.last_pos = delta.final_last_pos

    # Применяем финальные вычисленные значения
    state.net_worth = delta.final_net_worth
    state.peak_value = delta.final_peak_value
    state.last_potential = delta.final_last_potential
    
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
cdef _compute_reward_cython(
    float net_worth, float prev_net_worth, float event_reward,
    bint use_legacy_log_reward, bint use_potential_shaping,
    float gamma, float last_potential, float potential_shaping_coef,
    float units, float atr, float risk_aversion_variance,
    float peak_value, float risk_aversion_drawdown,
    int trades_this_step, float trade_frequency_penalty,
    double executed_notional, double turnover_penalty_coef
):
    ""
    Вознаграждение с базовым сигналом ΔPnL и опциональным наследуемым логарифмическим компонентом.
    ""
    cdef double reward = net_worth - prev_net_worth
    cdef double current_potential = 0.0

    if use_legacy_log_reward:
        cdef double clipped_ratio = fmax(0.1, fmin(net_worth / (prev_net_worth + 1e-9), 10.0))
        reward += log(clipped_ratio)

        if use_potential_shaping:
            cdef double risk_penalty = 0.0
            cdef double dd_penalty = 0.0

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

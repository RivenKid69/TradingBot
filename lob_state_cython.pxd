# cython: language_level=3, language=c++
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from array_specializations cimport ArrayDouble6, ArrayDouble6x6

from fast_lob cimport OrderBook, CythonLOB
from core_constants cimport MarketRegime

cdef extern from "cpp_microstructure_generator.h":
    cdef enum MicroEventType "MicroEventType":
        LIMIT "MicroEventType::LIMIT"
        MARKET "MicroEventType::MARKET"
        CANCEL "MicroEventType::CANCEL"

    cdef int CH_LIM_BUY
    cdef int CH_LIM_SELL
    cdef int CH_MKT_BUY
    cdef int CH_MKT_SELL
    cdef int CH_CAN_BUY
    cdef int CH_CAN_SELL
    cdef int CH_K

    cdef struct MicroEvent:
        MicroEventType type
        bint is_buy
        long long price_ticks
        double size
        unsigned long long order_id
        int timestamp

    cdef cppclass HawkesParams:
        HawkesParams() except +
        ArrayDouble6 mu
        ArrayDouble6x6 alpha
        ArrayDouble6x6 beta

    cdef cppclass SizeDist:
        double lognorm_m
        double lognorm_s
        double min_size
        double max_size

    cdef cppclass PlacementProfile:
        double at_best_prob
        double geometric_p
        int max_levels

    cdef cppclass ShockParams:
        bint enabled
        double prob_per_step
        double intensity_scale
        double price_bps_mu
        double price_bps_std

    cdef cppclass BlackSwanParams:
        bint enabled
        double prob_per_step
        double crash_min
        double crash_max
        double mania_min
        double mania_max
        int cooldown_steps
        double intensity_scale
        int duration_steps

    cdef struct MicroFeatures:
        long long best_bid
        long long best_ask
        double mid
        double spread_ticks
        double depth_bid_top1
        double depth_ask_top1
        double depth_bid_top5
        double depth_ask_top5
        double imbalance_top1
        double imbalance_top5
        ArrayDouble6 lambda_hat
        int last_trade_sign
        double last_trade_size

    cdef cppclass CppMicrostructureGenerator:
        CppMicrostructureGenerator() except +
        void set_seed(unsigned long long seed)
        void set_hawkes_params(const HawkesParams& hp)
        void set_size_models(const SizeDist& limit_sz, const SizeDist& market_sz)
        void set_placement_profile(const PlacementProfile& pp)
        void set_cancel_rate(double base_cancel_rate)
        void set_flash_shocks(const ShockParams& sp)
        void set_black_swan(const BlackSwanParams& bp)
        void set_regime(MarketRegime regime)
        void reset(long long mid0_ticks, long long best_bid_ticks=0, long long best_ask_ticks=0)
        int step(OrderBook& lob, int timestamp, MicroEvent* out_events, int cap)
        MicroFeatures current_features(const OrderBook& lob) const
        unsigned long long last_order_id() const
        void copy_lambda_hat(double* out) const

cdef extern from "AgentOrderTracker.h":
    cdef struct AgentOrderInfo:
        long long price
        bint is_buy_side

    cdef cppclass AgentOrderTracker:
        AgentOrderTracker() except +
        void add(long long order_id, long long price, bint is_buy)
        void remove(long long order_id)
        bint contains(long long order_id)
        const AgentOrderInfo* get_info(long long order_id)
        void clear()
        bint is_empty()
        vector[long long] get_all_ids()
        const pair[const long long, AgentOrderInfo]* get_first_order_info()
        pair[long long, long long] find_closest_order(long long price_ticks)

cdef class CyMicrostructureGenerator:
    cdef CppMicrostructureGenerator* thisptr
    cdef public double base_order_imbalance_ratio
    cdef public double base_cancel_ratio
    cpdef void set_seed(self, unsigned long long seed)
    cpdef unsigned long long generate_public_events_cy(
        self,
        vector[MicroEvent]& out_events,
        CythonLOB lob,
        int timestamp,
        int max_events=*
    )

cdef class EnvState:
    cdef public float cash
    cdef public float units
    cdef public float net_worth
    cdef public float prev_net_worth
    cdef public float peak_value
    cdef public double _position_value
    cdef public int step_idx
    cdef public bint is_bankrupt
    cdef AgentOrderTracker* agent_orders_ptr
    cdef public unsigned long long next_order_id

    cdef public double realized_pnl_cum

    cdef public bint use_atr_stop
    cdef public bint use_trailing_stop
    cdef public bint terminate_on_sl_tp
    cdef public bint _trailing_active

    cdef public double _entry_price
    cdef public double _atr_at_entry
    cdef public double _initial_sl
    cdef public double _initial_tp
    cdef public double _max_price_since_entry
    cdef public double _min_price_since_entry
    cdef public double _high_extremum
    cdef public double _low_extremum

    cdef public double atr_multiplier
    cdef public double trailing_atr_mult
    cdef public double tp_atr_mult
    cdef public double last_pos

    cdef public double taker_fee
    cdef public double maker_fee
    cdef public double spot_cost_taker_fee_bps
    cdef public double spot_cost_half_spread_bps
    cdef public double spot_cost_impact_coeff
    cdef public double spot_cost_impact_exponent
    cdef public double spot_cost_adv_quote
    cdef public double profit_close_bonus
    cdef public double loss_close_penalty
    cdef public double bankruptcy_threshold
    cdef public double bankruptcy_penalty
    cdef public double max_drawdown

    cdef public bint use_potential_shaping
    cdef public bint use_dynamic_risk
    cdef public bint use_legacy_log_reward
    cdef public double gamma
    cdef public double last_potential
    cdef public double potential_shaping_coef
    cdef public double risk_aversion_variance
    cdef public double risk_aversion_drawdown
    cdef public double trade_frequency_penalty
    cdef public double turnover_penalty_coef
    cdef public double last_executed_notional
    cdef public double last_bar_atr
    cdef public double risk_off_level
    cdef public double risk_on_level
    cdef public double max_position_risk_off
    cdef public double max_position_risk_on
    cdef public double market_impact_k
    cdef public double fear_greed_value
    cdef public long long price_scale

    cdef public int trailing_stop_trigger_count
    cdef public int atr_stop_trigger_count
    cdef public int tp_trigger_count

    cdef public double last_agent_fill_ratio
    cdef public double last_event_importance
    cdef public double time_since_event
    cdef public int last_event_step
    cdef public int token_index
    cdef public double last_realized_spread
    cdef public object lob

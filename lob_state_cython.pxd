# cython: language_level=3, language=c++
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "cpp_microstructure_generator.h":
    cdef enum MarketEventType:
        NO_EVENT
        PUBLIC_LIMIT_ADD
        PUBLIC_MARKET_MATCH
        PUBLIC_CANCEL_RANDOM
        AGENT_LIMIT_ADD
        AGENT_MARKET_MATCH
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
        CppMicrostructureGenerator(double momentum_factor, double mean_reversion_factor, double adversarial_factor) except +
        void generate_public_events(
            double bar_price,
            double bar_open,
            double bar_volume_usd,
            int bar_trade_count,
            double bar_taker_buy_volume,
            double agent_net_taker_flow,
            double agent_limit_buy_vol,
            double agent_limit_sell_vol,
            double base_order_imbalance_ratio,
            double base_cancel_ratio,
            int timestamp,
            vector[MarketEvent]& out_events,
            long long& next_public_order_id
        )

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
    cpdef long long generate_public_events_cy(
        self,
        vector[MarketEvent]& out_events,
        unsigned long long next_public_order_id,
        double bar_price,
        double bar_open,
        double bar_volume_usd,
        int bar_trade_count,
        double bar_taker_buy_volume,
        double agent_net_taker_flow,
        double agent_limit_buy_vol,
        double agent_limit_sell_vol,
        int timestamp
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

# cython: language_level=3, language=c++

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libc.stddef cimport size_t

ctypedef unsigned long long uint64_t
ctypedef unsigned int uint32_t

cdef extern from "OrderBook.h":
    cdef struct FeeModel:
        double maker_fee
        double taker_fee
        double slip_k

    cdef struct Order:
        uint64_t id
        double volume
        cpp_bool is_agent
        int timestamp
        int ttl_steps

    cdef cppclass OrderBook:
        OrderBook() except +
        void add_limit_order(cpp_bool is_buy_side,
                             long long price_ticks,
                             double volume,
                             uint64_t order_id,
                             cpp_bool is_agent,
                             int timestamp)
        void remove_order(cpp_bool is_buy_side,
                          long long price_ticks,
                          uint64_t order_id)
        uint32_t get_queue_position(uint64_t order_id) const
        int match_market_order(cpp_bool is_buy_side,
                               double volume,
                               int timestamp,
                               cpp_bool taker_is_agent,
                               double* out_prices,
                               double* out_volumes,
                               int* out_is_buy,
                               int* out_is_self,
                               long long* out_ids,
                               int max_len,
                               double* out_fee_total)
        long long get_best_bid()
        long long get_best_ask()
        void prune_stale_orders(int current_step, int max_age)
        void cancel_random_public_orders(cpp_bool is_buy_side, int n)
        cpp_bool contains_order(uint64_t order_id) const
        OrderBook* clone() const
        void swap(OrderBook& other)
        void set_fee_model(const FeeModel& fm)
        void set_seed(unsigned long long seed)
        cpp_bool set_order_ttl(uint64_t order_id, int ttl_steps)
        int decay_ttl_and_cancel(void (*on_cancel)(const Order&))

cdef class CythonLOB:
    cdef OrderBook* thisptr
    cdef uint64_t _next_id

    cpdef set_seed(self, unsigned long long seed)
    cpdef set_fee_model(self, double maker_fee, double taker_fee, double slip_k)
    cdef uint64_t _assign_order_id(self, unsigned long long order_id) noexcept
    cpdef tuple add_limit_order(self,
                                bint is_buy_side,
                                long long price_ticks,
                                double volume,
                                int timestamp,
                                bint taker_is_agent)
    cpdef tuple add_limit_order_with_id(self,
                                        bint is_buy_side,
                                        long long price_ticks,
                                        double volume,
                                        unsigned long long order_id,
                                        int timestamp,
                                        bint taker_is_agent)
    cpdef remove_order(self, bint is_buy_side, long long price_ticks, unsigned long long order_id)
    cpdef bint set_order_ttl(self, unsigned long long order_id, int ttl_steps)
    cpdef list decay_ttl_and_cancel(self)
    cpdef tuple match_market_order(self,
                                   bint is_buy_side,
                                   double volume,
                                   int timestamp,
                                   bint taker_is_agent,
                                   double[::1] out_prices,
                                   double[::1] out_volumes,
                                   int[::1] out_is_buy,
                                   int[::1] out_is_self,
                                   long long[::1] out_ids,
                                   int max_len)
    cpdef prune_stale_orders(self, int current_step, int max_age)
    cpdef cancel_random_orders_batch(self, object sides)
    cpdef bint contains_order(self, unsigned long long order_id)
    cpdef long long get_best_bid(self)
    cpdef long long get_best_ask(self)
    cpdef size_t raw_ptr(self)
    cpdef CythonLOB clone(self)
    cpdef swap(self, CythonLOB other)

# cython: language_level=3
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8, boundscheck=False, wraparound=False

cimport cython
cimport numpy as cnp
import numpy as np
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector

ctypedef unsigned long long uint64_t
ctypedef unsigned int uint32_t

cdef extern from "OrderBook.h":
    cdef struct FeeModel:
        double maker_fee
        double taker_fee
        double slip_k

    cppclass OrderBook:
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

cdef class CythonLOB:
    """
    Тонкая обёртка над C++ OrderBook.
    Совместима с вызовами из mediator/execution_sim:
      - add_limit_order(...) -> (order_id, queue_pos)
      - match_market_order(...) -> (n_trades, fee_total)
    """
    cdef OrderBook* thisptr
    cdef uint64_t _next_id

    def __cinit__(self):
        self.thisptr = new OrderBook()
        self._next_id = <uint64_t>1

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr
            self.thisptr = NULL

    cpdef set_seed(self, unsigned long long seed):
        self.thisptr.set_seed(seed)

    cpdef set_fee_model(self, double maker_fee, double taker_fee, double slip_k):
        cdef FeeModel fm
        fm.maker_fee = maker_fee
        fm.taker_fee = taker_fee
        fm.slip_k = slip_k
        self.thisptr.set_fee_model(fm)

    cpdef tuple add_limit_order(self,
                                bint is_buy_side,
                                long long price_ticks,
                                double volume,
                                int timestamp,
                                bint taker_is_agent):
        """
        Генерируем 64-битный ID на стороне обёртки, чтобы вернуть его вызывающему коду.
        Также возвращаем позицию в очереди по этому ID (если недоступна — None).
        """
        cdef uint64_t oid = self._next_id
        self._next_id += 1
        with nogil:
            self.thisptr.add_limit_order(is_buy_side, price_ticks, volume, oid,
                                         taker_is_agent, timestamp)
        cdef uint32_t pos = self.thisptr.get_queue_position(oid)
        # 0xFFFFFFFF обычно используют как «нет позиции»
        if pos == <uint32_t>0xFFFFFFFF:
            return (oid, None)
        return (oid, pos)

    cpdef remove_order(self, bint is_buy_side, long long price_ticks, unsigned long long order_id):
        with nogil:
            self.thisptr.remove_order(is_buy_side, price_ticks, <uint64_t>order_id)

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
                                   int max_len):
        """
        Возвращает (n_trades, fee_total). Буферы — одномерные C-contiguous memoryview.
        """
        cdef int m = max_len
        # ограничим по фактической длине всех буферов
        if m > out_prices.shape[0]:
            m = out_prices.shape[0]
        if m > out_volumes.shape[0]:
            m = out_volumes.shape[0]
        if m > out_is_buy.shape[0]:
            m = out_is_buy.shape[0]
        if m > out_is_self.shape[0]:
            m = out_is_self.shape[0]
        if m > out_ids.shape[0]:
            m = out_ids.shape[0]
        if m <= 0:
            return (0, 0.0)

        cdef double fee_total = 0.0
        cdef int n_trades
        with nogil:
            n_trades = self.thisptr.match_market_order(
                is_buy_side, volume, timestamp, taker_is_agent,
                &out_prices[0], &out_volumes[0],
                &out_is_buy[0], &out_is_self[0],
                &out_ids[0], m, &fee_total)

        return (n_trades, fee_total)

    cpdef prune_stale_orders(self, int current_step, int max_age):
        with nogil:
            self.thisptr.prune_stale_orders(current_step, max_age)

    cpdef cancel_random_orders_batch(self, cnp.ndarray[cnp.bool_t, ndim=1] sides):
        cdef Py_ssize_t i, n = sides.shape[0]
        for i in range(n):
            self.thisptr.cancel_random_public_orders(<cpp_bool>bool(sides[i]), 1)

    cpdef bint contains_order(self, unsigned long long order_id):
        return bool(self.thisptr.contains_order(<uint64_t>order_id))

    cpdef long long get_best_bid(self):
        return <long long>self.thisptr.get_best_bid()

    cpdef long long get_best_ask(self):
        return <long long>self.thisptr.get_best_ask()

    cpdef size_t raw_ptr(self):
        """
        Сырой указатель на внутренний OrderBook* (для дружеских Cython/C++-обёрток).
        Безопасен: ничего не меняет и не освобождает.
        """
        return <size_t>self.thisptr

    cpdef size_t raw_ptr(self):
        """
        Сырой указатель на внутренний OrderBook* (для Cython/C++-интеграции).
        Использовать только для friend-обёрток; руками не разыменовывать в Python.
        """
        return <size_t>self.thisptr

    cpdef CythonLOB clone(self):
        cdef CythonLOB other = CythonLOB.__new__(CythonLOB)
        other.thisptr = self.thisptr.clone()
        other._next_id = self._next_id
        return other

    cpdef swap(self, CythonLOB other):
        with nogil:
            self.thisptr.swap(other.thisptr[0])

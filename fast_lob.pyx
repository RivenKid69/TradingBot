# cython: language_level=3
# cython: language=c++
# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8, boundscheck=False, wraparound=False

cimport cython
cimport numpy as cnp
import numpy as np
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libc.stddef cimport size_t

cdef vector[uint64_t]* _ttl_buf = NULL

cdef void _collect_cancel(const Order& o) noexcept nogil:
    if _ttl_buf is not NULL:
        _ttl_buf.push_back(o.id)

cdef class CythonLOB:
    """
    Тонкая обёртка над C++ OrderBook.
    Совместима с вызовами из mediator/execution_sim:
      - add_limit_order(...) -> (order_id, queue_pos)
      - match_market_order(...) -> (n_trades, fee_total)
    """

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

    cdef inline uint64_t _assign_order_id(self, unsigned long long order_id) noexcept:
        cdef uint64_t oid
        if order_id != 0:
            oid = <uint64_t>order_id
            if oid >= self._next_id:
                self._next_id = oid + 1
        else:
            oid = self._next_id
            self._next_id += 1
        return oid

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
        cdef uint64_t oid = self._assign_order_id(0)
        self.thisptr.add_limit_order(is_buy_side, price_ticks, volume, oid,
                                     taker_is_agent, timestamp)
        cdef uint32_t pos = self.thisptr.get_queue_position(oid)
        # 0xFFFFFFFF обычно используют как «нет позиции»
        if pos == <uint32_t>0xFFFFFFFF:
            return (oid, None)
        return (oid, pos)

    cpdef tuple add_limit_order_with_id(self,
                                        bint is_buy_side,
                                        long long price_ticks,
                                        double volume,
                                        unsigned long long order_id,
                                        int timestamp,
                                        bint taker_is_agent):
        cdef uint64_t oid = self._assign_order_id(order_id)
        self.thisptr.add_limit_order(is_buy_side, price_ticks, volume, oid,
                                     taker_is_agent, timestamp)
        cdef uint32_t pos = self.thisptr.get_queue_position(oid)
        if pos == <uint32_t>0xFFFFFFFF:
            return (oid, None)
        return (oid, pos)

    cpdef remove_order(self, bint is_buy_side, long long price_ticks, unsigned long long order_id):
        self.thisptr.remove_order(is_buy_side, price_ticks, <uint64_t>order_id)
    cpdef bint set_order_ttl(self, unsigned long long order_id, int ttl_steps):
        return bool(self.thisptr.set_order_ttl(<uint64_t>order_id, ttl_steps))

    cpdef list decay_ttl_and_cancel(self):
        cdef vector[uint64_t] acc
        global _ttl_buf
        _ttl_buf = &acc
        self.thisptr.decay_ttl_and_cancel(_collect_cancel)
        _ttl_buf = NULL
        return [acc[i] for i in range(acc.size())]


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
        n_trades = self.thisptr.match_market_order(
            is_buy_side, volume, timestamp, taker_is_agent,
            &out_prices[0], &out_volumes[0],
            &out_is_buy[0], &out_is_self[0],
            &out_ids[0], m, &fee_total)

        return (n_trades, fee_total)

    cpdef prune_stale_orders(self, int current_step, int max_age):
        self.thisptr.prune_stale_orders(current_step, max_age)

    cpdef cancel_random_orders_batch(self, sides):
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

    cpdef CythonLOB clone(self):
        cdef CythonLOB other = CythonLOB.__new__(CythonLOB)
        other.thisptr = self.thisptr.clone()
        other._next_id = self._next_id
        return other

    cpdef swap(self, CythonLOB other):
        self.thisptr.swap(other.thisptr[0])

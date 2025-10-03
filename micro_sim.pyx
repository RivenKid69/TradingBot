# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
from numpy cimport ndarray
cimport numpy as cnp
import numpy as np

from core_constants cimport MarketRegime, NORMAL, CHOPPY_FLAT, STRONG_TREND, ILLIQUID

cnp.import_array()

# ---- C++ интерфейсы ----
cdef extern from "OrderBook.h":
    cdef cppclass OrderBook:
        pass

cdef extern from "cpp_microstructure_generator.h":
    cdef enum MicroEventType "MicroEventType":
        pass

    cdef struct MicroEvent:
        MicroEventType type
        bint is_buy
        long long price_ticks
        double size
        unsigned long long order_id
        int timestamp

    cdef struct MicroFeatures:
        long long best_bid
        long long best_ask
        double    mid
        double    spread_ticks
        double    depth_bid_top1
        double    depth_ask_top1
        double    depth_bid_top5
        double    depth_ask_top5
        double    imbalance_top1
        double    imbalance_top5
        int       last_trade_sign
        double    last_trade_size

    cdef cppclass MicrostructureGenerator:
        MicrostructureGenerator() except +
        void set_seed(unsigned long long seed)
        void set_regime(MarketRegime regime)
        void reset(long long mid0_ticks, long long best_bid_ticks, long long best_ask_ticks)
        int step(OrderBook& lob, int timestamp, MicroEvent* out_events, int cap)
        MicroFeatures current_features(const OrderBook& lob) const
        void copy_lambda_hat(double* out6) const

# --- публичные константы ---
cdef public int N_FEATURES = 18
cdef public int LAMBDA_DIM = 6

def lambda_channel_names():
    """
    Порядок λ̂-каналов:
      0: 'LIM_BUY', 1: 'LIM_SELL', 2: 'MKT_BUY',
      3: 'MKT_SELL', 4: 'CAN_BUY', 5: 'CAN_SELL'
    """
    return ("LIM_BUY","LIM_SELL","MKT_BUY","MKT_SELL","CAN_BUY","CAN_SELL")

# --- упаковка признаков ---

cdef inline cnp.ndarray[cnp.float64_t, ndim=1] _features_to_numpy(const MicroFeatures& f):
    """
    ndarray[float64] длиной N_FEATURES:
      0: mid
      1: spread_ticks
      2: depth_bid_top1
      3: depth_ask_top1
      4: depth_bid_top5
      5: depth_ask_top5
      6: imbalance_top1
      7: imbalance_top5
      8: last_trade_sign
      9: last_trade_size
     10: best_bid_ticks
     11: best_ask_ticks
     12..17: λ̂-каналы (см. lambda_channel_names)
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.empty((N_FEATURES,), dtype=np.float64)
    arr[0]  = f.mid
    arr[1]  = f.spread_ticks
    arr[2]  = f.depth_bid_top1
    arr[3]  = f.depth_ask_top1
    arr[4]  = f.depth_bid_top5
    arr[5]  = f.depth_ask_top5
    arr[6]  = f.imbalance_top1
    arr[7]  = f.imbalance_top5
    arr[8]  = <double>f.last_trade_sign
    arr[9]  = f.last_trade_size
    arr[10] = <double>f.best_bid
    arr[11] = <double>f.best_ask
    return arr

# --- основной класс ---

cdef class MicroSim:
    cdef MicrostructureGenerator gen
    cdef size_t _lob_ptr  # OrderBook*

    def __cinit__(self):
        self._lob_ptr = 0

    cpdef attach_lob(self, fast_lob):
        """
        Ожидается объект fast_lob.CythonLOB с методом raw_ptr() -> size_t
        """
        cdef size_t p = <size_t> fast_lob.raw_ptr()
        if p == 0:
            raise ValueError("CythonLOB.raw_ptr() returned null")
        self._lob_ptr = p

    cpdef set_seed(self, unsigned long long seed):
        self.gen.set_seed(seed)

    cpdef set_regime(self, int regime):
        if regime < NORMAL or regime > ILLIQUID:
            raise ValueError("regime out of range (expected 0..3)")
        self.gen.set_regime(<MarketRegime>regime)

    cpdef reset(self, long long mid_ticks=10000, long long best_bid_ticks=0, long long best_ask_ticks=0):
        self.gen.reset(mid_ticks, best_bid_ticks, best_ask_ticks)

    cpdef int step(self, int timestamp):
        if self._lob_ptr == 0:
            raise RuntimeError("MicroSim has no attached LOB (call attach_lob first)")
        cdef OrderBook* ob = <OrderBook*> self._lob_ptr
        return self.gen.step(ob[0], timestamp, <MicroEvent*>NULL, 0)

    cpdef ndarray features(self):
        """
        Вернёт фичи (shape=(N_FEATURES,)). Хвост [12..17] — λ̂-каналы.
        """
        if self._lob_ptr == 0:
            raise RuntimeError("MicroSim has no attached LOB (call attach_lob first)")
        cdef OrderBook* ob = <OrderBook*> self._lob_ptr
        cdef MicroFeatures f = self.gen.current_features(ob[0])
        cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = _features_to_numpy(f)
        cdef double[::1] view = arr
        self.gen.copy_lambda_hat(&view[12])
        return arr


    cpdef ndarray lambda_hat(self):
        """
        Вектор λ̂ (shape=(LAMBDA_DIM,)) в порядке lambda_channel_names()
        """

        cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.zeros((LAMBDA_DIM,), dtype=np.float64)
        cdef double[::1] view = arr
        self.gen.copy_lambda_hat(&view[0])
        return arr

    cpdef dict lambda_hat_dict(self):
        vals = self.lambda_hat()
        names = lambda_channel_names()
        return {names[i]: float(vals[i]) for i in range(LAMBDA_DIM)}

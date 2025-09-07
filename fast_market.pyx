# cython: language_level=3
# distutils: language = c++
from libcpp.vector cimport vector
cdef extern from "Python.h":
    ctypedef void PyObject

cdef extern from "include/latency_queue.h":
    cdef cppclass LatencyQueuePy:
        LatencyQueuePy(size_t delay=0) except +
        void push(PyObject* o)
        void tick()
        vector[PyObject*] pop_ready()
        void clear()
        void set_latency(size_t delay)
        size_t latency() const
        size_t slots() const

from cpython.ref cimport Py_DECREF

cdef class LatencyQueue:
    cdef LatencyQueuePy* _q

    def __cinit__(self, int delay=0):
        self._q = new LatencyQueuePy(<size_t>max(delay, 0))

    def __dealloc__(self):
        if self._q is not None:
            del self._q
            self._q = NULL

    cpdef void push(self, object o):
        self._q.push(<PyObject*>o)

    cpdef list pop_ready(self):
        cdef vector[PyObject*] v = self._q.pop_ready()
        cdef PyObject* p
        out = []
        for i in range(v.size()):
            p = v[i]
            out.append(<object>p)  # превращаем в Python-объект (даёт INCREF)
            Py_DECREF(p)           # балансируем INCREF, сделанный в C++
        return out

    cpdef void tick(self):
        self._q.tick()

    cpdef void clear(self):
        self._q.clear()

    cpdef void set_latency(self, int delay):
        self._q.set_latency(<size_t>max(delay, 0))

    @property
    def latency(self) -> int:
        return <int>self._q.latency()

    def __len__(self):
        # число слотов (latency+1); не суммарный размер очереди
        return <int>self._q.slots()

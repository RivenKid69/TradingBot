# cython: language_level=3
# cython: language=c++
# distutils: language = c++
from libcpp.vector cimport vector
from cython cimport Py_ssize_t
from cpython.object cimport PyObject
from libc.stddef cimport size_t

cdef extern from "include/latency_queue_py.h":
    cdef cppclass LatencyQueuePy:
        LatencyQueuePy() except +
        LatencyQueuePy(size_t delay) except +
        void push(PyObject* o)
        void tick()
        vector[PyObject*] pop_ready()
        void clear()
        void set_latency(size_t delay)
        size_t latency() const
        size_t slots() const

cdef extern from "Python.h":
    PyObject* PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject* list, Py_ssize_t i, PyObject* item)
    void Py_DECREF(PyObject* o)

cdef class LatencyQueue:
    cdef LatencyQueuePy* _q

    def __cinit__(self, int delay=0):
        self._q = new LatencyQueuePy(<size_t>max(delay, 0))

    def __dealloc__(self):
        if self._q is not NULL:
            del self._q
            self._q = NULL

    cpdef void push(self, object o):
        self._q.push(<PyObject*>o)

    cpdef list pop_ready(self):
        cdef vector[PyObject*] v = self._q.pop_ready()
        cdef Py_ssize_t n = <Py_ssize_t>v.size()
        cdef PyObject* list_obj = PyList_New(n)
        cdef PyObject* p
        cdef Py_ssize_t i

        if list_obj == NULL:
            for i in range(n):
                p = v[i]
                Py_DECREF(p)
            raise MemoryError()

        for i in range(n):
            p = v[i]
            PyList_SET_ITEM(list_obj, i, p)  # Steals reference

        return <list>list_obj

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

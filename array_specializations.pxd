# cython: language_level=3
from libc.stddef cimport size_t

cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass ArrayDouble4 "std::array<double, 4>":
        ArrayDouble4() except +
        double& operator[](size_t) except +

    cdef cppclass ArrayDouble6 "std::array<double, 6>":
        ArrayDouble6() except +
        double& operator[](size_t) except +

    cdef cppclass ArrayDouble168 "std::array<double, 168>":
        ArrayDouble168() except +
        double& operator[](size_t) except +

    cdef cppclass ArrayDouble6x6 "std::array<std::array<double, 6>, 6>":
        ArrayDouble6x6() except +
        ArrayDouble6& operator[](size_t) except +

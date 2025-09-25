#pragma once

#include <Python.h>

#include <cstddef>
#include <vector>

#include "latency_queue.h"

// Python-aware wrapper that keeps PyObject* alive while they are queued.
class LatencyQueuePy {
public:
    explicit LatencyQueuePy(std::size_t delay = 0) : m_queue(delay) {}

    ~LatencyQueuePy() {
        clear();
    }

    void push(PyObject* obj) {
        Py_INCREF(obj);
        m_queue.push(obj);
    }

    void tick() {
        auto ready = m_queue.pop_ready();
        for (auto* obj : ready) {
            Py_DECREF(obj);
        }
    }

    std::vector<PyObject*> pop_ready() {
        return m_queue.pop_ready();
    }

    void clear() {
        const std::size_t total_slots = m_queue.slots();
        for (std::size_t i = 0; i < total_slots; ++i) {
            auto ready = m_queue.pop_ready();
            for (auto* obj : ready) {
                Py_DECREF(obj);
            }
        }
        m_queue.clear();
    }

    void set_latency(std::size_t delay) {
        clear();
        m_queue.set_latency(delay);
    }

    std::size_t latency() const {
        return m_queue.latency();
    }

    std::size_t slots() const {
        return m_queue.slots();
    }

private:
    LatencyQueue<PyObject*> m_queue;
};

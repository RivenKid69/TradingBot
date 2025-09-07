// ---------------- include/latency_queue.h --------------------------------
#pragma once
#include <vector>
template<typename T>
class LatencyQueue {
public:
    explicit LatencyQueue(std::size_t delay = 0)
    : m_buf(delay + 1), m_lat(delay), m_head(0) {}

    inline void push(T&& v) {
        m_buf[(m_head + m_lat) % m_buf.size()].push_back(std::move(v));
    }
    template<class... Args> inline void emplace(Args&&... a) {
        m_buf[(m_head + m_lat) % m_buf.size()].emplace_back(std::forward<Args>(a)...);
    }
    std::vector<T> pop_ready() {
        auto& slot = m_buf[m_head];
        std::vector<T> out;
        out.swap(slot);
        m_head = (m_head + 1) % m_buf.size();
        return out;
    }
    void clear() {
        for (auto& v: m_buf) v.clear();
        m_head = 0;
    }
    void set_latency(std::size_t delay) {
        if (delay == m_lat) return;
        m_buf.assign(delay + 1, {});
        m_head = 0;
        m_lat = delay;
    }
private:
    std::vector<std::vector<T>> m_buf;
    std::size_t m_lat;
    std::size_t m_head;
};
// -------------------------------------------------------------------------

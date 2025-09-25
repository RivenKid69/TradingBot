#include "OrderBook.h"

#include <random>
#include <iterator>
#include <tuple>
#include <limits>

/* ------------------------------------------------------------------ */
// ctor
/* ------------------------------------------------------------------ */
OrderBook::OrderBook()
{
#ifdef ORDERBOOK_COW
    _d = std::make_shared<Impl>();
#else
    gen.seed(std::random_device{}());
#endif
}
// PATCH‑ID:P13_OB_ctor

/* ------------------------------------------------------------------ */
// add_limit_order
/* ------------------------------------------------------------------ */
static inline void reserve_if_empty(std::deque<Order>& dq)
{
#if defined(__cpp_lib_containers_reserve) || (__cplusplus >= 202302L)
    if (dq.empty()) dq.reserve(64);
#endif
}

/* --- top‑of‑book mid price (ticks → float) ---------------------- */
static inline double _mid_price(const OrderBook& ob) {
    long long bid = ob.get_best_bid();
    long long ask = ob.get_best_ask();
    if (bid > 0 && ask > 0) return 0.5 * (bid + ask) / PRICE_SCALE;
    if (bid > 0) return bid / static_cast<double>(PRICE_SCALE);
    if (ask > 0) return ask / static_cast<double>(PRICE_SCALE);
    return 1.0; // fallback
}
// PATCH‑ID:P14_OB_mid

int OrderBook::add_limit_order_ex(bool is_buy_side,
                                  long long price,
                                  double volume,
                                  uint64_t order_id,
                                  bool is_agent,
                                  int timestamp,
                                  TimeInForce tif)
{
    // POST_ONLY: отклоняем, если заявка кроссит лучшую встречную
#ifdef ORDERBOOK_COW
    const auto& _bids = d().bids;
    const auto& _asks = d().asks;
#else
    const auto& _bids = bids;
    const auto& _asks = asks;
#endif
    if (tif == TIF_POST_ONLY) {
        if (is_buy_side) {
            if (!_asks.empty() && price >= _asks.begin()->first) {
                return 0; // отклонена
            }
        } else {
            if (!_bids.empty() && price <= _bids.begin()->first) {
                return 0; // отклонена
            }
        }
    }

    // Пока IOC не активируем: ведём себя как GTC (будет реализовано в T2b)
    // if (tif == TIF_IOC) { /* TODO: лимитированный матч без размещения */ }

    // Обычное размещение как в add_limit_order(...)
    Order ord{order_id, volume, is_agent, timestamp};
    ord.ttl_steps = 0; // TTL задаётся позднее через set_order_ttl(...)

    if (is_buy_side) {
    #ifdef ORDERBOOK_COW
        auto &b = d().bids;
    #else
        auto &b = bids;
    #endif
        auto& dq = b[price];
        reserve_if_empty(dq);
        dq.push_back(ord);
        // индекс
    #ifdef ORDERBOOK_COW
        d().idx_map[order_id] = PriceLevelIdx{price, static_cast<uint32_t>(dq.size()-1), true};
    #else
        idx_map[order_id]     = PriceLevelIdx{price, static_cast<uint32_t>(dq.size()-1), true};
    #endif
    } else {
    #ifdef ORDERBOOK_COW
        auto &a = d().asks;
    #else
        auto &a = asks;
    #endif
        auto& dq = a[price];
        reserve_if_empty(dq);
        dq.push_back(ord);
    #ifdef ORDERBOOK_COW
        d().idx_map[order_id] = PriceLevelIdx{price, static_cast<uint32_t>(dq.size()-1), false};
    #else
        idx_map[order_id]     = PriceLevelIdx{price, static_cast<uint32_t>(dq.size()-1), false};
    #endif
    }
    return 1; // принята
}

void OrderBook::add_limit_order(bool is_buy_side,
long long price,
double volume,
long long order_id,
bool is_agent,
int timestamp)
{
    (void)add_limit_order_ex(is_buy_side, price, volume,
                             static_cast<uint64_t>(order_id),
                             is_agent, timestamp, TIF_GTC);
}
/* ------------------------------------------------------------------ */
/*  clone (deep‑copy if !COW, shallow if COW)                         */
/* ------------------------------------------------------------------ */
OrderBook* OrderBook::clone() const
{
#ifdef ORDERBOOK_COW
    OrderBook* ob = new OrderBook();
    ob->_d = _d;               // shared reference
    return ob;
#else
    OrderBook* ob = new OrderBook();
    ob->bids    = bids;
    ob->asks    = asks;
    ob->idx_map = idx_map;
    ob->gen     = gen;
    ob->fee_model = fee_model;
    return ob;
#endif
}

/* ------------------------------------------------------------------ */
/*  swap                                                              */
/* ------------------------------------------------------------------ */
void OrderBook::swap(OrderBook& other) noexcept
{
#ifdef ORDERBOOK_COW
    _d.swap(other._d);
#else
    std::swap(bids,     other.bids);
    std::swap(asks,     other.asks);
    std::swap(idx_map,  other.idx_map);
#endif
}
// PATCH‑ID:P13_OB_clone_swap
/* ------------------------------------------------------------------ */
// remove_order
/* ------------------------------------------------------------------ */
template <typename MapT>
static void erase_and_reindex(
    MapT& book,
    long long price,
    long long oid,
    std::unordered_map<long long, PriceLevelIdx>& idx)
{
    if (oid < 0) {
        return;
    }
    const auto target_id = static_cast<uint64_t>(oid);
    if (target_id > static_cast<uint64_t>(std::numeric_limits<long long>::max())) {
        return;
    }
    auto it = book.find(price);
    if (it == book.end()) {
        return;
    }

    auto& dq = it->second;
    for (std::size_t i = 0; i < dq.size(); ++i) {
        if (dq[i].id == target_id) {
            dq.erase(dq.begin() + static_cast<long long>(i));
            idx.erase(static_cast<long long>(target_id));
            for (std::size_t j = i; j < dq.size(); ++j) {
                idx[dq[j].id].position = static_cast<uint32_t>(j);
            }
            break;
        }
    }
    if (dq.empty()) {
        book.erase(it);
    }
}

void OrderBook::remove_order(bool is_buy_side,
                             long long price,
                             long long order_id)
{
    if (is_buy_side) {
        erase_and_reindex(bids, price, order_id, idx_map);
    } else {
        erase_and_reindex(asks, price, order_id, idx_map);
    }
}

bool OrderBook::set_order_ttl(uint64_t order_id, int ttl_steps) {
#ifdef ORDERBOOK_COW
    auto &b = d().bids;
    auto &a = d().asks;
#else
    auto &b = bids;
    auto &a = asks;
#endif
    for (auto &kv : b) {
        auto &dq = kv.second;
        for (auto &o : dq) {
            if (o.id == order_id) { o.ttl_steps = ttl_steps; return true; }
        }
    }
    for (auto &kv : a) {
        auto &dq = kv.second;
        for (auto &o : dq) {
            if (o.id == order_id) { o.ttl_steps = ttl_steps; return true; }
        }
    }
    return false;
}

int OrderBook::decay_ttl_and_cancel(const std::function<void(const Order&)>& on_cancel) {
#ifdef ORDERBOOK_COW
    auto &b = d().bids;
    auto &a = d().asks;
#else
    auto &b = bids;
    auto &a = asks;
#endif
    using Item = std::tuple<bool,long long,uint64_t>; // (is_buy, price, order_id)
    std::vector<Item> to_remove;
    int cancelled = 0;

    auto scan_side = [&](bool is_buy, auto& book) {
        for (auto &kv : book) {
            long long price = kv.first;
            auto &dq = kv.second;
            for (auto &o : dq) {
                if (o.ttl_steps > 0) {
                    --o.ttl_steps;
                    if (o.ttl_steps == 0) {
                        if (on_cancel) on_cancel(o);   // вызвать до удаления
                        to_remove.emplace_back(is_buy, price, o.id);
                        ++cancelled;
                    }
                }
            }
        }
    };

    scan_side(true,  b);
    scan_side(false, a);

    for (const auto &it : to_remove) {
        bool is_buy; long long price; uint64_t oid;
        std::tie(is_buy, price, oid) = it;
        remove_order(is_buy, price, static_cast<long long>(oid));
    }
    return cancelled;
}


/* ------------------------------------------------------------------ */
// internal matcher
/* ------------------------------------------------------------------ */
/* =================================================================
*  Commission & slippage (Phase 14)
* ================================================================= */
std::pair<int,int> OrderBook::_match_logic(
bool is_buy_side,
double volume,
int timestamp,
bool taker_is_agent,
const std::function<void(long long,double,const Order&)>& cb)
{
    double remain = volume;
    const double EPS = 1e-9;
    int trades = 0, full_exec = 0;

    auto process = [&](auto& book, bool taker_is_buy) {
        while (remain > EPS && !book.empty()) {
            auto lvl_it = book.begin();
            long long price_ticks = lvl_it->first;
            auto& dq = lvl_it->second;

            while (!dq.empty() && remain > EPS) {
                Order& maker = dq.front();

                // self‑trade prevention
                if (maker.is_agent && taker_is_agent) {
                    dq.pop_front();
                    idx_map.erase(maker.id);
                    for (std::size_t j = 0; j < dq.size(); ++j)
                        idx_map[dq[j].id].position = static_cast<uint32_t>(j);
                    continue;
                }

                double fill = std::min(remain, maker.volume);
                double trade_px = static_cast<double>(price_ticks) / PRICE_SCALE;

                /* ---------- slippage (linear impact, signed) ---------- */
                trade_px *= 1.0 + SLIP_K * (taker_is_buy ? +1.0 : -1.0);

                /* ---------- fee application --------------------------- */
                const double fee_maker = maker.is_agent ? MAKER_FEE : 0.0;
                const double fee_taker = taker_is_agent ? TAKER_FEE : 0.0;

                cb(price_ticks, fill, maker);     // существующий колбэк для сделки

                /* записываем стоимость комиссий как отрицательный cash через колбэк.
                   Для публичной стороны комиссии игнорируются. */
                if (taker_is_agent) {
                    cb(static_cast<long long>(-1),   // специальный маркер комиссии для агента‑taker
                       fill * trade_px * fee_taker,
                       maker);
                }
                if (maker.is_agent) {
                    cb(static_cast<long long>(-2),   // специальный маркер комиссии для агента‑maker
                       fill * trade_px * fee_maker,
                       maker);
                }

                trades++; maker.volume -= fill; remain -= fill;

                if (maker.volume < EPS) {
                    long long rid = maker.id;
                    dq.pop_front();
                    full_exec++;
                    idx_map.erase(rid);
                    for (std::size_t j = 0; j < dq.size(); ++j)
                        idx_map[dq[j].id].position = static_cast<uint32_t>(j);
                }
            }
            if (dq.empty()) book.erase(lvl_it);
        }
    };

    if (is_buy_side)
        process(asks, true);   // BUY market hits asks
    else
        process(bids, false);  // SELL market hits bids

    (void)timestamp;
    return {trades, full_exec};
}

/* ------------------------------------------------------------------ */
// match_market_order – dense-array variant
/* ------------------------------------------------------------------ */ 
int OrderBook::match_market_order(
    bool is_buy,
    double volume,
    int ts,
    bool taker_is_agent,
    double* out_prices,
    double* out_volumes,
    int*    out_is_buy,
    int*    out_is_self,
    long long* out_ids,
    int max_len,
    double* out_fee_total)
{
    int t_written = 0;
    double fee_total = 0.0;

    auto handler = [&](long long px, double fill, const Order& mk) {
        // fee-маркеры: px == -1 (taker), px == -2 (maker); fill содержит денежную величину комиссии (>0)
        if (px < 0) { fee_total += fill; return; }

        if (t_written < max_len) {
            const int idx = t_written;
            out_prices[idx]  = static_cast<double>(px) / PRICE_SCALE;
            out_volumes[idx] = fill;
            out_is_buy[idx]  = is_buy ? 1 : 0;
            out_is_self[idx] = mk.is_agent ? 1 : 0;
            out_ids[idx]     = static_cast<long long>(mk.id);
            ++t_written;
        }
    };

    _match_logic(is_buy, volume, ts, taker_is_agent, handler);
    if (out_fee_total) *out_fee_total = fee_total;
    return t_written;
}

int OrderBook::match_limit_order(
    bool is_buy,
    double volume,
    long long limit_px,
    int ts,
    bool taker_is_agent,
    double* out_prices,
    double* out_volumes,
    int*    out_is_buy,
    int*    out_is_self,
    long long* out_ids,
    int max_len,
    double* out_fee_total)
{
    int t_written = 0;
    double fee_total = 0.0;
    if (volume <= 0.0 || max_len <= 0) {
        if (out_fee_total) *out_fee_total = 0.0;
        return 0;
    }

#ifdef ORDERBOOK_COW
    auto &B = d().bids;
    auto &A = d().asks;
#else
    auto &B = bids;
    auto &A = asks;
#endif

    // Быстрый отказ: лучшая встречная уже вне лимита
    if (is_buy) {
        if (A.empty() || A.begin()->first > limit_px) {
            if (out_fee_total) *out_fee_total = 0.0;
            return 0;
        }
    } else {
        if (B.empty() || B.begin()->first < limit_px) {
            if (out_fee_total) *out_fee_total = 0.0;
            return 0;
        }
    }

    // Основной цикл: как market, но с проверкой limit_px на каждом уровне
    while (volume > 1e-12 && t_written < max_len) {
        if (is_buy) {
            if (A.empty()) break;
            auto it = A.begin();
            const long long px = it->first;
            if (px > limit_px) break; // дороже лимита — стоп
            auto &dq = it->second;
            while (volume > 1e-12 && !dq.empty() && t_written < max_len) {
                Order &maker = dq.front();
                const double trade_vol = (maker.volume < volume) ? maker.volume : volume;
                maker.volume -= trade_vol;
                volume       -= trade_vol;

                // Запись трейда
                out_prices[t_written]   = static_cast<double>(px);
                out_volumes[t_written]  = trade_vol;
                out_is_buy[t_written]   = 1;  // TAKER buy
                out_is_self[t_written]  = maker.is_agent ? 1 : 0;
                out_ids[t_written]      = static_cast<long long>(maker.id);
                // Комиссии (только суммарно)
                fee_total += (taker_is_agent ? fee_model.taker_fee : 0.0) * trade_vol;
                fee_total += (maker.is_agent ? fee_model.maker_fee : 0.0) * trade_vol;

                ++t_written;
                if (maker.volume <= 1e-12) {
                    dq.pop_front();
#ifdef ORDERBOOK_COW
                    d().idx_map.erase(maker.id);
#else
                    idx_map.erase(maker.id);
#endif
                }
            }
            if (dq.empty()) A.erase(it);
        } else {
            if (B.empty()) break;
            auto it = B.begin();
            const long long px = it->first;
            if (px < limit_px) break; // дешевле лимита — стоп (для продажи)
            auto &dq = it->second;
            while (volume > 1e-12 && !dq.empty() && t_written < max_len) {
                Order &maker = dq.front();
                const double trade_vol = (maker.volume < volume) ? maker.volume : volume;
                maker.volume -= trade_vol;
                volume       -= trade_vol;

                out_prices[t_written]   = static_cast<double>(px);
                out_volumes[t_written]  = trade_vol;
                out_is_buy[t_written]   = 0;  // TAKER sell
                out_is_self[t_written]  = maker.is_agent ? 1 : 0;
                out_ids[t_written]      = static_cast<long long>(maker.id);

                fee_total += (taker_is_agent ? fee_model.taker_fee : 0.0) * trade_vol;
                fee_total += (maker.is_agent ? fee_model.maker_fee : 0.0) * trade_vol;

                ++t_written;
                if (maker.volume <= 1e-12) {
                    dq.pop_front();
#ifdef ORDERBOOK_COW
                    d().idx_map.erase(maker.id);
#else
                    idx_map.erase(maker.id);
#endif
                }
            }
            if (dq.empty()) B.erase(it);
        }
    }

    if (out_fee_total) *out_fee_total = fee_total;
    return t_written;
}

/* ------------------------------------------------------------------ */
// best bid / ask
/* ------------------------------------------------------------------ */
long long OrderBook::get_best_bid() const { return bids.empty() ? -1 : bids.begin()->first; }
long long OrderBook::get_best_ask() const { return asks.empty() ? -1 : asks.begin()->first; }

/* ------------------------------------------------------------------ */
// prune stale
/* ------------------------------------------------------------------ */
void OrderBook::prune_stale_orders(int current_step, int max_age)
{
    auto prune = [&](auto& book) {
        for (auto it = book.begin(); it != book.end();) {
            auto& dq = it->second;
            dq.erase(std::remove_if(dq.begin(), dq.end(),
                                    [&](const Order& o){ return current_step - o.timestamp >= max_age; }),
                     dq.end());
            if (dq.empty()) it = book.erase(it);
            else ++it;
        }
    };
    prune(bids);
    prune(asks);
}

/* ------------------------------------------------------------------ */
// contains_order
/* ------------------------------------------------------------------ */
bool OrderBook::contains_order(long long oid) const
{
    return idx_map.find(oid) != idx_map.end();
}

/* ------------------------------------------------------------------ */
// cancel_random_public_orders
/* ------------------------------------------------------------------ */
void OrderBook::cancel_random_public_orders(bool is_buy_side, int n)
{
    if (n <= 0) return;
    std::vector<std::pair<long long,long long>> sample;
    sample.reserve(n);
    long long seen = 0;

    auto sample_book = [&](auto& book) {
        for (auto& [px, dq] : book) {
            for (auto& ord : dq) {
                if (ord.is_agent) continue;
                if (sample.size() < static_cast<std::size_t>(n)) {
                    sample.emplace_back(px, ord.id);
                } else {
                    std::uniform_int_distribution<long long> d(0, ++seen);
                    long long j = d(gen);
                    if (j < n) sample[j] = {px, ord.id};
                }
                ++seen;
            }
        }
    };

    if (is_buy_side) {
        sample_book(bids);
    } else {
        sample_book(asks);
    }

    for (auto& p : sample) {
        remove_order(is_buy_side, p.first, p.second);
    }
}

/* ------------------------------------------------------------------ */
// volume for top levels
/* ------------------------------------------------------------------ */
double OrderBook::get_volume_for_top_levels(bool is_buy_side, int levels) const
{
    double vol = 0.0;
    int counted = 0;
    if (is_buy_side) {
        for (const auto& [px, dq] : bids) {
            if (counted++ >= levels) break;
            for (const auto& o : dq) vol += o.volume;
        }
    } else {
        for (const auto& [px, dq] : asks) {
            if (counted++ >= levels) break;
            for (const auto& o : dq) vol += o.volume;
        }
    }
    return vol;
}

/* ------------------------------------------------------------------ */
// get_queue_position
/* ------------------------------------------------------------------ */
uint32_t OrderBook::get_queue_position(long long oid) const
{
    auto it = idx_map.find(oid);
    return (it == idx_map.end()) ? std::numeric_limits<uint32_t>::max()
                                 : it->second.position;
}

void OrderBook::set_seed(uint64_t seed) {
#ifdef ORDERBOOK_COW
    d().gen.seed(static_cast<uint32_t>(seed));
#else
    gen.seed(static_cast<uint32_t>(seed));
#endif
}

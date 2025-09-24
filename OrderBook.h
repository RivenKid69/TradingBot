#ifndef ORDER_BOOK_H
#define ORDER_BOOK_H

#include <map>
#include <deque>
#include <unordered_map>
#include <vector>
#include <random>
#include <functional>
#include <cstdint>
#include <limits>
#include <algorithm>
#include "core_constants.h"
#include "fee_config.h" 
 
// ---- Time In Force for limit orders ----
enum TimeInForce : int {
    TIF_GTC = 0,      // Good-Till-Cancel
    TIF_IOC = 1,      // Immediate-Or-Cancel (задел; станет активным в T2b)
    TIF_POST_ONLY = 2 // Никогда не кроссить, иначе отклонить
};


// -----------------------------------------------------------------------------
// 64‑битный генератор ID с rollover: 48‑битный счётчик + 16‑битный epoch
// -----------------------------------------------------------------------------
// При каждой выдаче ID увеличивается счётчик. Когда он достигает 2^48‑1,
// счётчик обнуляется, а epoch увеличивается. ID кодируется как
// (epoch << 48) | seq.
static uint64_t _next_id   = 0;
static uint16_t _epoch     = 0;
inline uint64_t make_order_id() {
    // 48‑битный счётчик
    if (_next_id == 0xFFFFFFFFFFFFULL) {
        ++_epoch;
        _next_id = 0;
    }
    return (static_cast<uint64_t>(_epoch) << 48) | _next_id++;
}
// PATCH‑ID:P12_P11_id_seed_spec



/* -------------------------------------------------------------------------
 * Basic order record
 * ------------------------------------------------------------------------- */
struct Order {
    uint64_t id;       // 64-битный идентификатор
    double   volume;
    bool     is_agent;
    int      timestamp;
    int      ttl_steps;   // TTL в тиках; 0 или <0 — не истекает автоматически
};

/* -------------------------------------------------------------------------
 * Fast queue-index lookup (O(1))
 * ------------------------------------------------------------------------- */
struct PriceLevelIdx {
    long long price_ticks;   // уровень цены (в тиках)
    uint32_t  position;      // позиция в очереди (0-based)
    bool      is_buy;        // true, если ордер в книге бидов
};

using BidsMap = std::map<long long, std::deque<Order>, std::greater<long long>>;
using AsksMap = std::map<long long, std::deque<Order>>;

// -------------------------------------------------------------------------
// Модель комиссий и проскальзывания
// -------------------------------------------------------------------------
struct FeeModel {
    double maker_fee {0.0};
    double taker_fee {0.0};
    double slip_k   {0.0};
};

// OrderBook – value‑semantics, non‑copyable

/* -------------------------------------------------------------------------
 * Copy-on-Write toggle (Phase 13).
 * Build with -DORDERBOOK_COW=ON to enable shared_ptr-based cloning.
 * ------------------------------------------------------------------------- */
#ifdef ORDERBOOK_COW
#  include <memory>
#endif
// PATCH‑ID:P13_OB_toggle

// ---------------------------------------------------------------------
class OrderBook {
#ifdef ORDERBOOK_COW
struct Impl {
    BidsMap bids;
    AsksMap asks;
    std::unordered_map<long long, PriceLevelIdx> idx_map;
    std::mt19937 gen;
    Impl() { gen.seed(std::random_device{}()); }
};
#endif
// PATCH‑ID:P13_OB_impl
public:
OrderBook();
~OrderBook() = default;
OrderBook(const OrderBook&) = delete;
OrderBook& operator=(const OrderBook&) = delete;
void set_seed(uint64_t seed);

// Расширенный вариант с управлением TIF; возвращает 1 если принята, 0 если отклонена
int add_limit_order_ex(bool is_buy_side,
                       long long price_ticks,
                       double volume,
                       uint64_t order_id,
                       bool is_agent,
                       int timestamp,
                       TimeInForce tif);

void add_limit_order(bool is_buy_side,
                     long long price_ticks,
                     double volume,
                     long long order_id,
                     bool is_agent,
                     int timestamp);

void remove_order(bool is_buy_side,
                  long long price_ticks,
                  long long order_id);

int match_market_order(bool is_buy_side,
                       double volume,
                       int timestamp,
                       bool taker_is_agent,
                       double* out_prices,
                       double* out_volumes,
                       int*    out_is_buy,
                       int*    out_is_self,
                       long long* out_ids,
                       int max_len,
                       double* out_fee_total);

// unified dense version – returns number of trades; also returns total agent fee (cash, >0)
int match_limit_order(bool is_buy_side,
                      double volume,
                      long long limit_price_ticks,
                      int timestamp,
                      bool taker_is_agent,
                      double* out_prices,
                      double* out_volumes,
                      int*    out_is_buy,
                      int*    out_is_self,
                      long long* out_ids,
                      int max_len,
                      double* out_fee_total);


/* helpers ----------------------------------------------------------- */
long long get_best_bid() const;
long long get_best_ask() const;

void prune_stale_orders(int current_step, int max_age);

const std::map<long long, std::deque<Order>, std::greater<long long>>&
get_bids() const { return bids; }
const std::map<long long, std::deque<Order>>&
get_asks() const { return asks; }

bool contains_order(long long order_id) const;
void cancel_random_public_orders(bool is_buy_side, int num_to_cancel);
double get_volume_for_top_levels(bool is_buy_side, int levels) const;

/* O(1) queue‑index lookup */
uint32_t get_queue_position(long long order_id) const;

// Установить TTL конкретному ордеру (сканирует обе стороны книги), true если найден
bool set_order_ttl(uint64_t order_id, int ttl_steps);
// Декремент TTL у всех ордеров; нулевые — удалить. Возвращает число отмен.
int decay_ttl_and_cancel(const std::function<void(const Order&)>& on_cancel = {});

// Copy-on-Write helpers (глубокая копия если !ORDERBOOK_COW)
OrderBook* clone() const;
void swap(OrderBook& other) noexcept;

/* -------------------------------------------------------------------------
 * Модель комиссий и проскальзывания для книги
 * ------------------------------------------------------------------------- */
FeeModel fee_model;
// установить модель комиссий (копируется)
void set_fee_model(const FeeModel& fm) { fee_model = fm; }

private:
#ifdef ORDERBOOK_COW
    std::shared_ptr<Impl> _d;
    Impl& d() { return *_d; }
    const Impl& d() const { return *_d; }
#else
    BidsMap bids;
    AsksMap asks;
    std::unordered_map<long long, PriceLevelIdx> idx_map;
    std::mt19937 gen;
#endif

    std::pair<int,int> _match_logic(
        bool  is_buy_side,
        double volume,
        int    timestamp,
        bool   taker_is_agent,
        const std::function<void(long long,double,const Order&)>& trade_handler);
};

#endif /* ORDER_BOOK_H */

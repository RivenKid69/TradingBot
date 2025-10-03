#pragma once
// cpp_microstructure_generator.h
// Полнофункциональный генератор микроструктуры (многовидовой Hawkes)
// для синтеза потока событий LIMIT/MARKET/CANCEL (buy/sell) и их применения к LOB.
//
// Совместимость:
//  - Ожидает C++ LOB с API из OrderBook.h (add_limit_order/remove_order/match_market_order/...).
//  - Идентификаторы ордеров задаются генератором (64-bit), что согласуется с fast_lob.pyx.
//  - Цена задаётся в "тиках" (long long). Масштаб см. core_constants.h: PRICE_SCALE.
//
// Обзор:
//  - 6 каналов событий: {LIMIT, MARKET, CANCEL} × {BUY, SELL}.
//  - Интенсивности λ_k(t) моделируются многовидовым Hawkes с экспоненциальными ядрами:
//        λ_k(t) = μ_k + Σ_j α_{k,j} Σ_{t_i^j < t} exp(-β_{k,j} (t - t_i^j))
//    В дискретном времени (шаг=1) используется коинтегрированное рекуррентное обновление.
//  - Размеры заявок (LIMIT/MARKET) — логнормальные хвосты (ограниченные min/max).
//  - Размещение лимиток по уровням — смесь "у лучшей цены" и геометрического хвоста по уровню ΔL>=0.
//  - CANCEL выбирает случайный активный ордер на стороне и уровне с вероятностью ∝ глубине.
//  - Редкие "черные лебеди" (крэш/мания) и краткосрочные flash-шоки модифицируют цену/интенсивности.
//
// Основные методы:
//  - set_* : настройка параметров (Hawkes, размеры, профиль размещения, cancel-rate, шоки).
//  - reset(...) : сброс состояния, инициализация интенсивностей.
//  - step(lob, ts, out_events, cap) : сгенерировать и применить события в текущем шаге;
//      вернуть количество событий, заполнить out_events (тип/сторона/цена/размер/id/ts).
//  - current_features(lob) : набор признаков (spread, mid, depth, imbalance, λ̂, last trade).
//
// Примечание: реализация в cpp_microstructure_generator.cpp
//

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <limits>
#include <utility>
#include <cmath>

#include "core_constants.h"   // PRICE_SCALE, MarketRegime

// Вперёд-объявление LOB из вашего OrderBook.h
struct FeeModel;
class OrderBook;

// ------------------------------- Каналы Hawkes -------------------------------

enum class MicroEventType : int { LIMIT = 0, MARKET = 1, CANCEL = 2 };

constexpr MicroEventType operator|(MicroEventType lhs, MicroEventType rhs) {
    return static_cast<MicroEventType>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

constexpr MicroEventType operator<<(MicroEventType lhs, int rhs) {
    return static_cast<MicroEventType>(static_cast<int>(lhs) << rhs);
}

constexpr MicroEventType operator*(MicroEventType lhs, MicroEventType rhs) {
    return static_cast<MicroEventType>(static_cast<int>(lhs) * static_cast<int>(rhs));
}

constexpr MicroEventType operator*(MicroEventType lhs, int rhs) {
    return static_cast<MicroEventType>(static_cast<int>(lhs) * rhs);
}

constexpr MicroEventType operator*(int lhs, MicroEventType rhs) {
    return static_cast<MicroEventType>(lhs * static_cast<int>(rhs));
}

static constexpr int CH_LIM_BUY  = 0;
static constexpr int CH_LIM_SELL = 1;
static constexpr int CH_MKT_BUY  = 2;
static constexpr int CH_MKT_SELL = 3;
static constexpr int CH_CAN_BUY  = 4;
static constexpr int CH_CAN_SELL = 5;
static constexpr int CH_K        = 6;

// ------------------------------- Структуры параметров -------------------------------

struct HawkesParams {
    // Базовые интенсивности (μ_k ≥ 0)
    std::array<double, CH_K> mu{};

    // Воздействия α_{k,j} (неотрицательные; диагональные элементы допустимы)
    std::array<std::array<double, CH_K>, CH_K> alpha{};

    // Затухания β_{k,j} > 0 (экспоненциальные ядра)
    std::array<std::array<double, CH_K>, CH_K> beta{};
};

struct SizeDist {
    // Логнормальные размеры: exp(N(m, s^2)), затем клип в [min_size, max_size]
    double lognorm_m = std::log(1.0);
    double lognorm_s = 1.0;
    double min_size  = 0.001;
    double max_size  = 1000.0;
};

struct PlacementProfile {
    // Вероятность разместить лимитку на лучшем уровне (ΔL=0)
    double at_best_prob = 0.65;
    // Геометрический хвост по уровням: P(ΔL = l | l>0) ∝ (1-p)^{l-1} p
    double geometric_p  = 0.35;
    // Ограничение на максимальный уровень (ΔL)
    int max_levels      = 50;
};

struct ShockParams {
    // Flash-шоки краткосрочной волатильности/дрейфа (перешивают интенсивности и цену на один шаг)
    bool   enabled         = false;
    double prob_per_step   = 0.02;     // вероятность шока на шаг
    double intensity_scale = 2.0;      // множитель к λ на один шаг
    double price_bps_mu    = 0.0;      // средний сдвиг цены в б.п. (100 б.п. = 1%)
    double price_bps_std   = 25.0;     // σ сдвига цены в б.п.
};

struct BlackSwanParams {
    // Редкие крупные сдвиги: крэш (−) и мания (+). На несколько шагов влияет цена и λ.
    bool   enabled         = true;
    double prob_per_step   = 0.001;    // суммарная вероятность
    double crash_min       = 0.20;     // амплитуда падения [min,max] как доля цены (0.20 → −20%)
    double crash_max       = 0.35;
    double mania_min       = 0.20;     // амплитуда роста [min,max] (0.20 → +20%)
    double mania_max       = 0.40;
    int    cooldown_steps  = 240;      // после срабатывания — пауза без повторов
    double intensity_scale = 3.0;      // множитель к λ в периоде события
    int    duration_steps  = 4;        // длительность влияния
};

// ------------------------------- Событие потока -------------------------------

struct MicroEvent {
    MicroEventType type = MicroEventType::LIMIT;
    bool is_buy = true;
    long long price_ticks = 0;
    double size = 0.0;
    std::uint64_t order_id = 0; // для LIMIT/CANCEL
    int timestamp = 0;
};

// ------------------------------- Признаки (features) -------------------------------

struct MicroFeatures {
    long long best_bid = 0;
    long long best_ask = 0;
    double    mid      = std::numeric_limits<double>::quiet_NaN();
    double    spread_ticks = std::numeric_limits<double>::quiet_NaN();

    // Глубина на top1 и суммарная на top5 (в "контрактах"/единицах объёма)
    double depth_bid_top1 = 0.0;
    double depth_ask_top1 = 0.0;
    double depth_bid_top5 = 0.0;
    double depth_ask_top5 = 0.0;

    // Имбаланс на top1/top5: (bid - ask) / (bid + ask) ∈ [-1,1]
    double imbalance_top1 = 0.0;
    double imbalance_top5 = 0.0;

    // Оценка текущих интенсивностей λ̂ каналов (после всех модификаторов)
    std::array<double, CH_K> lambda_hat{};

    // Последняя сделка (если была в этом шаге)
    int    last_trade_sign = 0;   // +1 buy, −1 sell, 0 нет
    double last_trade_size = 0.0;
};

// ------------------------------- Основной класс -------------------------------

class MicrostructureGenerator {
public:
    MicrostructureGenerator();

    // Настройка случайности
    void set_seed(std::uint64_t seed);

    // Базовые параметры Hawkes
    void set_hawkes_params(const HawkesParams& hp);

    // Размеры заявок
    void set_size_models(const SizeDist& limit_sz, const SizeDist& market_sz);

    // Профиль размещения лимиток
    void set_placement_profile(const PlacementProfile& pp);

    // Базовая интенсивность отмен (масштабирует каналы CANCEL)
    void set_cancel_rate(double base_cancel_rate);

    // Шоки
    void set_flash_shocks(const ShockParams& sp);
    void set_black_swan(const BlackSwanParams& bp);

    // Регимы рынка (мягкие множители к μ/α на этапах)
    void set_regime(MarketRegime regime);
    MarketRegime regime() const { return m_regime; }

    // Сброс состояния: очистить историю, активные ордера; инициализировать интенсивности на μ.
    // mid0_ticks — стартовая "средняя" цена (в тиках). bid/ask могут быть = mid0 ± 1 по умолчанию.
    void reset(long long mid0_ticks, long long best_bid_ticks = 0, long long best_ask_ticks = 0);

    // Один дискретный шаг времени:
    // - генерирует случайное число событий каждого вида,
    // - применяет их к LOB (через OrderBook),
    // - заполняет out_events (не обязательно все, если cap недостаточен),
    // - возвращает количество СФОРМИРОВАННЫХ событий (может быть > cap).
    int step(OrderBook& lob, int timestamp,
             MicroEvent* out_events, int cap);

    // Текущие признаки микроструктуры (после step или после reset).
    MicroFeatures current_features(const OrderBook& lob) const;

    // Последний сгенерированный ID ордера (монотонно увеличивается)
    std::uint64_t last_order_id() const { return m_next_order_id ? (m_next_order_id - 1) : 0; }

    // Доступ к λ̂ (после применения шоков/режимов)
    const std::array<double, CH_K>& lambda_hat() const { return m_lambda_hat; }
    void copy_lambda_hat(double* out) const {
        if (!out) return;
        for (int k = 0; k < CH_K; ++k) {
            out[k] = m_lambda_hat[k];
        }
    }

private:
    // ----------------- внутреннее состояние -----------------
    HawkesParams  m_hp{};
    SizeDist      m_sz_limit{};
    SizeDist      m_sz_market{};
    PlacementProfile m_place{};
    ShockParams   m_sp{};
    BlackSwanParams m_bp{};
    MarketRegime  m_regime = MarketRegime::NORMAL;

    // λ̂ текущие (после дискретного обновления ядра). Обновляются в step().
    std::array<double, CH_K> m_lambda_hat{};

    // Интегральная «память» Hawkes (рекуррентная часть по эксп. ядрам)
    // Для дискретизации: e^{−βΔt} фактор по каждому (k,j).
    std::array<std::array<double, CH_K>, CH_K> m_state_contrib{}; // S_{k,j} на тек. шаге

    // Случайность
    std::mt19937_64 m_rng;
    std::uniform_real_distribution<double> m_unif{0.0, 1.0};
    std::normal_distribution<double> m_stdN{0.0, 1.0};

    // Управление ID и пулами ордеров (для CANCEL нужны id)
    std::uint64_t m_next_order_id = 1;

    // По каждой стороне хранить множество активных ID и группировку по цене
    struct SideBook {
        std::unordered_set<std::uint64_t> ids;
        std::unordered_map<long long, std::vector<std::uint64_t>> by_price;
    };
    SideBook m_buy_side, m_sell_side;

    // Черный лебедь: охлаждение и остаточная длительность эффекта
    int m_bw_cooldown = 0;
    int m_bw_left     = 0;
    int m_last_trade_sign = 0;
    double m_last_trade_size = 0.0;

    // ----------------- внутренние методы -----------------
    // Обновить λ̂ на текущем шаге (с учётом режимов/шоков/остаточного влияния).
    void _update_lambdas();

    // Сгенерировать количество событий в канале k по Пуассону с интенсивностью λ̂_k
    int _poisson_count(double lambda);

    // Выбор уровня ΔL для лимитки
    int _sample_limit_level(bool is_buy, long long best_bid, long long best_ask);

    // Сэмпл размера по модели
    double _sample_size(const SizeDist& sd);

    // Применить LIMIT к LOB, зарегистрировать ID/уровень
    void _emit_limit(OrderBook& lob, bool is_buy, long long price_ticks, double size, int ts,
                     MicroEvent* out, int& out_n, int out_cap);

    // Применить MARKET к LOB, обновить last_trade_* (по исполненным сделкам)
    void _emit_market(OrderBook& lob, bool is_buy, double size, int ts,
                      MicroEvent* out, int& out_n, int out_cap);

    // Применить CANCEL к LOB (выбираем случайный активный ордер на стороне)
    void _emit_cancel(OrderBook& lob, bool is_buy, int ts,
                      MicroEvent* out, int& out_n, int out_cap);

    // Учёт ID в пулах
    void _track_new_order(bool is_buy, std::uint64_t oid, long long price);
    void _untrack_order(bool is_buy, std::uint64_t oid, long long price);

    // Подготовить признаки на основе LOB
    MicroFeatures _features_from_lob(const OrderBook& lob) const;
};

using CppMicrostructureGenerator = MicrostructureGenerator;

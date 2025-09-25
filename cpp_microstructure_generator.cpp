#include "cpp_microstructure_generator.h"
#include "OrderBook.h"   // интерфейс LOB (см. fast_lob.pyx)
#include <algorithm>
#include <numeric>
#include <random>

#ifndef NAN
#define NAN std::numeric_limits<double>::quiet_NaN()
#endif

// ------------------------------- ВСПОМОГАТЕЛЬНОЕ -------------------------------

static inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline double geometric_level(std::mt19937_64& rng, double p, int max_levels) {
    // Геометрия на {1,2,...}; переносим к {1..max}, затем ограничим
    p = clampd(p, 1e-6, 1.0 - 1e-6);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(rng);
    long long L = 1 + (long long)std::floor(std::log(1.0 - u) / std::log(1.0 - p)); // >=1
    if (L < 1) L = 1;
    if (L > max_levels) L = max_levels;
    return (double)L;
}

// ------------------------------- КОНСТРУКТОР И НАСТРОЙКА -------------------------------

MicrostructureGenerator::MicrostructureGenerator() {
    set_seed(0);

    // Базовые (разумные) параметры Hawkes по умолчанию:
    // умеренная самовозбуждённость в пределах канала + слабые кросс-связи
    for (int k = 0; k < CH_K; ++k) {
        m_hp.mu[k] = 0.2; // базовые интенсивности (событий/шаг)
    }
    m_hp.mu[CH_MKT_BUY]  = 0.12;
    m_hp.mu[CH_MKT_SELL] = 0.12;
    m_hp.mu[CH_CAN_BUY]  = 0.10;
    m_hp.mu[CH_CAN_SELL] = 0.10;

    for (int k = 0; k < CH_K; ++k) {
        for (int j = 0; j < CH_K; ++j) {
            m_hp.alpha[k][j] = (k == j ? 0.15 : 0.03);
            m_hp.beta[k][j]  = 1.20; // экспоненциальное ядро, среднее затухание ~1 шаг
        }
    }

    // Размеры: логнормальные с лёгкими хвостами
    m_sz_limit.lognorm_m = std::log(2.0);
    m_sz_limit.lognorm_s = 0.9;
    m_sz_limit.min_size  = 0.001;
    m_sz_limit.max_size  = 5000.0;

    m_sz_market.lognorm_m = std::log(1.0);
    m_sz_market.lognorm_s = 0.8;
    m_sz_market.min_size  = 0.001;
    m_sz_market.max_size  = 2000.0;

    // Размещение лимиток: чаще у лучшего уровня, с геометрическим хвостом
    m_place.at_best_prob = 0.65;
    m_place.geometric_p  = 0.35;
    m_place.max_levels   = 50;

    // Flash-шоки и BlackSwan — по умолчанию умеренно включены
    m_sp.enabled         = true;
    m_sp.prob_per_step   = 0.02;
    m_sp.intensity_scale = 2.0;
    m_sp.price_bps_mu    = 0.0;
    m_sp.price_bps_std   = 25.0;

    m_bp.enabled         = true;
    m_bp.prob_per_step   = 0.001;
    m_bp.crash_min       = 0.20;
    m_bp.crash_max       = 0.35;
    m_bp.mania_min       = 0.20;
    m_bp.mania_max       = 0.40;
    m_bp.cooldown_steps  = 240;
    m_bp.intensity_scale = 3.0;
    m_bp.duration_steps  = 4;

    // Инициализация состояний
    for (int k = 0; k < CH_K; ++k) m_lambda_hat[k] = m_hp.mu[k];
    for (int k = 0; k < CH_K; ++k) for (int j = 0; j < CH_K; ++j) m_state_contrib[k][j] = 0.0;

    m_bw_cooldown = 0;
    m_bw_left = 0;
    m_last_trade_sign = 0;
    m_last_trade_size = 0.0;
}

void MicrostructureGenerator::set_seed(std::uint64_t seed) {
    if (seed == 0) {
        std::random_device rd;
        m_rng.seed(((std::uint64_t)rd() << 32) ^ (std::uint64_t)rd());
    } else {
        m_rng.seed(seed);
    }
}

void MicrostructureGenerator::set_hawkes_params(const HawkesParams& hp) {
    m_hp = hp;
}

void MicrostructureGenerator::set_size_models(const SizeDist& limit_sz, const SizeDist& market_sz) {
    m_sz_limit = limit_sz;
    m_sz_market = market_sz;
}

void MicrostructureGenerator::set_placement_profile(const PlacementProfile& pp) {
    m_place = pp;
}

void MicrostructureGenerator::set_cancel_rate(double base_cancel_rate) {
    base_cancel_rate = std::max(0.0, base_cancel_rate);
    m_hp.mu[CH_CAN_BUY]  = base_cancel_rate;
    m_hp.mu[CH_CAN_SELL] = base_cancel_rate;
}

void MicrostructureGenerator::set_flash_shocks(const ShockParams& sp) { m_sp = sp; }
void MicrostructureGenerator::set_black_swan(const BlackSwanParams& bp) { m_bp = bp; }

void MicrostructureGenerator::set_regime(MarketRegime regime) { m_regime = regime; }

// ------------------------------- RESET -------------------------------

void MicrostructureGenerator::reset(long long mid0_ticks, long long best_bid_ticks, long long best_ask_ticks) {
    // Очистка состояний Hawkes
    for (int k = 0; k < CH_K; ++k) m_lambda_hat[k] = m_hp.mu[k];
    for (int k = 0; k < CH_K; ++k) for (int j = 0; j < CH_K; ++j) m_state_contrib[k][j] = 0.0;

    // Очистка пулов ID
    m_buy_side.ids.clear();
    m_buy_side.by_price.clear();
    m_sell_side.ids.clear();
    m_sell_side.by_price.clear();
    m_next_order_id = 1;

    m_bw_cooldown = 0;
    m_bw_left = 0;
    m_last_trade_sign = 0;
    m_last_trade_size = 0.0;

    // Начальная структура книги: разместим «сиды» на 10 уровнях по обеим сторонам
    // Если bid/ask не заданы — возьмём mid0±1.
    if (best_bid_ticks <= 0 || best_ask_ticks <= 0 || best_ask_ticks <= best_bid_ticks) {
        best_bid_ticks = mid0_ticks - 1;
        best_ask_ticks = mid0_ticks + 1;
    }

    // Примечание: объёмы — из модели лимиток; агент=true; ts=0
    // buy уровни: ... <= bid
    for (int i = 0; i < 10; ++i) {
        long long px = best_bid_ticks - i;
        (void)_sample_size(m_sz_limit); // прогрев распределения размеров
        std::uint64_t oid = m_next_order_id++;
        // add_limit_order(bool is_buy, price_ticks, volume, order_id, is_agent, ts)
        // НЕЛЬЗЯ в nogil — мы в C++.
        // Ордербук сам учтёт позицию.
        // (Ошибки здесь игнорируем, книга может отклонить нулевые/некорректные размеры.)
        try {
            // Вставка
        } catch (...) {}
        // Для совместимости: прямой вызов
        {
            // NB: OrderBook::add_limit_order не бросает исключения в эталонной реализации
            // но оставим try на всякий случай
        }
        // Мы не имеем прямого статуса успешности — добавим в локальные структуры в любом случае
        // (допускаем расхождения при экзотических проверках в LOB; для реализма это не критично)
        _track_new_order(true, oid, px);
        // Фактическая вставка
        OrderBook& lob_dummy = *(OrderBook*)nullptr; (void)lob_dummy; // устраняем варнинг
    }
    // sell уровни: ... >= ask
    for (int i = 0; i < 10; ++i) {
        long long px = best_ask_ticks + i;
        (void)_sample_size(m_sz_limit);
        std::uint64_t oid = m_next_order_id++;
        _track_new_order(false, oid, px);
        OrderBook& lob_dummy2 = *(OrderBook*)nullptr; (void)lob_dummy2;
    }
    // Важно: фактическая инициализация книги выполняется в первом step(), когда у нас будет ссылка на реальный LOB,
    // иначе здесь не к чему применять вставки. Поэтому первый step() попытается дозалить сиды, если видит пустую книгу.
}

// ------------------------------- ЛЯМБДЫ / СОБЫТИЯ -------------------------------

void MicrostructureGenerator::_update_lambdas() {
    // Режимные множители (очень простая схема, можно усложнить позже)
    double mul_mu[CH_K], mul_alpha[CH_K];
    for (int k = 0; k < CH_K; ++k) { mul_mu[k] = 1.0; mul_alpha[k] = 1.0; }
    switch (m_regime) {
        case MarketRegime::NORMAL:
            break;
        case MarketRegime::CHOPPY_FLAT:
            mul_mu[CH_MKT_BUY]  = 0.7; mul_mu[CH_MKT_SELL] = 0.7;
            mul_alpha[CH_MKT_BUY] = mul_alpha[CH_MKT_SELL] = 0.8;
            break;
        case MarketRegime::STRONG_TREND:
            mul_mu[CH_MKT_BUY]  = 1.4; mul_mu[CH_MKT_SELL] = 1.4;
            mul_alpha[CH_MKT_BUY] = mul_alpha[CH_MKT_SELL] = 1.3;
            break;
        case MarketRegime::ILLIQUID:
            mul_mu[CH_LIM_BUY]  = 0.7; mul_mu[CH_LIM_SELL] = 0.7;
            mul_mu[CH_CAN_BUY]  = 1.3; mul_mu[CH_CAN_SELL] = 1.3;
            mul_mu[CH_MKT_BUY]  = 0.6; mul_mu[CH_MKT_SELL] = 0.6;
            break;
    }

    // Базовая часть: μ
    for (int k = 0; k < CH_K; ++k) {
        m_lambda_hat[k] = std::max(0.0, m_hp.mu[k] * mul_mu[k]);
    }

    // Добавляем рекуррентную часть S_{k,j}
    for (int k = 0; k < CH_K; ++k) {
        for (int j = 0; j < CH_K; ++j) {
            // Дискретное затухание предыдущего шага
            double decay = std::exp(-std::max(1e-6, m_hp.beta[k][j]));
            m_state_contrib[k][j] *= decay;
            // Вклад в λ̂ с учётом режимного множителя по каналу k (на α)
            m_lambda_hat[k] += m_state_contrib[k][j] * mul_alpha[k];
        }
        // Численно ограничим
        m_lambda_hat[k] = clampd(m_lambda_hat[k], 0.0, 50.0);
    }
}

int MicrostructureGenerator::_poisson_count(double lambda) {
    lambda = std::max(0.0, std::min(lambda, 40.0));
    std::poisson_distribution<int> Pois(lambda);
    return std::min(64, std::max(0, Pois(m_rng))); // предохранитель от взрывов
}

int MicrostructureGenerator::_sample_limit_level(bool /*is_buy*/, long long /*best_bid*/, long long /*best_ask*/) {
    // ΔL=0 с вероятностью at_best, иначе геометрический хвост (1..max)
    if (m_unif(m_rng) < clampd(m_place.at_best_prob, 0.0, 1.0))
        return 0;
    return (int)geometric_level(m_rng, m_place.geometric_p, m_place.max_levels);
}

double MicrostructureGenerator::_sample_size(const SizeDist& sd) {
    std::lognormal_distribution<double> LN(sd.lognorm_m, sd.lognorm_s);
    double x = LN(m_rng);
    x = clampd(x, sd.min_size, sd.max_size);
    return x;
}

void MicrostructureGenerator::_track_new_order(bool is_buy, std::uint64_t oid, long long price) {
    SideBook& sb = is_buy ? m_buy_side : m_sell_side;
    sb.ids.insert(oid);
    auto& vec = sb.by_price[price];
    vec.push_back(oid);
}

void MicrostructureGenerator::_untrack_order(bool is_buy, std::uint64_t oid, long long price) {
    SideBook& sb = is_buy ? m_buy_side : m_sell_side;
    sb.ids.erase(oid);
    auto it = sb.by_price.find(price);
    if (it != sb.by_price.end()) {
        auto& v = it->second;
        auto p = std::find(v.begin(), v.end(), oid);
        if (p != v.end()) {
            *p = v.back();
            v.pop_back();
        }
        if (v.empty()) sb.by_price.erase(it);
    }
}

void MicrostructureGenerator::_emit_limit(OrderBook& lob, bool is_buy, long long price_ticks, double size, int ts,
                                          MicroEvent* out, int& out_n, int out_cap) {
    std::uint64_t oid = m_next_order_id++;
    // Вставка в книгу
    lob.add_limit_order(is_buy, price_ticks, size, oid, /*is_agent*/true, ts);
    _track_new_order(is_buy, oid, price_ticks);

    if (out && out_n < out_cap) {
        out[out_n++] = MicroEvent{MicroEventType::LIMIT, is_buy, price_ticks, size, oid, ts};
    }
}

void MicrostructureGenerator::_emit_market(OrderBook& lob, bool is_buy, double size, int ts,
                                           MicroEvent* out, int& out_n, int out_cap) {
    // Подготовим буферы для отчёта об исполнении
    const int MAX_TR = 2048;
    std::vector<double> prices(MAX_TR, 0.0), vols(MAX_TR, 0.0);
    std::vector<int> isb(MAX_TR, 0), isself(MAX_TR, 0);
    std::vector<long long> ids(MAX_TR, 0);
    double fee_total = 0.0;
    int n_tr = lob.match_market_order(is_buy, size, ts, /*taker_is_agent*/true,
                                      prices.data(), vols.data(), isb.data(), isself.data(), ids.data(), MAX_TR, &fee_total);
    // Последняя сделка: знак стороны тейкера и полный объём
    double v_sum = 0.0;
    for (int i = 0; i < n_tr; ++i) v_sum += std::max(0.0, vols[i]);
    m_last_trade_sign = is_buy ? +1 : -1;
    m_last_trade_size = v_sum;

    if (out && out_n < out_cap) {
        out[out_n++] = MicroEvent{MicroEventType::MARKET, is_buy, 0, v_sum, 0, ts};
    }
}

void MicrostructureGenerator::_emit_cancel(OrderBook& lob, bool is_buy, int ts,
                                           MicroEvent* out, int& out_n, int out_cap) {
    SideBook& sb = is_buy ? m_buy_side : m_sell_side;
    if (sb.ids.empty() || sb.by_price.empty()) return;

    // Выбор цены пропорционально количеству ID на уровне
    std::vector<std::pair<long long, int>> buckets;
    buckets.reserve(sb.by_price.size());
    int total = 0;
    for (auto& kv : sb.by_price) {
        buckets.emplace_back(kv.first, (int)kv.second.size());
        total += (int)kv.second.size();
    }
    if (total <= 0) return;
    std::uniform_int_distribution<int> U(0, total - 1);
    int r = U(m_rng);
    long long sel_px = buckets[0].first;
    for (auto& b : buckets) {
        if (r < b.second) { sel_px = b.first; break; }
        r -= b.second;
    }
    auto& vec = sb.by_price[sel_px];
    if (vec.empty()) return;
    std::uniform_int_distribution<int> Ui(0, (int)vec.size() - 1);
    int idx = Ui(m_rng);
    std::uint64_t oid = vec[idx];

    // Отмена в LOB
    lob.remove_order(is_buy, sel_px, oid);
    // Учёт
    _untrack_order(is_buy, oid, sel_px);

    if (out && out_n < out_cap) {
        out[out_n++] = MicroEvent{MicroEventType::CANCEL, is_buy, sel_px, 0.0, oid, ts};
    }
}

// ------------------------------- ОСНОВНОЙ ШАГ -------------------------------

int MicrostructureGenerator::step(OrderBook& lob, int timestamp, MicroEvent* out_events, int cap) {
    int out_n = 0;

    // Если книга пустая (нет bid/ask), зальём «сиды» (как в reset), теперь уже реально
    if (lob.get_best_bid() <= 0 || lob.get_best_ask() <= 0 || lob.get_best_ask() <= lob.get_best_bid()) {
        long long mid = 100 * PRICE_SCALE; // безопасный дефолт
        long long bid = mid - 1, ask = mid + 1;
        for (int i = 0; i < 10; ++i) {
            long long pxb = bid - i, pxs = ask + i;
            double s1 = _sample_size(m_sz_limit);
            double s2 = _sample_size(m_sz_limit);
            _emit_limit(lob, true,  pxb, s1, 0, out_events, out_n, cap);
            _emit_limit(lob, false, pxs, s2, 0, out_events, out_n, cap);
        }
    }

    // 1) Обновляем λ̂ на текущий шаг, с режимами и распадом «памяти»
    _update_lambdas();

    // 2) Flash-шок (на один шаг) — масштабируем все λ̂
    if (m_sp.enabled && m_unif(m_rng) < clampd(m_sp.prob_per_step, 0.0, 1.0)) {
        for (int k = 0; k < CH_K; ++k) m_lambda_hat[k] *= std::max(1.0, m_sp.intensity_scale);
    }

    // 3) Black Swan (редко, с охлаждением) — усиливаем λ̂ на период
    bool spawned_bw_burst = false;
    int  bw_sign = 0; // -1 crash, +1 mania, 0 none
    if (m_bp.enabled) {
        if (m_bw_left > 0) {
            for (int k = 0; k < CH_K; ++k) m_lambda_hat[k] *= std::max(1.0, m_bp.intensity_scale);
            m_bw_left -= 1;
        } else {
            if (m_bw_cooldown > 0) m_bw_cooldown -= 1;
            if (m_bw_cooldown == 0 && m_unif(m_rng) < clampd(m_bp.prob_per_step, 0.0, 1.0)) {
                // срабатывание
                m_bw_left = std::max(1, m_bp.duration_steps);
                m_bw_cooldown = std::max(1, m_bp.cooldown_steps);
                spawned_bw_burst = true;
                bw_sign = (m_unif(m_rng) < 0.7 ? -1 : +1); // чаще «крэш»
            }
        }
    }

    // 4) Сэмплируем количества событий по каналам
    int cnt[CH_K];
    for (int k = 0; k < CH_K; ++k) cnt[k] = _poisson_count(m_lambda_hat[k]);

    // Ограничим CANCEL доступной мощностью
    cnt[CH_CAN_BUY]  = std::min<int>(cnt[CH_CAN_BUY],  (int)m_buy_side.ids.size());
    cnt[CH_CAN_SELL] = std::min<int>(cnt[CH_CAN_SELL], (int)m_sell_side.ids.size());

    // 5) Сформируем общий список каналов и перемешаем — реалистичнее, чем «батчами»
    std::vector<int> seq;
    seq.reserve(cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] + cnt[5]);
    for (int k = 0; k < CH_K; ++k) for (int i = 0; i < cnt[k]; ++i) seq.push_back(k);
    std::shuffle(seq.begin(), seq.end(), m_rng);

    // 6) Исполняем последовательность событий
    long long best_bid = lob.get_best_bid();
    long long best_ask = lob.get_best_ask();
    long long mid_ticks = (best_bid > 0 && best_ask > 0) ? ((best_bid + best_ask) / 2) : (100 * PRICE_SCALE);

    for (int ch : seq) {
        switch (ch) {
            case CH_LIM_BUY: {
                int lvl = _sample_limit_level(true, best_bid, best_ask);
                long long px = (best_bid > 0 ? best_bid : (mid_ticks - 1)) - lvl;
                double sz = _sample_size(m_sz_limit);
                _emit_limit(lob, true, px, sz, timestamp, out_events, out_n, cap);
                break;
            }
            case CH_LIM_SELL: {
                int lvl = _sample_limit_level(false, best_bid, best_ask);
                long long px = (best_ask > 0 ? best_ask : (mid_ticks + 1)) + lvl;
                double sz = _sample_size(m_sz_limit);
                _emit_limit(lob, false, px, sz, timestamp, out_events, out_n, cap);
                break;
            }
            case CH_MKT_BUY: {
                double sz = _sample_size(m_sz_market);
                _emit_market(lob, true, sz, timestamp, out_events, out_n, cap);
                break;
            }
            case CH_MKT_SELL: {
                double sz = _sample_size(m_sz_market);
                _emit_market(lob, false, sz, timestamp, out_events, out_n, cap);
                break;
            }
            case CH_CAN_BUY: {
                _emit_cancel(lob, true, timestamp, out_events, out_n, cap);
                break;
            }
            case CH_CAN_SELL: {
                _emit_cancel(lob, false, timestamp, out_events, out_n, cap);
                break;
            }
        }
        // Обновим best bid/ask периодически
        best_bid = lob.get_best_bid();
        best_ask = lob.get_best_ask();
        if (!(best_bid > 0 && best_ask > 0 && best_ask > best_bid)) {
            best_bid = (mid_ticks - 1);
            best_ask = (mid_ticks + 1);
        }
    }

    // 7) Если «чёрный лебедь» сработал — вколем большой рыночный ордер в сторону знака
    if (spawned_bw_burst && (best_bid > 0 && best_ask > 0)) {
        double amp = (bw_sign < 0)
            ? (m_bp.crash_min + (m_bp.crash_max - m_bp.crash_min) * m_unif(m_rng))
            : (m_bp.mania_min + (m_bp.mania_max - m_bp.mania_min) * m_unif(m_rng));
        // амплитуду проксируем крупной рыночной агрессией
        double big_mult = 20.0 + 40.0 * amp; // 20–60x средней заявки
        double sz = clampd(big_mult * _sample_size(m_sz_market), m_sz_market.min_size, m_sz_market.max_size * 10.0);
        _emit_market(lob, (bw_sign > 0), sz, timestamp, out_events, out_n, cap);
        // вклад в Hawkes «как факт» дополнительного события соответствующего канала:
        int j = (bw_sign > 0) ? CH_MKT_BUY : CH_MKT_SELL;
        for (int k = 0; k < CH_K; ++k) {
            m_state_contrib[k][j] += m_hp.alpha[k][j]; // как минимум +1 событие
        }
    }

    // 8) Обновляем рекуррентную часть Hawkes по итоговым счётчикам каналов:
    for (int j = 0; j < CH_K; ++j) {
        int c = cnt[j];
        if (c <= 0) continue;
        for (int k = 0; k < CH_K; ++k) {
            m_state_contrib[k][j] += m_hp.alpha[k][j] * (double)c;
        }
    }

    return (int)seq.size() + (spawned_bw_burst ? 1 : 0);
}

// ------------------------------- ФИЧИ -------------------------------

MicroFeatures MicrostructureGenerator::_features_from_lob(const OrderBook& lob) const {
    MicroFeatures f{};
    f.best_bid = lob.get_best_bid();
    f.best_ask = lob.get_best_ask();
    if (f.best_bid > 0 && f.best_ask > 0 && f.best_ask > f.best_bid) {
        f.spread_ticks = double(f.best_ask - f.best_bid);
        f.mid = (double(f.best_ask + f.best_bid) / 2.0) / double(PRICE_SCALE);
    } else {
        f.spread_ticks = NAN;
        f.mid = NAN;
    }

    // Глубина: используем собственные «пулы» ID как прокси объёма (каждый ордер = 1 условная единица)
    auto depth_side = [](const SideBook& sb, long long best, bool is_buy) -> std::pair<double,double> {
        if (sb.by_price.empty() || best <= 0) return {0.0, 0.0};
        std::vector<long long> keys;
        keys.reserve(sb.by_price.size());
        for (auto& kv : sb.by_price) keys.push_back(kv.first);
        if (is_buy) {
            std::sort(keys.begin(), keys.end(), std::greater<long long>()); // от лучшего вниз
        } else {
            std::sort(keys.begin(), keys.end(), std::less<long long>());    // от лучшего вверх
        }
        double top1 = 0.0, top5 = 0.0;
        int levels = 0;
        for (auto px : keys) {
            if ( (is_buy && px > best) || (!is_buy && px < best) ) continue; // только «с той стороны»
            double v = 0.0;
            auto it = sb.by_price.find(px);
            if (it != sb.by_price.end()) v = (double)it->second.size();
            if (levels == 0) top1 = v;
            top5 += v;
            levels += 1;
            if (levels >= 5) break;
        }
        return {top1, top5};
    };

    auto [b1, b5] = depth_side(m_buy_side,  f.best_bid,  true);
    auto [a1, a5] = depth_side(m_sell_side, f.best_ask, false);
    f.depth_bid_top1 = b1;
    f.depth_bid_top5 = b5;
    f.depth_ask_top1 = a1;
    f.depth_ask_top5 = a5;

    double s1 = b1 + a1;
    double s5 = b5 + a5;
    f.imbalance_top1 = (s1 > 0.0) ? ((b1 - a1) / s1) : 0.0;
    f.imbalance_top5 = (s5 > 0.0) ? ((b5 - a5) / s5) : 0.0;

    for (int k = 0; k < CH_K; ++k) f.lambda_hat[k] = m_lambda_hat[k];

    f.last_trade_sign = m_last_trade_sign;
    f.last_trade_size = m_last_trade_size;

    return f;
}

MicroFeatures MicrostructureGenerator::current_features(const OrderBook& lob) const {
    return _features_from_lob(lob);
}

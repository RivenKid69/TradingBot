#include "MarketSimulator.h"
#include <cstring> // std::memset
#include <fstream>
#include <regex>
#include <sstream>
#include <cstdlib>

#ifndef NAN
#define NAN std::numeric_limits<double>::quiet_NaN()
#endif

static inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

MarketSimulator::MarketSimulator(
    double* price,
    double* open,
    double* high,
    double* low,
    double* volume_usd,
    std::size_t n_steps,
    std::uint64_t seed
)
    : m_close(price)
    , m_open(open)
    , m_high(high)
    , m_low(low)
    , m_volume_usd(volume_usd)
    , m_n(n_steps)
    , m_last_close(0.0)
    , m_last_high(0.0)
    , m_last_low(0.0)
    , m_curr_regime(MarketRegime::NORMAL)
    , m_shocks_enabled(false)
    , m_shock_p(0.0)
    , m_stdnorm(0.0, 1.0)
    , m_unif(0.0, 1.0)
{
    set_seed(seed);
    initialize_defaults();

    // Инициализация индикаторов NaN
    auto init_vec = [this](std::vector<double>& v) { v.assign(m_n, NAN); };
    init_vec(v_ma5); init_vec(v_ma20); init_vec(v_atr); init_vec(v_rsi);
    init_vec(v_macd); init_vec(v_macd_signal); init_vec(v_mom); init_vec(v_cci);
    init_vec(v_obv); init_vec(v_bb_low); init_vec(v_bb_up);

    m_shock_marks.assign(m_n, 0);
    m_black_swan_marks.assign(m_n, 0);

    // Если внешние буферы не нулевые — почистим их
    if (m_close) std::memset(m_close, 0, sizeof(double) * m_n);
    if (m_open) std::memset(m_open, 0, sizeof(double) * m_n);
    if (m_high) std::memset(m_high, 0, sizeof(double) * m_n);
    if (m_low) std::memset(m_low, 0, sizeof(double) * m_n);
    if (m_volume_usd) std::memset(m_volume_usd, 0, sizeof(double) * m_n);

    // seasonality multipliers default to 1
    m_liq_mult.fill(1.0);
}

void MarketSimulator::initialize_defaults() {
    // Попытка загрузить параметры режимов из JSON-конфигурации
    std::string path = "configs/market_regimes.json";
    if (const char* env_p = std::getenv("MARKET_REGIMES_JSON")) {
        path = env_p;
    }
    bool loaded = false;
    std::ifstream in(path);
    if (in.good()) {
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto parse_regime = [&](const std::string& name, RegimeParams& out) {
            try {
                std::regex re("\"" + name + "\"\\s*:\\s*\{[^}]*\"mu\"\\s*:\\s*([-0-9.eE]+)[^}]*\"sigma\"\\s*:\\s*([-0-9.eE]+)[^}]*\"kappa\"\\s*:\\s*([-0-9.eE]+)[^}]*\"avg_volume\"\\s*:\\s*([-0-9.eE]+)[^}]*\"avg_spread\"\\s*:\\s*([-0-9.eE]+)");
                std::smatch m;
                if (std::regex_search(s, m, re)) {
                    out.mu = std::stod(m[1]);
                    out.sigma = std::stod(m[2]);
                    out.kappa = std::stod(m[3]);
                    out.avg_volume = std::stod(m[4]);
                    out.avg_spread = std::stod(m[5]);
                    return true;
                }
            } catch (const std::regex_error&) {
                // ignore malformed patterns and fall back to defaults
            }
            return false;
        };
        loaded = parse_regime("NORMAL",       m_params[(int)MarketRegime::NORMAL]);
        loaded = parse_regime("CHOPPY_FLAT",  m_params[(int)MarketRegime::CHOPPY_FLAT]) || loaded;
        loaded = parse_regime("STRONG_TREND", m_params[(int)MarketRegime::STRONG_TREND]) || loaded;
        loaded = parse_regime("ILLIQUID",     m_params[(int)MarketRegime::ILLIQUID]) || loaded;
        // распределение режимов
        try {
            std::regex re_probs(R"("regime_probs"\s*:\s*\[([^\]]+)\])");
            std::smatch mp;
            if (std::regex_search(s, mp, re_probs)) {
                std::string arr = mp[1];
                std::stringstream ss(arr);
                for (int i = 0; i < 4; ++i) {
                    std::string token;
                    if (!std::getline(ss, token, ',')) break;
                    try { m_regime_probs[i] = std::max(0.0, std::stod(token)); }
                    catch (...) { m_regime_probs[i] = 0.0; }
                }
                double sum = 0.0; for (double p: m_regime_probs) sum += p; if (sum>0) for (double& p: m_regime_probs) p/=sum;
            } else {
                m_regime_probs = {0.25,0.25,0.25,0.25};
            }
            // flash shock
            std::regex re_fp(R"("flash_shock"[^}]*"probability"\s*:\s*([-0-9.eE]+))");
            if (std::regex_search(s, mp, re_fp)) {
                m_shock_p = clampd(std::stod(mp[1]),0.0,1.0);
                m_shocks_enabled = m_shock_p > 0.0;
            }
            std::regex re_mags(R"("flash_shock"[^}]*"magnitudes"\s*:\s*\[([^\]]+)\])");
            if (std::regex_search(s, mp, re_mags)) {
                std::string arr = mp[1];
                std::stringstream ss(arr);
                std::string token;
                m_shock_mags.clear();
                while (std::getline(ss, token, ',')) {
                    try { m_shock_mags.push_back(std::stod(token)); } catch (...) {}
                }
            }
        } catch (const std::regex_error&) {
            m_regime_probs = {0.25,0.25,0.25,0.25};
        }
    }
    if (!loaded) {
        // Параметры режимов (часовой шаг, лог-доходности)
        // mu ~ средний дрейф (в час), sigma ~ вола
        m_params[(int)MarketRegime::NORMAL]       = { 0.0000, 0.0100, 0.00, 1000.0, 0.0005 };
        m_params[(int)MarketRegime::CHOPPY_FLAT]  = { 0.0000, 0.0040, 0.50, 800.0, 0.0007 }; // OU reversion
        m_params[(int)MarketRegime::STRONG_TREND] = { 0.0008, 0.0120, 0.00, 1200.0, 0.0004 };
        m_params[(int)MarketRegime::ILLIQUID]     = { 0.0000, 0.0200, 0.00, 500.0, 0.0010 };
        // Равномерное распределение режимов по умолчанию
        m_regime_probs = {0.25, 0.25, 0.25, 0.25};
        m_shock_p = 0.01;
        m_shocks_enabled = true;
        m_shock_mags = {0.005,0.01,0.015,0.02};
    }

    // Индикаторы — нач. значения
    sum5 = 0.0; sum20 = 0.0; sum20_sq = 0.0;
    rsi_init = false; avg_gain14 = 0.0; avg_loss14 = 0.0;
    atr_init = false; atr14 = 0.0; prev_close_for_atr = NAN;
    ema12 = ema26 = ema9 = 0.0; ema12_init = ema26_init = ema9_init = false;
    obv = 0.0;
    w_close5.clear(); w_close20.clear(); w_tp20.clear(); w_var_close20.clear();
    w_close10.clear();
}

void MarketSimulator::set_seed(std::uint64_t seed) {
    if (seed == 0) {
        std::random_device rd;
        m_rng.seed(((std::uint64_t)rd() << 32) ^ (std::uint64_t)rd());
    } else {
        m_rng.seed(seed);
    }
}

void MarketSimulator::set_regime_distribution(const std::array<double, 4>& probs) {
    double s = 0.0;
    for (double p: probs) s += std::max(0.0, p);
    if (s <= 0.0) {
        m_regime_probs = {0.25, 0.25, 0.25, 0.25};
    } else {
        for (int i = 0; i < 4; ++i) m_regime_probs[i] = std::max(0.0, probs[i]) / s;
    }
}

void MarketSimulator::enable_random_shocks(bool enable, double probability_per_step) {
    m_shocks_enabled = enable;
    m_shock_p = clampd(probability_per_step, 0.0, 1.0);
}

void MarketSimulator::force_market_regime(MarketRegime regime, std::size_t start, std::size_t duration) {
    if (duration == 0) return;
    m_forced.push_back({regime, start, start + duration});
}

MarketRegime MarketSimulator::pick_regime(std::size_t i) {
    // Проверка принудительных окон
    for (const auto& seg : m_forced) {
        if (i >= seg.start && i < seg.end) return seg.regime;
    }
    // Сэмплинг по распределению
    double u = m_unif(m_rng);
    double acc = 0.0;
    for (int k = 0; k < 4; ++k) {
        acc += m_regime_probs[k];
        if (u <= acc) return (MarketRegime)k;
    }
    return MarketRegime::NORMAL;
}

void MarketSimulator::apply_black_swan_if_any(std::size_t i, double& close_ref) {
    // Метка уже должна быть установлена из step() при триггере
    int mark = m_black_swan_marks[i];
    if (mark == 0) return;
    // Масштаб скачка
    double impact = 0.0;
    if (mark < 0) {
        // "крэш": -35%…-20%
        impact = -(0.20 + 0.15 * m_unif(m_rng));
    } else {
        // "мания": +20%…+40%
        impact = +(0.20 + 0.20 * m_unif(m_rng));
    }
    close_ref = close_ref * (1.0 + impact);
}

void MarketSimulator::apply_flash_shock_if_any(std::size_t i, double& close_ref) {
    if (!m_shocks_enabled) return;
    int mark = m_shock_marks[i];
    if (mark == 0) return;
    double mag = 0.0;
    if (!m_shock_mags.empty()) {
        std::uniform_int_distribution<std::size_t> idx(0, m_shock_mags.size() - 1);
        mag = m_shock_mags[idx(m_rng)];
    } else {
        // fallback uniform range
        mag = 0.005 + 0.015 * m_unif(m_rng);
    }
    if (mark < 0) mag = -mag;
    close_ref = close_ref * (1.0 + mag);
}

void MarketSimulator::set_liquidity_seasonality(const std::array<double, 168>& multipliers) {
    m_liq_mult = multipliers;
}

void MarketSimulator::update_ohlc_and_volume(std::size_t i, double prev_close, double new_close) {
    // Простейшая модель формирования high/low вокруг open/close:
    double openv = prev_close;
    double closev = new_close;
    double swing = std::fabs(closev - openv);
    double wiggle = (0.25 + 0.75 * m_unif(m_rng)) * (swing + 1e-8);

    double highv = std::max(openv, closev) + wiggle;
    double lowv  = std::min(openv, closev) - wiggle;
    if (lowv < 0.0) lowv = 0.0;

    if (m_open)  m_open[i] = openv;
    if (m_high)  m_high[i] = highv;
    if (m_low)   m_low[i]  = lowv;
    if (m_close) m_close[i] = closev;

    // Объём: базовый объём нарастает с волатильностью; перевод в USD уже учтён (close*vol)
    double base_vol = 1e3; // базовый условный объём
    double vol_mult = 1.0 + 25.0 * swing / std::max(1.0, openv);
    double vol_usd = std::max(0.0, base_vol * vol_mult);
    vol_usd *= m_liq_mult[i % 168];
    if (m_volume_usd) m_volume_usd[i] = vol_usd;
}

static inline double ema_step(double prev, double x, double alpha, bool& init) {
    if (!init) { init = true; return x; }
    return alpha * x + (1.0 - alpha) * prev;
}

void MarketSimulator::update_indicators(std::size_t i) {
    const double closev = m_close ? m_close[i] : m_last_close;
    const double highv  = m_high  ? m_high[i]  : m_last_high;
    const double lowv   = m_low   ? m_low[i]   : m_last_low;
    const double tp = (highv + lowv + closev) / 3.0;

    // MA5/MA20 + Bollinger(20)
    w_close5.push_back(closev); sum5 += closev;
    if (w_close5.size() > 5) { sum5 -= w_close5.front(); w_close5.pop_front(); }
    if (w_close5.size() == 5) v_ma5[i] = sum5 / 5.0;

    w_close20.push_back(closev); sum20 += closev; sum20_sq += closev * closev;
    if (w_close20.size() > 20) {
        double old = w_close20.front(); w_close20.pop_front();
        sum20 -= old; sum20_sq -= old * old;
    }
    if (w_close20.size() == 20) {
        double mean = sum20 / 20.0;
        double var  = std::max(0.0, sum20_sq / 20.0 - mean * mean);
        double sd   = std::sqrt(var);
        v_ma20[i]   = mean;
        v_bb_low[i] = mean - 2.0 * sd;
        v_bb_up[i]  = mean + 2.0 * sd;
    }

    // ATR(14)
    double tr = 0.0;
    if (i == 0 || std::isnan(prev_close_for_atr)) {
        tr = highv - lowv;
    } else {
        double cprev = prev_close_for_atr;
        tr = std::max({highv - lowv, std::fabs(highv - cprev), std::fabs(lowv - cprev)});
    }
    prev_close_for_atr = closev;
    if (!atr_init && i >= 13) { // накопили 14 TR
        // простая средняя первых 14 TR
        // Для точности можно хранить окно TR, но упростим: при i==13 считаем первичную среднюю через перезапуск.
        atr_init = true;
        atr14 = tr; // первый запуск — не идеально, но дальше будет Wilder-сглаживание
    }
    if (atr_init) {
        // Wilder smoothing: ATR_t = (ATR_{t-1}*13 + TR_t)/14
        atr14 = (atr14 * 13.0 + tr) / 14.0;
        v_atr[i] = atr14;
    }

    // RSI(14), Wilder
    static double prev_close_for_rsi = NAN;
    double change = 0.0;
    if (i > 0 && !std::isnan(prev_close_for_rsi)) change = closev - prev_close_for_rsi;
    prev_close_for_rsi = closev;
    double gain = change > 0 ? change : 0.0;
    double loss = change < 0 ? -change : 0.0;
    if (!rsi_init && i >= 14) {
        rsi_init = true;
        avg_gain14 = gain;
        avg_loss14 = loss;
    }
    if (rsi_init) {
        avg_gain14 = (avg_gain14 * 13.0 + gain) / 14.0;
        avg_loss14 = (avg_loss14 * 13.0 + loss) / 14.0;
        double rs = (avg_loss14 == 0.0) ? std::numeric_limits<double>::infinity() : (avg_gain14 / avg_loss14);
        double rsi = 100.0 - (100.0 / (1.0 + rs));
        v_rsi[i] = rsi;
    }

    // MACD(12,26) + signal(9) на close
    const double alpha12 = 2.0 / (12.0 + 1.0);
    const double alpha26 = 2.0 / (26.0 + 1.0);
    const double alpha9  = 2.0 / ( 9.0 + 1.0);
    ema12 = ema_step(ema12, closev, alpha12, ema12_init);
    ema26 = ema_step(ema26, closev, alpha26, ema26_init);
    double macd = ema12 - ema26;
    v_macd[i] = macd;
    ema9  = ema_step(ema9, macd, alpha9, ema9_init);
    v_macd_signal[i] = ema9;

    // Momentum(10)
    w_close10.push_back(closev);
    if (w_close10.size() > 10) w_close10.pop_front();
    if (w_close10.size() == 10) v_mom[i] = closev - w_close10.front();

    // CCI(20): (TP - SMA20) / (0.015 * mean_dev)
    w_tp20.push_back(tp);
    if (w_tp20.size() > 20) w_tp20.pop_front();
    if (w_close20.size() == 20) {
        double sma = v_ma20[i];
        double md = 0.0;
        for (double x : w_tp20) md += std::fabs(x - sma);
        md /= 20.0;
        if (md > 0.0) v_cci[i] = (tp - sma) / (0.015 * md);
    }

    // OBV: условно используем volume_usd как прокси объёма
    double volu = m_volume_usd ? m_volume_usd[i] : 0.0;
    if (i > 0) {
        double prevc = m_close ? m_close[i - 1] : m_last_close;
        if (closev > prevc) obv += volu;
        else if (closev < prevc) obv -= volu;
    }
    v_obv[i] = obv;
}

double MarketSimulator::get_or_nan(const std::vector<double>& v, std::size_t i) {
    return (i < v.size()) ? v[i] : NAN;
}

double MarketSimulator::step(std::size_t i, double black_swan_probability, bool is_training_mode) {
    if (i >= m_n) return m_last_close;

    // Инициализация первого шага
    if (i == 0) {
        // начальную цену возьмём из внешнего буфера (если там >0) иначе 100.0
        double init = (m_close && m_close[0] > 0.0) ? m_close[0] : 100.0;
        m_last_close = init;
        m_last_high = init;
        m_last_low  = init;
        if (m_open)  m_open[0] = init;
        if (m_high)  m_high[0] = init;
        if (m_low)   m_low[0]  = init;
        if (m_close) m_close[0] = init;
        if (m_volume_usd) m_volume_usd[0] = 0.0;
        update_indicators(0);
        return init;
    }

    // Выбор/поддержание режима
    m_curr_regime = pick_regime(i);
    const auto rp = m_params[(int)m_curr_regime];

    // Генерация лог-доходности в зависимости от режима
    double z = m_stdnorm(m_rng);
    double r = rp.mu + rp.sigma * z;

    // CHOPPY_FLAT: добавим OU-компонент (reversion)
    static double ou_state = 0.0;
    if (m_curr_regime == MarketRegime::CHOPPY_FLAT) {
        double eta = 0.003 * m_stdnorm(m_rng);
        ou_state = (1.0 - rp.kappa) * ou_state + eta;
        r += ou_state;
    }

    // Новый close
    double new_close = m_last_close * std::exp(r);

    // Вероятный flash-шок?
    m_shock_marks[i] = 0;
    if (m_shocks_enabled && m_unif(m_rng) < m_shock_p) {
        m_shock_marks[i] = (m_unif(m_rng) < 0.5) ? -1 : +1;
    }
    apply_flash_shock_if_any(i, new_close);

    // "Черный лебедь"?
    m_black_swan_marks[i] = 0;
    if (is_training_mode && black_swan_probability > 0.0 && m_unif(m_rng) < black_swan_probability) {
        m_black_swan_marks[i] = (m_unif(m_rng) < 0.7) ? -1 : +1; // чаще "крэш"
    }
    apply_black_swan_if_any(i, new_close);

    // Обновляем OHLCV и индикаторы
    update_ohlc_and_volume(i, m_last_close, new_close);
    m_last_high = m_high ? m_high[i] : std::max(m_last_close, new_close);
    m_last_low  = m_low  ? m_low[i]  : std::min(m_last_close, new_close);
    m_last_close = new_close;
    update_indicators(i);

    return new_close;
}

// --- геттеры индикаторов ---
int MarketSimulator::shock_triggered(std::size_t i) const {
    if (i >= m_shock_marks.size()) return 0;
    return m_shock_marks[i];
}
double MarketSimulator::get_ma5(std::size_t i) const        { return get_or_nan(v_ma5, i); }
double MarketSimulator::get_ma20(std::size_t i) const       { return get_or_nan(v_ma20, i); }
double MarketSimulator::get_atr(std::size_t i) const        { return get_or_nan(v_atr, i); }
double MarketSimulator::get_rsi(std::size_t i) const        { return get_or_nan(v_rsi, i); }
double MarketSimulator::get_macd(std::size_t i) const       { return get_or_nan(v_macd, i); }
double MarketSimulator::get_macd_signal(std::size_t i) const{ return get_or_nan(v_macd_signal, i); }
double MarketSimulator::get_momentum(std::size_t i) const   { return get_or_nan(v_mom, i); }
double MarketSimulator::get_cci(std::size_t i) const        { return get_or_nan(v_cci, i); }
double MarketSimulator::get_obv(std::size_t i) const        { return get_or_nan(v_obv, i); }
double MarketSimulator::get_bb_lower(std::size_t i) const   { return get_or_nan(v_bb_low, i); }
double MarketSimulator::get_bb_upper(std::size_t i) const   { return get_or_nan(v_bb_up, i); }

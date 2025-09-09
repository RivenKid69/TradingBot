#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>
#include <random>
#include <utility>
#include <limits>
#include <cmath>
#include <algorithm>
#include <deque>

#include "core_constants.h" // MarketRegime, PRICE_SCALE

// Полнофункциональный стохастический симулятор рынка (1h бары) с режимами,
// случайными "шоками", редкими "черными лебедями" и онлайн-индикаторами.
// Генерирует OHLCV в переданные массивы-буферы и вычисляет индикаторы.

class MarketSimulator {
public:
    // Внешние буферы (длиной >= n_steps):
    // - price[i]  : close (в денежных единицах)
    // - open[i], high[i], low[i] : OHLC (денежные)
    // - volume_usd[i]: объём в денежных единицах
    MarketSimulator(
        double* price,
        double* open,
        double* high,
        double* low,
        double* volume_usd,
        std::size_t n_steps,
        std::uint64_t seed = 0
    );

    // Один шаг симуляции. Возвращает close цены на шаге i.
    // black_swan_probability: вероятность запуска "черного лебедя" на шаге.
    // is_training_mode: если true, "лебедь" может случиться; если false — подавляется.
    double step(std::size_t i, double black_swan_probability, bool is_training_mode);

    // Настройка распределения режимов (NORMAL, CHOPPY_FLAT, STRONG_TREND, ILLIQUID).
    // Вектор нормализуется внутри.
    void set_regime_distribution(const std::array<double, 4>& probs);

    // Включить/выключить случайные шоки flash-движений (buy/sell) с вероятностью per-step.
    void enable_random_shocks(bool enable, double probability_per_step);

    // Зафиксировать принудительный режим на [start, start+duration).
    void force_market_regime(MarketRegime regime, std::size_t start, std::size_t duration);

    // Установить сезонные коэффициенты ликвидности (168 часов недели).
    void set_liquidity_seasonality(const std::array<double, 168>& multipliers);

    // Была ли на шаге i триггернута вспышка-шок? (-1: sell, +1: buy, 0: нет)
    int shock_triggered(std::size_t i) const;

    // --- Геттеры индикаторов (NaN, если недоступно/индекс вне диапазона) ---
    double get_ma5(std::size_t i) const;
    double get_ma20(std::size_t i) const;
    double get_atr(std::size_t i) const;            // ATR(14), Wilder
    double get_rsi(std::size_t i) const;            // RSI(14), Wilder
    double get_macd(std::size_t i) const;           // MACD(12,26)
    double get_macd_signal(std::size_t i) const;    // signal(9)
    double get_momentum(std::size_t i) const;       // MOM(10)
    double get_cci(std::size_t i) const;            // CCI(20)
    double get_obv(std::size_t i) const;            // OBV (в "условных" единицах)
    double get_bb_lower(std::size_t i) const;       // Bollinger 20, K=2
    double get_bb_upper(std::size_t i) const;

    // Установить зерно ГПСЧ
    void set_seed(std::uint64_t seed);

private:
    // Внешние буферы
    double* m_close;
    double* m_open;
    double* m_high;
    double* m_low;
    double* m_volume_usd;
    std::size_t m_n;

    // Текущее состояние
    double m_last_close;
    double m_last_high;
    double m_last_low;
    MarketRegime m_curr_regime;

    // Параметры режимов (дрейф/вола для лог-возврата; OU для FLAT)
    struct RegimeParams {
        double mu;         // средний дрейф лог-доходности
        double sigma;      // волатильность лог-доходности
        double kappa;      // сила возврата к среднему (для CHOPPY_FLAT)
        double avg_volume; // средний объём
        double avg_spread; // средний спред
    };
    std::array<RegimeParams, 4> m_params;

    // Распределение режимов
    std::array<double, 4> m_regime_probs;

    // Принудительные окна режима
    struct ForcedSeg { MarketRegime regime; std::size_t start; std::size_t end; };
    std::vector<ForcedSeg> m_forced;

    // Шоки
    bool m_shocks_enabled;
    double m_shock_p;
    std::vector<int> m_shock_marks; // -1,0,+1
    std::vector<double> m_shock_mags; // распределение величины шока

    // "Черные лебеди" (флаги по шагам)
    std::vector<int> m_black_swan_marks; // -1 crash, +1 mania, 0 none

    // Индикаторы (на каждый шаг)
    std::vector<double> v_ma5, v_ma20, v_atr, v_rsi, v_macd, v_macd_signal;
    std::vector<double> v_mom, v_cci, v_obv, v_bb_low, v_bb_up;

    // Внутренние окна/накопители
    std::deque<double> w_close5, w_close20, w_tp20; // для MA и CCI/BB
    std::deque<double> w_var_close20;               // для std (BB)
    double sum5, sum20, sum20_sq;
    // RSI (Wilder)
    bool rsi_init;
    double avg_gain14, avg_loss14;
    // ATR (Wilder)
    bool atr_init;
    double atr14, prev_close_for_atr;
    // MACD EMA
    double ema12, ema26, ema9;
    bool ema12_init, ema26_init, ema9_init;
    // OBV
    double obv;
    // Momentum
    std::deque<double> w_close10;

    // ГПСЧ
    std::mt19937_64 m_rng;
    std::normal_distribution<double> m_stdnorm;
    std::uniform_real_distribution<double> m_unif;

    // Сезонные коэффициенты ликвидности по часам недели
    std::array<double, 168> m_liq_mult;

    // --- внутренние методы ---
    void initialize_defaults();
    MarketRegime pick_regime(std::size_t i);
    void apply_black_swan_if_any(std::size_t i, double& close_ref);
    void apply_flash_shock_if_any(std::size_t i, double& close_ref);

    void update_ohlc_and_volume(std::size_t i, double prev_close, double new_close);
    void update_indicators(std::size_t i);

    // безопасное получение значения вектора (NaN при выходе)
    static double get_or_nan(const std::vector<double>& v, std::size_t i);
};

// core_constants.h
// Единый источник правды для C++ уровня. Согласован с core_constants.py и core_constants.pxd.

#ifndef CORE_CONSTANTS_H
#define CORE_CONSTANTS_H

// PRICE_SCALE: количество тиков на единицу цены.
// 100 означает шаг цены 0.01 в денежных единицах.
static constexpr int PRICE_SCALE = 100;

enum MarketRegime {
    NORMAL = 0,
    CHOPPY_FLAT = 1,
    STRONG_TREND = 2,
    ILLIQUID = 3
};

#endif // CORE_CONSTANTS_H

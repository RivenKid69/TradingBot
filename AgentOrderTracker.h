#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <optional>
#include <utility>
#include <limits>
#include <cmath>

struct AgentOrderInfo {
    long long price;    // price in ticks
    bool is_buy_side;   // true=buy, false=sell
};

class AgentOrderTracker {
public:
    AgentOrderTracker() = default;

    // Добавить новую запись или обновить существующую (переместит ID между "корзинами цен").
    // Возвращает true, если запись была создана впервые; false — если обновлена.
    bool add_or_update(std::uint64_t order_id, long long price, bool is_buy_side) {
        auto it = id_to_info_map.find(order_id);
        if (it != id_to_info_map.end()) {
            // Уже был — возможно, сменился уровень цены/сторона: вынести из старой корзины.
            const auto &old = it->second;
            auto pit = price_to_ids_map.find(old.price);
            if (pit != price_to_ids_map.end()) {
                pit->second.erase(order_id);
                if (pit->second.empty()) {
                    price_to_ids_map.erase(pit);
                }
            }
            it->second.price = price;
            it->second.is_buy_side = is_buy_side;
            price_to_ids_map[price].insert(order_id);
            return false;
        } else {
            id_to_info_map.emplace(order_id, AgentOrderInfo{price, is_buy_side});
            price_to_ids_map[price].insert(order_id);
            return true;
        }
    }

    // Удалить запись по order_id. Возвращает true, если запись существовала.
    bool remove(std::uint64_t order_id) {
        auto it = id_to_info_map.find(order_id);
        if (it == id_to_info_map.end()) return false;
        const long long old_price = it->second.price;
        auto pit = price_to_ids_map.find(old_price);
        if (pit != price_to_ids_map.end()) {
            pit->second.erase(order_id);
            if (pit->second.empty()) {
                price_to_ids_map.erase(pit);
            }
        }
        id_to_info_map.erase(it);
        return true;
    }

    // Очистить все данные.
    void clear() {
        id_to_info_map.clear();
        price_to_ids_map.clear();
    }

    // Есть ли запись по ID.
    bool contains(std::uint64_t order_id) const {
        return id_to_info_map.find(order_id) != id_to_info_map.end();
    }

    // Размер (кол-во отслеживаемых ордеров).
    std::size_t size() const {
        return id_to_info_map.size();
    }

    // Получить информацию по ID.
    std::optional<AgentOrderInfo> get_info(std::uint64_t order_id) const {
        auto it = id_to_info_map.find(order_id);
        if (it == id_to_info_map.end()) return std::nullopt;
        return it->second;
    }

    // Вернуть любую (первую попавшуюся) запись: удобно для отладки.
    // Возвращает false, если трекер пуст.
    bool get_first_order_info(std::uint64_t &order_id_out, AgentOrderInfo &info_out) const {
        if (id_to_info_map.empty()) return false;
        auto it = id_to_info_map.begin();
        order_id_out = it->first;
        info_out = it->second;
        return true;
    }

    // Найти ближайшую по цене корзину и вернуть (order_id, price).
    // Детализация выбора:
    // 1) если есть точное совпадение уровня — берём минимальный order_id из этой корзины;
    // 2) иначе сравниваем ближайшие нижнюю и верхнюю корзины по |price - target|;
    //    при равенстве расстояний — предпочитаем НИЖНЮЮ цену; при полном равенстве — меньший order_id.
    std::pair<std::uint64_t, long long> nearest_by_price(long long target_price) const {
        if (price_to_ids_map.empty()) {
            return {0u, std::numeric_limits<long long>::min()};
        }

        auto it_lower_or_equal = price_to_ids_map.upper_bound(target_price);
        auto it_ge = it_lower_or_equal; // первая строго больше target

        // Кандидат A: >= target
        std::optional<std::pair<long long, std::uint64_t>> cand_ge;
        if (it_ge != price_to_ids_map.end()) {
            long long p = it_ge->first;
            // детерминированно — берём минимальный order_id в корзине
            std::uint64_t oid = it_ge->second.empty() ? 0u : *it_ge->second.begin();
            cand_ge = std::make_pair(p, oid);
        }

        // Кандидат B: <= target (предыдущий в map)
        std::optional<std::pair<long long, std::uint64_t>> cand_le;
        if (it_lower_or_equal != price_to_ids_map.begin()) {
            auto it_le = std::prev(it_lower_or_equal);
            long long p = it_le->first;
            std::uint64_t oid = it_le->second.empty() ? 0u : *it_le->second.begin();
            cand_le = std::make_pair(p, oid);
        }

        // Если есть точное совпадение — победа сразу
        if (cand_ge.has_value() && cand_ge->first == target_price && cand_ge->second != 0u) {
            return {cand_ge->second, cand_ge->first};
        }
        if (cand_le.has_value() && cand_le->first == target_price && cand_le->second != 0u) {
            return {cand_le->second, cand_le->first};
        }

        // Если один из кандидатов отсутствует — берём другой
        if (cand_ge.has_value() && !cand_le.has_value()) {
            return {cand_ge->second, cand_ge->first};
        }
        if (cand_le.has_value() && !cand_ge.has_value()) {
            return {cand_le->second, cand_le->first};
        }
        if (!cand_ge.has_value() && !cand_le.has_value()) {
            return {0u, std::numeric_limits<long long>::min()};
        }

        // Оба есть — сравнить расстояния
        long long p_ge = cand_ge->first;
        long long p_le = cand_le->first;
        std::uint64_t oid_ge = cand_ge->second;
        std::uint64_t oid_le = cand_le->second;

        long long d_ge = std::llabs(p_ge - target_price);
        long long d_le = std::llabs(target_price - p_le);

        if (d_ge < d_le) {
            return {oid_ge, p_ge};
        } else if (d_le < d_ge) {
            return {oid_le, p_le};
        } else {
            // равные расстояния — предпочитаем нижнюю цену, при равенстве — меньший order_id
            if (p_le < p_ge) return {oid_le, p_le};
            if (p_ge < p_le) return {oid_ge, p_ge};
            // p_le == p_ge — выберем меньший ID
            return {oid_le < oid_ge ? oid_le : oid_ge, p_le};
        }
    }

private:
    std::unordered_map<std::uint64_t, AgentOrderInfo> id_to_info_map;
    std::map<long long, std::unordered_set<std::uint64_t>> price_to_ids_map;
};

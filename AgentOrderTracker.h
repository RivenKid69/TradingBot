#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>
#include <limits>

struct AgentOrderInfo {
    long long price;    // price in ticks
    bool is_buy_side;   // true=buy, false=sell
};

class AgentOrderTracker {
public:
    AgentOrderTracker() = default;

    // Добавить новую запись или обновить существующую (переместит ID между "корзинами цен").
    void add(long long order_id, long long price, bool is_buy_side) {
        auto it = id_to_info_map.find(order_id);
        if (it != id_to_info_map.end()) {
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
        } else {
            id_to_info_map.emplace(order_id, AgentOrderInfo{price, is_buy_side});
        }
        price_to_ids_map[price].insert(order_id);
    }

    // Удалить запись по order_id. Если записи не было — ничего не делает.
    void remove(long long order_id) {
        auto it = id_to_info_map.find(order_id);
        if (it == id_to_info_map.end()) return;
        const long long old_price = it->second.price;
        auto pit = price_to_ids_map.find(old_price);
        if (pit != price_to_ids_map.end()) {
            pit->second.erase(order_id);
            if (pit->second.empty()) {
                price_to_ids_map.erase(pit);
            }
        }
        id_to_info_map.erase(it);
    }

    // Очистить все данные.
    void clear() {
        id_to_info_map.clear();
        price_to_ids_map.clear();
    }

    // Есть ли запись по ID.
    bool contains(long long order_id) const {
        return id_to_info_map.find(order_id) != id_to_info_map.end();
    }

    // Размер (кол-во отслеживаемых ордеров).
    std::size_t size() const {
        return id_to_info_map.size();
    }

    // Получить информацию по ID.
    const AgentOrderInfo* get_info(long long order_id) const {
        auto it = id_to_info_map.find(order_id);
        if (it == id_to_info_map.end()) return nullptr;
        return &it->second;
    }

    // Вернуть любую (первую попавшуюся) запись: удобно для отладки.
    // Возвращает nullptr, если трекер пуст.
    const std::pair<const long long, AgentOrderInfo>* get_first_order_info() const {
        if (id_to_info_map.empty()) return nullptr;
        return &(*id_to_info_map.begin());
    }

    bool is_empty() const {
        return id_to_info_map.empty();
    }

    std::vector<long long> get_all_ids() const {
        std::vector<long long> result;
        result.reserve(id_to_info_map.size());
        for (const auto &kv : id_to_info_map) {
            result.push_back(kv.first);
        }
        return result;
    }

    // Найти ближайшую по цене корзину и вернуть (order_id, price).
    // Детализация выбора:
    // 1) если есть точное совпадение уровня — берём минимальный order_id из этой корзины;
    // 2) иначе сравниваем ближайшие нижнюю и верхнюю корзины по |price - target|;
    //    при равенстве расстояний — предпочитаем НИЖНЮЮ цену; при полном равенстве — меньший order_id.
    std::pair<long long, long long> find_closest_order(long long target_price) const {
        if (price_to_ids_map.empty()) {
            return {-1, std::numeric_limits<long long>::min()};
        }

        auto it_exact = price_to_ids_map.find(target_price);
        if (it_exact != price_to_ids_map.end() && !it_exact->second.empty()) {
            return {static_cast<long long>(*it_exact->second.begin()), target_price};
        }

        std::optional<std::pair<long long, long long>> cand_ge;
        auto it_ge = price_to_ids_map.lower_bound(target_price);
        if (it_ge != price_to_ids_map.end()) {
            long long price = it_ge->first;
            if (!it_ge->second.empty()) {
                cand_ge = std::make_pair(price, static_cast<long long>(*it_ge->second.begin()));
            }
        }

        std::optional<std::pair<long long, long long>> cand_le;
        if (it_ge == price_to_ids_map.begin()) {
            // нет меньшего элемента
            if (it_ge == price_to_ids_map.end() && !price_to_ids_map.empty()) {
                auto last = std::prev(price_to_ids_map.end());
                if (!last->second.empty()) {
                    cand_le = std::make_pair(last->first, static_cast<long long>(*last->second.begin()));
                }
            }
        } else {
            auto it_prev = (it_ge == price_to_ids_map.end()) ? std::prev(price_to_ids_map.end()) : std::prev(it_ge);
            if (!it_prev->second.empty()) {
                cand_le = std::make_pair(it_prev->first, static_cast<long long>(*it_prev->second.begin()));
            }
        }

        if (cand_ge.has_value() && !cand_le.has_value()) {
            return {cand_ge->second, cand_ge->first};
        }
        if (cand_le.has_value() && !cand_ge.has_value()) {
            return {cand_le->second, cand_le->first};
        }
        if (!cand_ge.has_value() && !cand_le.has_value()) {
            return {-1, std::numeric_limits<long long>::min()};
        }

        unsigned long long d_ge = abs_diff(cand_ge->first, target_price);
        unsigned long long d_le = abs_diff(target_price, cand_le->first);

        if (d_ge < d_le) {
            return {cand_ge->second, cand_ge->first};
        }
        if (d_le < d_ge) {
            return {cand_le->second, cand_le->first};
        }

        if (cand_le->first < cand_ge->first) {
            return {cand_le->second, cand_le->first};
        }
        if (cand_ge->first < cand_le->first) {
            return {cand_ge->second, cand_ge->first};
        }

        return {std::min(cand_le->second, cand_ge->second), cand_le->first};
    }

private:
    std::map<long long, AgentOrderInfo> id_to_info_map;
    std::map<long long, std::set<long long>> price_to_ids_map;

    static unsigned long long abs_diff(long long lhs, long long rhs) {
#if defined(__SIZEOF_INT128__)
        __int128 diff = static_cast<__int128>(lhs) - static_cast<__int128>(rhs);
        if (diff < 0) diff = -diff;
        return static_cast<unsigned long long>(diff);
#else
        long long diff = lhs >= rhs ? lhs - rhs : rhs - lhs;
        if (diff < 0) {
            // Saturate on overflow (should be extremely rare for realistic price values)
            return std::numeric_limits<unsigned long long>::max();
        }
        return static_cast<unsigned long long>(diff);
#endif
    }
};

class CythonLOB:
    def __init__(self):
        self.next_id = 1
        self.orders = {}

    def add_limit_order(self, is_buy_side, price_ticks, volume, timestamp, taker_is_agent=True):
        # Simple crossing logic: if order crosses existing opposite order, fill it immediately
        opposite = [oid for oid, o in self.orders.items() if o['is_buy'] != bool(is_buy_side)]
        if is_buy_side and opposite:
            best_ask = min(opposite, key=lambda oid: self.orders[oid]['price'])
            if price_ticks >= self.orders[best_ask]['price']:
                del self.orders[best_ask]
                return best_ask, 0
        if not is_buy_side and opposite:
            best_bid = max(opposite, key=lambda oid: self.orders[oid]['price'])
            if price_ticks <= self.orders[best_bid]['price']:
                del self.orders[best_bid]
                return best_bid, 0
        oid = self.next_id
        self.next_id += 1
        self.orders[oid] = {'is_buy': bool(is_buy_side), 'price': int(price_ticks), 'volume': float(volume), 'ttl': 0}
        return oid, 0

    def add_limit_order_with_id(self, is_buy_side, price_ticks, volume, order_id, timestamp, taker_is_agent=True):
        if order_id is not None and order_id != 0:
            oid = int(order_id)
            if oid >= self.next_id:
                self.next_id = oid + 1
        else:
            oid = self.next_id
            self.next_id += 1
        self.orders[oid] = {'is_buy': bool(is_buy_side), 'price': int(price_ticks), 'volume': float(volume), 'ttl': 0}
        return oid, 0

    def remove_order(self, is_buy_side, price_ticks, order_id):
        return self.orders.pop(order_id, None) is not None

    def contains_order(self, order_id):
        return order_id in self.orders

    def set_order_ttl(self, order_id, ttl_steps):
        if order_id in self.orders:
            self.orders[order_id]['ttl'] = int(ttl_steps)
            return True
        return False

    def decay_ttl_and_cancel(self):
        cancelled = []
        for oid in list(self.orders):
            ttl = self.orders[oid]['ttl']
            if ttl > 0:
                ttl -= 1
                if ttl <= 0:
                    cancelled.append(oid)
                    del self.orders[oid]
                else:
                    self.orders[oid]['ttl'] = ttl
        return cancelled

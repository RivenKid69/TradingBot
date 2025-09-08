# Архитектура проекта

В репозитории используется слойная структура. Имена файлов и модулей начинаются с префиксов, отражающих их принадлежность к слою.

## Слои

- `core_`: базовые сущности, контракты и модели. Не зависит от других слоёв.
- `impl_`: конкретные реализации инфраструктуры и внешних зависимостей. Допустима зависимость только от `core_`.
- `service_`: сервисы, объединяющие реализацию и бизнес‑логику. Может зависеть от `core_` и `impl_`.
- `strategies`: торговые стратегии и алгоритмы. Допускаются зависимости от всех предыдущих слоёв (`service_`, `impl_`, `core_`).
- `scripts_`: запускаемые скрипты и утилиты. Могут использовать код из любых слоёв.

Допустимые направления зависимостей идут снизу вверх:

```
core_ → impl_ → service_ → strategies → scripts_
```

Каждый слой может зависеть только от слоёв, расположенных левее.

Общий план развития проекта приведён в файле [План.txt](План.txt).

## Слой strategies

В пакете `strategies` располагаются торговые алгоритмы. Они могут
использовать код из слоёв `core_`, `impl_` и `service_`, но не должны
зависеть от других стратегий или скриптов.

Стратегии реализуют протокол [`Strategy`](core_strategy.py) и обычно
наследуются от `BaseStrategy`. Сервисы (`service_*`) получают стратегию
через DI-контейнер (`di_registry`) и взаимодействуют с ней только через
интерфейс `Strategy`.

### Decision

Решение стратегии описывается датаклассом `Decision` со следующими
полями:

- `side` — "BUY" или "SELL";
- `volume_frac` — целевая величина заявки в долях позиции (диапазон
  `[-1.0; 1.0]`);
- `price_offset_ticks` — смещение цены в тиках для лимитных заявок
  (для рыночных равно `0`);
- `tif` — срок действия заявки (`GTC`, `IOC` или `FOK`);
- `client_tag` — опциональная строка для пометки действий.

### Пример модуля

```python
# strategies/momentum.py
from core_strategy import Strategy, Decision

class MomentumStrategy(Strategy):
    def decide(self, ctx: dict) -> list[Decision]:
        if ctx["ref_price"] > ctx["features"]["ma"]:
            return [Decision(side="BUY", volume_frac=0.1)]
        return []
```

## Конфигурации запусков

Конфигурации описываются в формате YAML. Для загрузки и валидации
используйте функцию `load_config`:

```yaml
# configs/run_sim.yaml
mode: sim
components:
  market_data:
    target: impl_offline_data:OfflineCSVBarSource
    params: {paths: ["data/sample.csv"], timeframe: "1m"}
  executor:
    target: impl_sim_executor:SimExecutor
    params: {symbol: "BTCUSDT"}
data:
  symbols: ["BTCUSDT"]
  timeframe: "1m"
```

```python
from core_config import load_config

cfg = load_config("configs/run_sim.yaml")
```

### CLI-скрипты

Несколько вспомогательных скриптов принимают путь к YAML через
флаг `--config` и запускают соответствующие сервисы через `from_config`:

```
python run_realtime_signaler.py --config configs/config_live.yaml
python run_sandbox.py          --config configs/config_sim.yaml
python evaluate_performance.py --config configs/config_eval.yaml
```

## ServiceTrain

`ServiceTrain` подготавливает датасет и запускает обучение модели.  Он
ожидает реализацию протокола `FeaturePipe`.  Для оффлайн‑расчёта фич
можно использовать класс `OfflineFeaturePipe`, который оборачивает
функцию `apply_offline_features`.

Пример запуска обучения:

```python
from core_config import CommonRunConfig
from service_train import from_config, TrainConfig

cfg_run = CommonRunConfig(...)
trainer = ...
cfg = TrainConfig(input_path="data/train.parquet")
from_config(cfg_run, trainer=trainer, train_cfg=cfg)
```
=======
## Проверка паритета фич

Для валидации соответствия оффлайн и онлайнового расчёта признаков используйте скрипт `check_feature_parity.py`.

Пример запуска:

```
python check_feature_parity.py --data path/to/prices.csv --threshold 1e-6
```

Скрипт вычисляет признаки обоими способами и сообщает о строках, где абсолютное различие превышает `--threshold`. При отсутствии расхождений выводится подтверждение паритета.


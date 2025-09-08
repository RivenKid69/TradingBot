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
# configs/config_sim.yaml
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

cfg = load_config("configs/config_sim.yaml")
```

### CLI-скрипты

Несколько вспомогательных скриптов принимают путь к YAML через
флаг `--config` и запускают соответствующие сервисы через `from_config`:

```
python script_live.py    --config configs/config_live.yaml
python script_backtest.py --config configs/config_sim.yaml
python script_eval.py    --config configs/config_eval.yaml
```

### Сравнение запусков

Для агрегирования результатов нескольких прогонов используйте скрипт
`script_compare_runs.py`. Он принимает список путей к файлам
`metrics.json` или каталогам запусков и формирует таблицу ключевых
метрик:

```bash
python script_compare_runs.py run1/ run2/metrics.json --csv summary.csv
```

В консоль выводятся значения `run_id`, `Sharpe`, `Sortino`, `MDD`, `PnL`,
`Hit-rate`, `CVaR` и других найденных показателей. При указании флага
`--csv` таблица сохраняется в указанный файл.

## CLI‑точки входа

Все консольные скрипты используют DI‑контейнер и не содержат бизнес‑логики. Они
описывают аргументы командной строки и делегируют работу соответствующим
сервисам:

- `script_train.py` — запускает обучение через `ServiceTrain`.
- `script_backtest.py` — проводит бэктест через `ServiceBacktest`.
- `script_eval.py` — рассчитывает метрики через `ServiceEval`.
- `script_live.py` — исполняет стратегию на живых данных через `ServiceSignalRunner`.
- `script_calibrate_tcost.py` — калибрует параметры T‑cost через `ServiceCalibrateTCost`.
- `script_calibrate_slippage.py` — калибрует проскальзывание через `ServiceCalibrateSlippage`.
- `script_compare_runs.py` — агрегирует метрики нескольких запусков.

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
## Логи и отчёты

Сервисы автоматически пишут журналы сделок и отчёты по эквити через
класс `LogWriter` из модуля [`logging.py`](logging.py). По умолчанию
создаются два файла.

### `logs/log_trades_<runid>.csv`

- Каждая строка соответствует датаклассу
  [`TradeLogRow`](core_models.py).
- Обязательные колонки: `ts`, `run_id`, `symbol`, `side`, `order_type`,
  `price`, `quantity`, `fee`, `fee_asset`, `exec_status`, `liquidity`,
  `client_order_id`, `order_id`, `trade_id`, `pnl`, а также добавленные
  `mark_price` и `equity`.
- Пример строки:

```csv
1700000000000,sim,BTCUSDT,BUY,LIMIT,30000,0.01,0.0005,USDT,FILLED,TAKER,c1,o1,t1,15.0,30010,1005.0,{}
```

### `logs/report_equity_<runid>.csv`

- Строки соответствуют [`EquityPoint`](core_models.py).
- Обязательные колонки: `ts`, `run_id`, `symbol`, `fee_total`,
  `position_qty`, `realized_pnl`, `unrealized_pnl`, `equity`,
  `mark_price`, `drawdown`, `risk_paused_until_ms`, `risk_events_count`,
  `funding_events_count`, `cash`, `meta`.
- Пример строки:

```csv
1700000000000,sim,BTCUSDT,1.2,0.05,100.0,5.0,105.0,30050,-0.02,0,0,0,,{}
```

Логи формируются и обновляются автоматически во всех сервисах
(`service_*`, `execution_sim`) и могут сохраняться как в CSV, так и в
формате Parquet.
=======
## Проверка паритета фич

Для валидации соответствия оффлайн и онлайнового расчёта признаков используйте скрипт `check_feature_parity.py`.

Пример запуска:

```
python check_feature_parity.py --data path/to/prices.csv --threshold 1e-6
```

Скрипт вычисляет признаки обоими способами и сообщает о строках, где абсолютное различие превышает `--threshold`. При отсутствии расхождений выводится подтверждение паритета.


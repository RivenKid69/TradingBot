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
  timeframe: "1m"
```

```python
from core_config import load_config

cfg = load_config("configs/config_sim.yaml")
```

The runner loads the symbol universe from ``data/universe/symbols.json`` by default.
Override it via the ``--symbols`` CLI flag or an explicit ``data.symbols`` field.

Отдельные параметры можно переопределить из командной строки. Например,
так временно изменяются проскальзывание и задержка:

```bash
python train_model_multi_patch.py --config configs/config_train.yaml --slippage.bps 5 --latency.mean_ms 50
```

Те же значения можно указать напрямую в YAML:

```yaml
slippage:
  bps: 5
latency:
  mean_ms: 50
```

### Сохранение состояния

Параметры сохранения промежуточного состояния находятся в файле
`configs/state.yaml`. Он задаёт расположение, тип хранилища и периодичность
создания снапшотов:

```yaml
enabled: false
backend: json
dir: state
path: state/state_store.json
snapshot_interval_s: 60
# snapshot_interval_ms: null
flush_on_event: true
backup_keep: 3
lock_path: state/state.lock
last_processed_per_symbol: false
```

* `enabled` — включить сохранение состояния.
* `backend` — тип хранилища (`json` или `sqlite`).
* `dir` — каталог, в котором будут храниться файлы состояния (создаётся автоматически).
* `path` — путь к основному файлу с состоянием.
* `snapshot_interval_s` / `snapshot_interval_ms` — периодичность автосохранения.
* `flush_on_event` — писать состояние при принудительном сбросе.
* `backup_keep` — количество резервных копий.
* `lock_path` — путь к файлу блокировки.
* `last_processed_per_symbol` — сохранять прогресс по каждому инструменту, если доступно.

### Профили исполнения

Конфигурация может содержать несколько профилей исполнения. Каждый профиль
описан параметрами `slippage_bps`, `offset_bps`, `ttl` (в мс) и `tif`, которые
определяют цену выставления заявки и её поведение во времени.

| Профиль       | `slippage_bps` | `offset_bps` | `ttl`, мс | `tif` | Поведение |
|---------------|----------------|--------------|-----------|-------|-----------|
| `conservative`| 5              | 2            | 5000      | GTC   | Пассивные лимитные заявки, ожидание исполнения |
| `balanced`    | 3              | 0            | 2000      | GTC   | Заявки около середины книги, умеренное ожидание |
| `aggressive`  | 1              | -1           | 500       | IOC   | Кроссует спред и быстро отменяет невыполненные заявки |

Пример описания профилей в YAML:

```yaml
profile: balanced
profiles:
  conservative:
    slippage_bps: 5
    offset_bps: 2
    ttl: 5000
    tif: GTC
  balanced:
    slippage_bps: 3
    offset_bps: 0
    ttl: 2000
    tif: GTC
  aggressive:
    slippage_bps: 1
    offset_bps: -1
    ttl: 500
    tif: IOC
```

Скрипт `script_eval.py` позволяет выбрать профиль через `--profile` или
запустить оценку всех профилей флагом `--all-profiles`. В последнем случае
`ServiceEval` формирует отдельные наборы метрик и отчётов для каждого
профиля. Значения `Sharpe`, `PnL` и другие показатели следует анализировать
по каждому профилю отдельно и сравнивать между ними.

### CLI-скрипты

Несколько вспомогательных скриптов принимают путь к YAML через
флаг `--config` и запускают соответствующие сервисы через `from_config`:

```
python train_model_multi_patch.py --config configs/config_train.yaml
python script_live.py    --config configs/config_live.yaml
python script_backtest.py --config configs/config_sim.yaml
python script_eval.py    --config configs/config_eval.yaml --profile vwap
python script_eval.py    --config configs/config_eval.yaml --all-profiles
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

- `train_model_multi_patch.py` — запускает обучение через `ServiceTrain`.
- `script_backtest.py` — проводит бэктест через `ServiceBacktest`.
- `script_eval.py` — рассчитывает метрики через `ServiceEval` (поддерживает `--profile` и `--all-profiles`).
- `script_live.py` — исполняет стратегию на живых данных через `ServiceSignalRunner`.
- `script_calibrate_tcost.py` — калибрует параметры T‑cost через `ServiceCalibrateTCost`.
- `script_calibrate_slippage.py` — калибрует проскальзывание через `ServiceCalibrateSlippage`.
- `script_compare_runs.py` — агрегирует метрики нескольких запусков.

## ServiceTrain

`ServiceTrain` подготавливает датасет и запускает обучение модели.  Он
ожидает реализацию протокола `FeaturePipe`.  Для оффлайн‑расчёта фич
используется тот же класс `FeaturePipe`, оборачивающий функцию
`apply_offline_features`.

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
класс `LogWriter` из модуля [`sim_logging.py`](sim_logging.py). По умолчанию
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

## Сезонность ликвидности, спреда и задержек

Сезонность моделирует систематические изменения параметров в течение 168
часов недели (0 = понедельник 00:00 UTC). Генерация коэффициентов
происходит скриптом `scripts/build_hourly_seasonality.py`, который
рассчитывает средние значения по историческим данным и сохраняет их в
`data/latency/liquidity_latency_seasonality.json`.

Функции `load_seasonality` и `load_hourly_seasonality` из `utils_time.py`
читают JSON, проверяют контрольную сумму и ограничивают коэффициенты.
`TradingPatch` и `MarketSimulator` масштабируют ликвидность и спред, а
`LatencyImpl` применяет коэффициенты к задержке; опция
`seasonality_interpolate` включает сглаживание между часами.

Для проверки и визуализации доступны скрипты
`scripts/validate_seasonality.py` и `scripts/plot_seasonality.py`. Все
временные метки должны быть в UTC во избежание ошибок индексации по
часам недели.
=======
## Проверка паритета фич

Для валидации соответствия оффлайн и онлайнового расчёта признаков используйте скрипт `check_feature_parity.py`.

Пример запуска:

```
python check_feature_parity.py --data path/to/prices.csv --threshold 1e-6
```

Скрипт вычисляет признаки обоими способами и сообщает о строках, где абсолютное различие превышает `--threshold`. При отсутствии расхождений выводится подтверждение паритета.


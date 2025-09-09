# TradingBot

Скрипты `script_*.py` выступают CLI‑точками входа в сервисы. Все они
используют dependency injection и не содержат бизнес‑логики, ограничиваясь
описанием аргументов и вызовом соответствующих сервисов.

## Примеры запуска

Сравнить результаты нескольких запусков можно с помощью
`script_compare_runs.py`. Передайте ему пути к каталогам или файлам
`metrics.json`. По умолчанию таблица будет сохранена в
`compare_runs.csv`, а флаг `--stdout` выведет её в консоль.

```bash
python script_backtest.py --config configs/config_sim.yaml
python script_train.py --config configs/train.yaml --trainer-module mypackage.trainer:MyTrainer --no-trade-mode weight
python script_compare_runs.py run1 run2 run3            # сохранит compare_runs.csv
python script_compare_runs.py run1 metrics.json --stdout  # вывод в stdout
python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json
```

Флаг `--no-trade-mode` управляет применением **no‑trade** окон из конфигурации:
`drop` удаляет такие строки из датасета, а `weight` оставляет их с
добавленным столбцом `train_weight=0.0`.

### no-trade-mask утилита

Для предварительной фильтрации датасетов можно воспользоваться
консольным скриптом `no-trade-mask` (устанавливается вместе с пакетами
через `setup.py/pyproject.toml`). Он принимает путь к входным данным и
конфигурации с описанием окон `no_trade` и поддерживает два режима:

```bash
# удалить запрещённые интервалы
no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode drop

# пометить строки train_weight=0.0, оставив их в датасете
no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode weight
```

Загрузка настроек `no_trade` централизована: функция
`no_trade_config.get_no_trade_config()` считывает секцию `no_trade` из YAML‑файла
и возвращает модель `NoTradeConfig`. Все модули используют её как единый
источник правды, исключая расхождения в трактовке конфигурации.

## Профили исполнения

В конфигурации можно описать несколько профилей исполнения. Каждый профиль
задаёт параметры симуляции и ожидаемое поведение выставляемых ордеров.

| Профиль       | `slippage_bps` | `offset_bps` | `ttl`, мс | `tif` | Поведение |
|---------------|----------------|--------------|-----------|-------|-----------|
| `conservative`| 5              | 2            | 5000      | GTC   | Пассивные лимитные заявки, ожидание исполнения |
| `balanced`    | 3              | 0            | 2000      | GTC   | Заявки около середины книги, умеренное ожидание |
| `aggressive`  | 1              | -1           | 500       | IOC   | Кроссует спред и быстро отменяет невыполненные заявки |

Пример YAML‑конфига с переключением профиля:

```yaml
profile: balanced  # используется по умолчанию
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

Скрипт `script_eval.py` позволяет выбрать конкретный профиль или
оценить все сразу:

```bash
python script_eval.py --config configs/config_eval.yaml --profile aggressive
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

При мульти‑профильной оценке метрики (`Sharpe`, `PnL` и т.д.)
сохраняются отдельно для каждого профиля (`metrics_conservative.json`,
`metrics_balanced.json`, ...). Их следует интерпретировать как результаты
при соответствующих предположениях исполнения и сравнивать между
профилями.


## Проверка кривой проскальзывания

Скрипт `compare_slippage_curve.py` строит кривые `slippage_bps` по квантилям
размера ордера для исторических и симуляционных сделок и сравнивает их.
Если отклонение по каждой точке превышает допустимый порог, выполнение
заканчивается кодом ошибки.

```bash
python compare_slippage_curve.py hist.csv sim.csv --tolerance 5
```

Критерий акцептанса: абсолютное различие между средним `slippage_bps`
в соответствующих квантилях не должно превышать указанного порога в bps.

## Проверка PnL симулятора

Скрипт `tests/pnl_report_check.py` прогоняет серию сделок через
`ExecutionSimulator` и сверяет сумму `realized_pnl + unrealized_pnl` из отчёта
с пересчитанным значением по трейдам и полям `bid/ask/mtm_price`.

```bash
python tests/pnl_report_check.py
```

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
python script_train.py --config configs/train.yaml --trainer-module mypackage.trainer:MyTrainer
python script_compare_runs.py run1 run2 run3            # сохранит compare_runs.csv
python script_compare_runs.py run1 metrics.json --stdout  # вывод в stdout
python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json
```


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

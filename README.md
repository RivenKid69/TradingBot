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
```


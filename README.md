# TradingBot

Скрипты `script_*.py` и `train_model_multi_patch.py` выступают CLI‑точками
входа в сервисы. Все они используют dependency injection и не содержат
бизнес‑логики, ограничиваясь описанием аргументов и вызовом соответствующих
сервисов.

## Примеры запуска

Сравнить результаты нескольких запусков можно с помощью
`script_compare_runs.py`. Передайте ему пути к каталогам или файлам
`metrics.json`. По умолчанию таблица будет сохранена в
`compare_runs.csv`, а флаг `--stdout` выведет её в консоль.

```bash
python script_backtest.py --config configs/config_sim.yaml
python train_model_multi_patch.py --config configs/config_train.yaml \
  --regime-config configs/market_regimes.json \
  --liquidity-seasonality configs/liquidity_seasonality.json
python script_compare_runs.py run1 run2 run3            # сохранит compare_runs.csv
python script_compare_runs.py run1 metrics.json --stdout  # вывод в stdout
python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json
```

Параметры симуляции можно временно переопределить через CLI:

```bash
python train_model_multi_patch.py --config configs/config_train.yaml --slippage.bps 5 --latency.mean_ms 50
```

Дополнительно доступны опции `--regime-config` и `--liquidity-seasonality`,
позволяющие указать пути к откалиброванным JSON‑файлам с параметрами
рыночных режимов и сезонностью ликвидности соответственно. По умолчанию
используются файлы из каталога `configs/`.

Те же значения можно задать в YAML‑конфиге:

```yaml
slippage:
  bps: 5
latency:
  mean_ms: 50
```

Обработка окон **no‑trade** описывается в конфигурации; подробности
см. [docs/no_trade.md](docs/no_trade.md).

Параллельные окружения и контроль случайности описаны в
[docs/parallel.md](docs/parallel.md).

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

После выполнения утилита выводит процент заблокированных строк и сводку
`NoTradeConfig`. При указании `--histogram` дополнительно печатается
гистограмма длительностей блоков:

```bash
$ no-trade-mask --data data.csv --sandbox_config configs/legacy_sandbox.yaml --mode drop --histogram
Готово. Всего строк: 3. Запрещённых (no_trade): 2 (66.67%). Вышло: 1.
NoTradeConfig: {'funding_buffer_min': 5, 'daily_utc': ['00:00-00:05', '08:00-08:05', '16:00-16:05'], 'custom_ms': []}
Гистограмма длительностей блоков (минуты):
-0.5-0.5: 2
```

Загрузка настроек `no_trade` централизована: функция
`no_trade_config.get_no_trade_config()` считывает секцию `no_trade` из YAML‑файла
и возвращает модель `NoTradeConfig`. Все модули используют её как единый
источник правды, исключая расхождения в трактовке конфигурации.
Подробнее о полях конфигурации и сценариях использования см. [docs/no_trade.md](docs/no_trade.md).

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

`ExecutionSimulator` исполняет сделки по лучшим котировкам:
ордер `BUY` заполняется по цене `ask`, а ордер `SELL` — по `bid`.
Незакрытые позиции помечаются по рынку (mark‑to‑market) также
по лучшим котировкам: для длинной позиции используется `bid`,
для короткой — `ask`. Если в отчёте присутствует поле `mtm_price`,
оно переопределяет цену маркировки.

В отчётах симуляции присутствуют поля:

* `bid` и `ask` — текущие лучшие котировки;
* `mtm_price` — фактическая цена для mark‑to‑market
  (может отсутствовать/быть `0`, тогда используется `bid/ask`).

Проверочный скрипт пересчитывает `realized_pnl + unrealized_pnl`
по логу трейдов и указанным ценам. Пример пересчёта:

```python
from tests.test_pnl_report_check import _recompute_total

trades = [
    {"side": "BUY", "price": 101.0, "qty": 1.0},
    {"side": "SELL", "price": 102.0, "qty": 1.0},
]
total = _recompute_total(trades, bid=102.0, ask=103.0, mtm_price=None)
# total == 1.0 (realized_pnl + unrealized_pnl)
```

Регрессионный тест `tests/test_pnl_report_check.py` запускает
симулятор и сравнивает отчёт с пересчитанным результатом.
Выполнить его можно командой:

```bash
pytest tests/test_pnl_report_check.py
```

## Проверка реалистичности симуляции

`scripts/sim_reality_check.py` сопоставляет метрики симуляции с
историческими данными и эталонной кривой капитала. Скрипт принимает пути к
логу сделок симуляции (`--trades`), историческому логу (`--historical-trades`),
опциональному файлу капитальной кривой (`--equity`), бенчмарку (`--benchmark`) и
JSON‑файлу с допустимыми диапазонами KPI (`--kpi-thresholds`). Параметр
`--quantiles` задаёт число квантилей для построения статистики по размерам
ордеров.

При запуске формируются отчёты `sim_reality_check.json` и
`sim_reality_check.md`, а также файл `sim_reality_check_buckets.csv` и график
среднего `spread/slippage` по квантилям. Если значения KPI выходят за
пороговые диапазоны, список нарушений выводится в консоль и попадает в отчёт.

```bash
# все KPI в пределах порогов
python scripts/sim_reality_check.py \
  --trades sim_trades.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet
Saved reports to run/sim_reality_check.json and run/sim_reality_check.md
Saved bucket stats to run/sim_reality_check_buckets.csv and run/sim_reality_check_buckets.png

# пример с нарушением порогов
python scripts/sim_reality_check.py \
  --trades sim_bad.parquet \
  --historical-trades hist_trades.parquet \
  --equity sim_equity.parquet \
  --benchmark bench_equity.parquet \
  --kpi-thresholds benchmarks/sim_kpi_thresholds.json
Saved reports to run/sim_reality_check.json and run/sim_reality_check.md
Saved bucket stats to run/sim_reality_check_buckets.csv and run/sim_reality_check_buckets.png
Unrealistic KPIs detected:
 - equity.sharpe: нереалистично
```

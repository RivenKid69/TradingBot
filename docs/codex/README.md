# Codex Project Handbook

Эта папка предназначена для быстрой ориентации модели Codex в репозитории
`TradingBot`. Здесь собраны основные сведения о структуре проекта, ключевых
скриптах, конфигурации и типичных рабочих командах.

## 1. Обзор проекта
- **Домен**: алгоритмическая торговля, симуляция и исполнение стратегий для
  Binance (spot/futures) с поддержкой барового режима.
- **Языки**: Python (основная логика и CLI), C++/Cython (симулятор, LOB и
  высокопроизводительные компоненты).
- **Архитектура**: CLI-скрипты `script_*.py` и `train_model_multi_patch.py`
  выступают точками входа и передают управление сервисам через dependency
  injection; бизнес-логика вынесена в модули `services/`, `execution_*`,
  `impl_*` и др.

- **Выполненные задачи**: большинство скриптов принимают YAML-конфиги и
  используют `core_config` для валидации входных параметров, что упрощает
  повторное использование настроек между оффлайн- и онлайн-режимами.

### 1.1 Типичный рабочий цикл Codex
1. **Чтение задачи** → зафиксировать целевые файлы и команды.
2. **Поиск контекста** → пройтись по `docs/`, `configs/`, исходникам с помощью `rg`.
3. **Черновая проверка** → запустить локальный скрипт/тест, если требуется.
4. **Внесение изменений** → соблюдать длину строк ≤ 200 символов (flake8) и не
   добавлять глобальные побочные эффекты.
5. **Тесты и форматирование** → `make format` / таргетные команды.
6. **Документация** → обновить эту памятку либо профильные файлы при
   добавлении нового пайплайна/конфига.



## 2. Ключевые директории
- `configs/` — YAML-конфигурации для симуляции, трейдинга и обучения.
- `data/` — наборы данных и кеши (например, universe символов, seasonality,
  биржевые спецификации).
- `docs/` — тематическая документация (pipeline, bar execution, universe,
  moving average, permissions и др.).
- `services/` — реализация сервисов (бек-тест, тренировка, получение
  спецификаций, signal runner и т.п.).
- `execution_*`, `impl_*`, `fast_*`, `risk_*` — специализированные модули
  исполнения ордеров, оценки рисков, работы с латентностью.
- `tests/` — выборочные сценарии тестирования и примеры (например,
  `run_no_trade_mask_sample.py`).

- `tools/` — вспомогательные скрипты для подготовки данных и интеграции с
  внешними сервисами.
- `notebooks/` — исследовательские ноутбуки; при переносе логики в продовый
  код добавляйте ссылки на notebook-источники.

### 2.1 Быстрая карта зависимостей
- Модули `execution_*` зависят от Cython-оберток (`exec*.pyx`, `execlob_book.*`).
- `services/` импортируют вспомогательные функции из `impl_*` и `core_*`.
- `pipeline.py` и `feature_pipe.py` связывают генерацию фич с сервисами.
- `reward.pyx`, `risk_manager.pyx` подключают типы из `risk_enums.pxd`.
- Подробности специфичных пайплайнов смотрите в `docs/pipeline.md` и
  комментариях рядом с соответствующими сервисами.


## 3. Установка зависимостей
Дополнительные зависимости для скриптов данных и моделей:
```bash
pip install -r requirements_extra.txt
# либо через extras
pip install ".[extra]"
```


### 3.1 Сборка Cython и C++
После изменения `.pyx`/`.pxd`/`.cpp` файлов:
```bash
pip install -e .  # пересоберёт расширения
```
Убедитесь в наличии `build-essential`, `python3-dev`, а также корректной
версии `cython` (см. `pyproject.toml`). При локальной разработке полезно
очищать кеши `build/` и `*.c` при смене версий компилятора.

### 3.2 Работа с виртуальными окружениями
- Создайте окружение: `python -m venv .venv && source .venv/bin/activate`.
- Базовые зависимости ставятся через `pip install -e .`.
- Дополнительные пакеты (analyzer, notebooks) — через `pip install -r requirements_extra.txt`.
- При подготовке релизных окружений фиксируйте версии в `requirements_lock.txt`
  (создать при необходимости `pip freeze > requirements_lock.txt`).


## 4. Частые команды
- Бэктест с симуляцией: `python script_backtest.py --config configs/config_sim.yaml`
- Тренировка модели: `python train_model_multi_patch.py --config configs/config_train.yaml --regime-config configs/market_regimes.json --liquidity-seasonality data/latency/liquidity_latency_seasonality.json`
- Сравнение запусков: `python script_compare_runs.py run1 run2 run3`
  (по умолчанию создаст `compare_runs.csv`; добавьте `--stdout` для вывода в консоль).
- Получение биржевых спецификаций: `python script_fetch_exchange_specs.py --market futures --symbols BTCUSDT,ETHUSDT --out data/exchange_specs.json`
- Проверка seasonality: `python scripts/validate_seasonality.py --historical path/to/trades.csv --multipliers data/latency/liquidity_latency_seasonality.json`


### 4.1 Диагностика и профилирование
- Проверка целостности данных: `python data_validation.py --config configs/data_validation.yaml`.
- Профилирование latency: `python latency_volatility_cache.py --config configs/latency_cache.yaml`.
- Отладка сигналов без публикации сделок: `python service_signal_runner.py --config configs/config_live.yaml --dry-run`.
- Агрегация логов исполнения: `python aggregate_exec_logs.py --input reports/latest --out reports/summary.csv`.


## 5. Управление списком символов
Сервисы читают юниверс символов из `data/universe/symbols.json`. Обновление
и проверка:
```bash
python -m services.universe --output data/universe/symbols.json --liquidity-threshold 1e6

python - <<'PY'
import json, os, time
path = "data/universe/symbols.json"
print("age_s", round(time.time() - os.path.getmtime(path), 1))
with open(path, "r", encoding="utf-8") as fh:
    symbols = json.load(fh)
print("first", symbols[:5])
print("count", len(symbols))
PY
```
Конфигурации используют `core_config.get_symbols`; для проверки загрузки из
конфига:
```bash
python - <<'PY'
from core_config import load_config
cfg = load_config("configs/config_live.yaml")
print("runner_symbols", cfg.data.symbols[:5])
PY
```

## 6. Интрабарная конфигурация исполнения
В `configs/config_sim.yaml` блок `execution` управляет выбором модели
интрабарного исполнения:
- `intrabar_price_model`: `bridge` (Brownian bridge) или `reference` (детерминированная M1 ссылка).
- `timeframe_ms`: длительность бара.
- `reference_prices_path`: путь к M1-данным для режима `reference`.
Также предусмотрен вложенный блок `execution.bridge` для bridge-адаптера.

## 7. Баровый режим
Для включения барового режима используйте шаблон `configs/runtime_trade.yaml`
или CLI-переключатели:
```yaml
portfolio:
  equity_usd: 1_000_000.0
costs:
  taker_fee_bps: 7.5
  half_spread_bps: 1.5
  impact:
    sqrt_coeff: 15.0
    linear_coeff: 2.5
execution:
  mode: bar
  bar_price: close
  min_rebalance_step: 0.05
```
Соответствующий сигнал должен удовлетворять контракту envelope в
`docs/bar_execution.md` (поля `edge_bps`, `cost_bps`, `net_bps`, `turnover_usd`, `act_now`, `execution.safety_margin_bps`).

## 8. Фильтры и спецификации биржи
Файлы `binance_filters.json` и `exchange_specs.json` содержат раздел
`metadata` с диагностикой (`built_at`, `source`, `symbols_count`, `generated_at`,
`source_dataset`, `version`). Обновляйте их через соответствующие скрипты из
корня репозитория.

## 9. Форматирование и проверки
Доступные команды Makefile:
```bash
make format  # black .
make lint    # flake8 с max line length = 200 на ключевых модулях
make no-trade-mask-sample  # python tests/run_no_trade_mask_sample.py
```
При необходимости запускайте специфичные тестовые/скриптовые файлы напрямую.

## 10. Дополнительные материалы
Изучите документацию в `docs/`:
- `docs/pipeline.md` — обзор пайплайна принятия решений.
- `docs/bar_execution.md` — формат сигналов и баровое исполнение.
- `docs/moving_average.md` — утилита скользящего среднего.
- `docs/universe.md` — детали обновления списка символов.
- `docs/permissions.md` — владение файлами и роли.

Эта памятка предназначена для быстрого старта и контекстуализации задач.
При внесении значительных изменений обновляйте соответствующие разделы.


## 11. Частые сценарии и подсказки
- **Добавление нового сервиса**: создайте модуль в `services/`, подключите его в
  CLI (`script_*.py` или `service_*.py`), опишите конфиг в `configs/` и
  обновите документацию.
- **Работа с событиями**: типы событий определены в `core_events.py`; для
  подписки/публикации используйте `EventBus` из `event_bus.py`.
- **Ошибки исполнения**: изучите `core_errors.py` и `risk_guard.py` для
  типовых исключений и стратегий защиты.
- **Логирование**: модуль `sim_logging.py` и конфиги logging управляют
  выводом; ротация файлов настраивается через `logging.handlers`.
- **Расширение набора признаков**: добавляйте фичи в `features_pipeline.py`
  и конкретные реализации в `features/` или `featuresbasic.py`.
- **Соглашения по тестам**: smoke-сценарии хранятся в `tests/`; используйте
  вспомогательные состояния из `state/` для детерминированных проверок.

## 12. Чек-лист перед коммитом
- [ ] Изменения покрыты соответствующими тестами или прогонены скрипты.
- [ ] Запущены `make format` и `make lint` (или эквивалентные команды).
- [ ] Обновлена документация (`docs/` или эта памятка) при изменении публичных API/конфигов.
- [ ] Проверены пути к данным и права доступа (см. `docs/permissions.md`).
- [ ] Подготовлено краткое и информативное описание для PR.


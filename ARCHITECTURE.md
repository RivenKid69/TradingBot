# Архитектура проекта

В репозитории используется слойная структура. Имена файлов и модулей начинаются с префиксов, отражающих их принадлежность к слою.

## Слои

- `core_`: базовые сущности, контракты и модели. Не зависит от других слоёв.
- `impl_`: конкретные реализации инфраструктуры и внешних зависимостей. Допустима зависимость только от `core_`.
- `service_`: сервисы, объединяющие реализацию и бизнес‑логику. Может зависеть от `core_` и `impl_`.
- `strategy_`: торговые стратегии и алгоритмы. Допускаются зависимости от всех предыдущих слоёв (`service_`, `impl_`, `core_`).
- `scripts_`: запускаемые скрипты и утилиты. Могут использовать код из любых слоёв.

Допустимые направления зависимостей идут снизу вверх:

```
core_ → impl_ → service_ → strategy_ → scripts_
```

Каждый слой может зависеть только от слоёв, расположенных левее.

Общий план развития проекта приведён в файле [План.txt](План.txt).


## ServiceTrain

`ServiceTrain` подготавливает датасет и запускает обучение модели.  Он
ожидает реализацию протокола `FeaturePipe`.  Для оффлайн‑расчёта фич
можно использовать класс `OfflineFeaturePipe`, который оборачивает
функцию `apply_offline_features`.

Пример запуска обучения:

```python
from transformers import FeatureSpec
from offline_feature_pipe import OfflineFeaturePipe
from service_train import ServiceTrain, TrainConfig

spec = FeatureSpec(lookbacks_prices=[5, 15, 60], rsi_period=14)
fp = OfflineFeaturePipe(spec, price_col="ref_price")
trainer = ...
cfg = TrainConfig(input_path="data/train.parquet")
ServiceTrain(fp, trainer, cfg).run()
```
=======
## Проверка паритета фич

Для валидации соответствия оффлайн и онлайнового расчёта признаков используйте скрипт `check_feature_parity.py`.

Пример запуска:

```
python check_feature_parity.py --data path/to/prices.csv --threshold 1e-6
```

Скрипт вычисляет признаки обоими способами и сообщает о строках, где абсолютное различие превышает `--threshold`. При отсутствии расхождений выводится подтверждение паритета.


# TradingBot

Скрипты `script_*.py` выступают CLI‑точками входа в сервисы. Все они
используют dependency injection и не содержат бизнес‑логики, ограничиваясь
описанием аргументов и вызовом соответствующих сервисов.

## Примеры запуска

```bash
python script_backtest.py --config configs/config_sim.yaml
python script_train.py --config configs/train.yaml --trainer-module mypackage.trainer:MyTrainer
```


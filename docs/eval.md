# Оценка стратегии

Сервис `service_eval` позволяет вычислять метрики эффективности на основе логов
трейдов и кривой капитала. Для сравнения предположений об исполнении
можно запустить оценку сразу для всех профилей исполнения
`ExecutionProfile`.

```bash
python script_eval.py --config configs/config_eval.yaml --all-profiles
```

Или задать это через конфигурацию:

```yaml
all_profiles: true
input:
  trades_path: "logs/log_trades_<profile>.csv"
  equity_path: "logs/report_equity_<profile>.csv"
```

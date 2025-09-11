# Seasonality Quick Start

Follow these steps to regenerate and validate hour-of-week seasonality multipliers on a fresh clone of the repository.

1. **Install dependencies**

   ```bash
   pip install -r requirements_extra.txt
   ```

2. **Prepare source data**

   Place a Parquet or CSV file with historical trades under `data/seasonality_source/` or point the scripts to your own path. Timestamps must be in UTC.

3. **Build multipliers**

   ```bash
   python scripts/build_hourly_seasonality.py \
     --data data/seasonality_source/latest.parquet \
     --out configs/liquidity_latency_seasonality.json
   ```

4. **Visualise** *(optional)*

   ```bash
   python scripts/plot_seasonality.py --multipliers configs/liquidity_latency_seasonality.json
   ```

5. **Validate**

   ```bash
   python scripts/validate_seasonality.py \
     --historical data/seasonality_source/latest.parquet \
     --multipliers configs/liquidity_latency_seasonality.json
   ```

6. **Use in simulation**

   Pass the generated JSON to CLI scripts via `--liquidity-seasonality` or specify it in the YAML config:

   ```bash
   python train_model_multi_patch.py --config configs/config_train.yaml \
     --liquidity-seasonality configs/liquidity_latency_seasonality.json
   ```

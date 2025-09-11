.PHONY: format lint no-trade-mask-sample

format:
	black .

lint:
	flake8 --max-line-length=200 config.py transformers.py tune_threshold.py update_and_infer.py utils_time.py validate_processed.py watchdog_vec_env.py

no-trade-mask-sample:
	python tests/run_no_trade_mask_sample.py

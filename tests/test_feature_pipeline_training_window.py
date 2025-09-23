from pathlib import Path

import pandas as pd
import pytest

from features_pipeline import FeaturePipeline

FIXTURE = Path(__file__).resolve().parent / "data" / "pipeline_time_split.csv"


def _load_fixture() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(FIXTURE)
    return {"BTC": df}


def test_feature_pipeline_uses_train_mask(tmp_path):
    data = _load_fixture()
    df = data["BTC"]
    pipe = FeaturePipeline()
    pipe.fit(data, train_mask_column="wf_role", train_mask_values={"train"})

    stats = pipe.stats["feat"]
    train_values = df.loc[df["wf_role"] == "train", "feat"]
    assert stats["mean"] == pytest.approx(train_values.mean())
    assert pipe.metadata["train_rows_total"] == len(train_values)

    saved = tmp_path / "pipeline.json"
    pipe.save(saved)
    loaded = FeaturePipeline.load(saved)
    mask_values = loaded.metadata["filters"]["train_mask_values"]
    assert mask_values == ["train"]
    assert loaded.metadata["filters"]["train_mask_column"] == "wf_role"


def test_feature_pipeline_time_window_changes_stats():
    data = _load_fixture()
    df = data["BTC"]
    start = int(df["timestamp"].min())
    mid = int(df.loc[df["timestamp"] == 1700003600, "timestamp"].iloc[0])
    extended = int(df.loc[df["timestamp"] == 1700014400, "timestamp"].iloc[0])

    pipe_short = FeaturePipeline()
    pipe_short.fit(
        data,
        train_start_ts=start,
        train_end_ts=mid,
        timestamp_column="timestamp",
        train_intervals=[(start, mid)],
        split_version="fixture_v1",
    )

    pipe_long = FeaturePipeline()
    pipe_long.fit(
        data,
        train_start_ts=start,
        train_end_ts=extended,
        timestamp_column="timestamp",
        train_intervals=[(start, extended)],
        split_version="fixture_v1",
    )

    assert pipe_short.stats["feat"]["mean"] != pipe_long.stats["feat"]["mean"]

    metadata = pipe_long.get_metadata()
    assert metadata["filters"]["train_start_ts"] == start
    assert metadata["filters"]["train_end_ts"] == extended
    intervals = metadata["filters"]["train_intervals"]
    assert intervals == [{"start_ts": start, "end_ts": extended}]
    assert metadata.get("split_version") == "fixture_v1"

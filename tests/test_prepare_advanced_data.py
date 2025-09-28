import prepare_advanced_data


def test_prepare_advanced_data_handles_single_column_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "fear_greed.csv"
    csv_path.write_text("timestamp\n123\n456\n", encoding="utf-8")

    monkeypatch.setattr(prepare_advanced_data, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(prepare_advanced_data, "OUT_PATH", str(csv_path))
    monkeypatch.setattr(prepare_advanced_data, "fetch_fng", lambda limit=0: [])

    prepare_advanced_data.main()

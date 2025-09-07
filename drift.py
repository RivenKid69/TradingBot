# training/drift.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class NumericBins:
    edges: List[float]          # границы бинов (включая -inf и +inf)
    probs: List[float]          # доли по бинам на baseline (Qi)
    n: int                      # объём baseline

    def to_dict(self) -> Dict:
        return {"type": "numeric", "edges": list(self.edges), "probs": list(self.probs), "n": int(self.n)}

    @classmethod
    def from_dict(cls, d: Dict) -> "NumericBins":
        return cls(edges=[float(x) for x in d["edges"]], probs=[float(p) for p in d["probs"]], n=int(d.get("n", 0)))


@dataclass
class CategoricalDist:
    categories: List[str]       # список категорий (включая "OTHER", если был тримминг)
    probs: List[float]          # доли по категориям на baseline (Qi)
    n: int                      # объём baseline

    def to_dict(self) -> Dict:
        return {"type": "categorical", "categories": list(self.categories), "probs": list(self.probs), "n": int(self.n)}

    @classmethod
    def from_dict(cls, d: Dict) -> "CategoricalDist":
        return cls(categories=[str(x) for x in d["categories"]], probs=[float(p) for p in d["probs"]], n=int(d.get("n", 0)))


BaselineSpec = Dict[str, Union[NumericBins, CategoricalDist]]


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hist_from_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Возвращает количество элементов по бинам согласно edges (как в np.histogram).
    Все NaN игнорируются. Возвращает массив длины len(edges)-1.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    cnt, _ = np.histogram(x, bins=edges)
    return cnt.astype(float)


def _psi_from_counts(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """
    PSI = Σ (Pi - Qi) * ln(Pi/Qi)
    p_counts — текущие (prod window), q_counts — baseline.
    Малые значения заменяем на eps, чтобы избежать деления на 0.
    """
    eps = 1e-8
    p = p_counts.astype(float)
    q = q_counts.astype(float)
    if p.sum() <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    if q.sum() <= 0:
        q = np.ones_like(q) / len(q)
    else:
        q = q / q.sum()
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))


def _build_numeric_bins_baseline(series: pd.Series, bins: int = 10) -> NumericBins:
    """
    Строим квантили по baseline и превращаем в «замкнутые» бины: [-inf, q1, q2, ..., +inf].
    """
    x = _safe_to_numeric(series).dropna().to_numpy(dtype=float)
    n = int(x.size)
    if n == 0:
        edges = np.array([-np.inf, np.inf], dtype=float)
        probs = np.array([1.0], dtype=float)
        return NumericBins(edges=list(edges), probs=list(probs), n=0)

    qs = np.linspace(0.0, 1.0, bins + 1)
    # избегаем совпадающих квантилей
    raw_edges = np.quantile(x, qs)
    # гарантируем строгую возрастающую последовательность через добавление -inf/+inf и устранение дублей
    edges = [-np.inf]
    for val in raw_edges[1:-1]:
        if len(edges) == 0 or val > edges[-1]:
            edges.append(float(val))
        else:
            # если квантили совпали, слегка подвинем
            edges.append(float(edges[-1] + 1e-12))
    edges.append(np.inf)
    edges = np.array(edges, dtype=float)

    counts = _hist_from_edges(x, edges)
    if counts.sum() <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / counts.sum()
    return NumericBins(edges=list(edges), probs=list(probs), n=n)


def _build_categorical_baseline(series: pd.Series, top_k: int = 20) -> CategoricalDist:
    """
    Берём top_k категорий по частоте. Остальные — в "OTHER".
    """
    s = series.astype("string")
    vc = s.value_counts(dropna=True)
    n = int(vc.sum())
    cats = vc.index.tolist()
    if len(cats) == 0:
        return CategoricalDist(categories=["OTHER"], probs=[1.0], n=0)
    if len(cats) > top_k:
        head = vc.iloc[:top_k]
        other = float(vc.iloc[top_k:].sum())
        categories = head.index.tolist() + ["OTHER"]
        counts = head.to_numpy(dtype=float).tolist() + [other]
    else:
        categories = cats
        counts = vc.to_numpy(dtype=float).tolist()
    counts = np.array(counts, dtype=float)
    if counts.sum() <= 0:
        probs = np.ones_like(counts) / len(counts)
    else:
        probs = counts / counts.sum()
    return CategoricalDist(categories=[str(c) for c in categories], probs=list(probs), n=n)


def make_baseline(
    df: pd.DataFrame,
    features: List[str],
    *,
    bins: int = 10,
    categorical: Optional[List[str]] = None,
    top_k_cats: int = 20,
) -> Dict[str, Dict]:
    """
    Строит baseline-спецификацию: для числовых — квантили и доли, для категориальных — частоты категорий.
    Возвращает dict, пригодный для сохранения в JSON.
    """
    categorical = set(categorical or [])
    spec: Dict[str, Dict] = {}
    for col in features:
        if col not in df.columns:
            continue
        s = df[col]
        if col in categorical or (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)):
            cat = _build_categorical_baseline(s, top_k=top_k_cats)
            spec[col] = cat.to_dict()
        else:
            num = _build_numeric_bins_baseline(s, bins=bins)
            spec[col] = num.to_dict()
    return spec


def save_baseline_json(spec: Dict[str, Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)


def load_baseline_json(path: str) -> BaselineSpec:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: BaselineSpec = {}
    for k, v in d.items():
        t = str(v.get("type", "numeric")).lower()
        if t == "numeric":
            out[k] = NumericBins.from_dict(v)
        elif t == "categorical":
            out[k] = CategoricalDist.from_dict(v)
        else:
            raise ValueError(f"Неизвестный тип в baseline для {k}: {t}")
    return out


def _psi_numeric_current(series: pd.Series, nb: NumericBins) -> float:
    x = _safe_to_numeric(series).to_numpy(dtype=float)
    edges = np.asarray(nb.edges, dtype=float)
    q_counts = np.asarray(nb.probs, dtype=float) * max(nb.n, 1)
    p_counts = _hist_from_edges(x, edges)
    return _psi_from_counts(p_counts, q_counts)


def _psi_categorical_current(series: pd.Series, cd: CategoricalDist) -> float:
    s = series.astype("string")
    vc = s.value_counts(dropna=True)
    cats = list(cd.categories)
    # соберём counts по cats, остальные — в OTHER (если есть)
    counts = []
    other_count = 0.0
    for c in vc.index.tolist():
        val = float(vc[c])
        if c in cats:
            # будет добавлено ниже
            pass
        else:
            other_count += val
    for c in cats:
        if c == "OTHER":
            counts.append(other_count)
        else:
            counts.append(float(vc.get(c, 0.0)))
    p_counts = np.asarray(counts, dtype=float)
    q_counts = np.asarray(cd.probs, dtype=float) * max(cd.n, 1)
    return _psi_from_counts(p_counts, q_counts)


def compute_psi(
    current_df: pd.DataFrame,
    baseline: BaselineSpec,
    *,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Считает PSI для заданных колонок по baseline-спецификации.
    Возвращает DataFrame с колонками: ['feature','psi','type','n_current','n_baseline'].
    """
    feats = list(features) if features is not None else list(baseline.keys())
    rows: List[Dict] = []
    for col in feats:
        if col not in baseline:
            continue
        if col not in current_df.columns:
            rows.append({"feature": col, "psi": float("nan"), "type": "missing", "n_current": 0, "n_baseline": 0})
            continue
        spec = baseline[col]
        try:
            if isinstance(spec, NumericBins) or (isinstance(spec, dict) and spec.get("type") == "numeric"):
                nb = spec if isinstance(spec, NumericBins) else NumericBins.from_dict(spec)  # type: ignore
                psi = _psi_numeric_current(current_df[col], nb)
                rows.append({"feature": col, "psi": float(psi), "type": "numeric", "n_current": int(current_df[col].notna().sum()), "n_baseline": int(nb.n)})
            elif isinstance(spec, CategoricalDist) or (isinstance(spec, dict) and spec.get("type") == "categorical"):
                cd = spec if isinstance(spec, CategoricalDist) else CategoricalDist.from_dict(spec)  # type: ignore
                psi = _psi_categorical_current(current_df[col], cd)
                rows.append({"feature": col, "psi": float(psi), "type": "categorical", "n_current": int(current_df[col].notna().sum()), "n_baseline": int(cd.n)})
            else:
                rows.append({"feature": col, "psi": float("nan"), "type": "unknown", "n_current": 0, "n_baseline": 0})
        except Exception:
            rows.append({"feature": col, "psi": float("nan"), "type": "error", "n_current": 0, "n_baseline": 0})
    res = pd.DataFrame(rows)
    res = res.sort_values(["psi"], ascending=[False]).reset_index(drop=True)
    return res


def default_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Простая эвристика: все числовые фичи, начинающиеся с 'f_' или заканчивающиеся на '_z', плюс 'score', если есть.
    """
    out: List[str] = []
    for c in df.columns:
        if c.startswith("f_") or c.endswith("_z"):
            out.append(c)
    if "score" in df.columns:
        out.append("score")
    return sorted(list(dict.fromkeys(out)))

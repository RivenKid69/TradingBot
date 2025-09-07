# training/calibration.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utils -----------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    # численно стабильный сигмоид
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def ece_score(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Expected Calibration Error (исходная формула с L1).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    if p.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    total = len(p)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            continue
        conf = float(np.mean(p[sel]))
        acc = float(np.mean(y[sel]))
        w = float(np.sum(sel)) / float(total)
        ece += w * abs(acc - conf)
    return float(ece)


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Возвращает таблицу по корзинам: [bin_low, bin_high, n, mean_score, empirical_rate].
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict] = []
    idx = np.digitize(p, edges[1:-1], right=False)
    for b in range(bins):
        sel = idx == b
        if np.any(sel):
            rows.append({
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(np.sum(sel)),
                "mean_score": float(np.mean(p[sel])),
                "empirical_rate": float(np.mean(y[sel])),
            })
        else:
            rows.append({
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": 0,
                "mean_score": float("nan"),
                "empirical_rate": float("nan"),
            })
    return pd.DataFrame(rows)


# ----------------------------- base class -----------------------------

class BaseCalibrator:
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BaseCalibrator":
        raise NotImplementedError

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: Dict) -> "BaseCalibrator":
        raise NotImplementedError

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> "BaseCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = str(d.get("type", "")).lower()
        if t == "platt":
            return PlattCalibrator.from_dict(d)
        if t == "isotonic":
            return IsotonicCalibrator.from_dict(d)
        raise ValueError(f"Неизвестный тип калибратора: {t}")


# ----------------------------- Platt scaling -----------------------------

@dataclass
class PlattCalibrator(BaseCalibrator):
    """
    p = sigmoid(w * x + b)
    Обучение: Ньютона-Рафсона для логистической регрессии в одномерном признаке (score).
    """
    w: float = 0.0
    b: float = 0.0

    def fit(self, scores: np.ndarray, labels: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> "PlattCalibrator":
        x = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            self.w = 0.0
            self.b = float(np.log((np.mean(y) + 1e-6) / (1.0 - np.mean(y) + 1e-6)))
            return self

        # центрируем x для стабильности
        xm = float(np.mean(x))
        xs = x - xm

        w = 0.0
        b = float(np.log((np.mean(y) + 1e-6) / (1.0 - np.mean(y) + 1e-6)))

        for _ in range(int(max_iter)):
            z = w * xs + b
            p = _sigmoid(z)
            # градиент и Гессиан
            r = p - y
            g_w = float(np.dot(xs, r))
            g_b = float(np.sum(r))
            W = p * (1.0 - p)
            H_ww = float(np.dot(xs * xs, W))
            H_bb = float(np.sum(W))
            H_wb = float(np.dot(xs, W))
            # решаем 2x2
            det = H_ww * H_bb - H_wb * H_wb
            if abs(det) < 1e-12:
                break
            dw = -( H_bb * g_w - H_wb * g_b) / det
            db = -(-H_wb * g_w + H_ww * g_b) / det
            w_new = w + dw
            b_new = b + db
            if max(abs(dw), abs(db)) < tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new

        # возвращаем смещение обратно
        self.w = float(w)
        self.b = float(b - w * xm)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        x = np.asarray(scores, dtype=float).reshape(-1)
        z = self.w * x + self.b
        return _sigmoid(z)

    def to_dict(self) -> Dict:
        return {"type": "platt", "w": float(self.w), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d: Dict) -> "PlattCalibrator":
        return cls(w=float(d.get("w", 0.0)), b=float(d.get("b", 0.0)))


# ----------------------------- Isotonic regression -----------------------------

@dataclass
class IsotonicCalibrator(BaseCalibrator):
    """
    Монотонная калибровка (PAV).
    Сохраняем ступенчатую функцию как пары (x_thresholds, y_values).
    """
    x_thresholds: List[float] = None
    y_values: List[float] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            self.x_thresholds = [0.0, 1.0]
            self.y_values = [float(np.mean(y) if y.size else 0.5), float(np.mean(y) if y.size else 0.5)]
            return self

        # агрегируем по уникальным x: среднее y и вес (частота)
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]

        uniq_x = []
        mean_y = []
        weight = []
        i = 0
        n = len(xs)
        while i < n:
            j = i + 1
            s = ys[i]
            w = 1
            while j < n and xs[j] == xs[i]:
                s += ys[j]
                w += 1
                j += 1
            uniq_x.append(float(xs[i]))
            mean_y.append(float(s / w))
            weight.append(int(w))
            i = j

        # PAV
        v = np.array(mean_y, dtype=float)
        w = np.array(weight, dtype=float)

        # стек блоков
        blocks = []
        for i in range(len(v)):
            blocks.append([v[i], w[i], uniq_x[i], uniq_x[i]])  # [mean, weight, x_left, x_right]
            # сливаем, пока нарушена монотонность
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                m2, w2, l2, r2 = blocks.pop()
                m1, w1, l1, r1 = blocks.pop()
                m = (m1 * w1 + m2 * w2) / (w1 + w2)
                blocks.append([m, w1 + w2, l1, r2])

        # извлекаем пороги и значения (ступени)
        x_thresh = [blk[3] for blk in blocks]  # правые границы блоков
        y_vals = [blk[0] for blk in blocks]

        # уберём возможные дубликаты границ
        xt = []
        yv = []
        for i in range(len(x_thresh)):
            if i == 0 or x_thresh[i] > x_thresh[i - 1]:
                xt.append(float(x_thresh[i]))
                yv.append(float(y_vals[i]))
            else:
                # если одинаковые — перезапишем последним
                xt[-1] = float(x_thresh[i])
                yv[-1] = float(y_vals[i])

        # крайние значения для экстраполяции
        self.x_thresholds = xt
        self.y_values = yv
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.x_thresholds or not self.y_values:
            # не обучен — вернём 0.5
            return np.full(shape=(len(np.asarray(scores).reshape(-1)),), fill_value=0.5, dtype=float)
        x = np.asarray(scores, dtype=float).reshape(-1)
        out = np.empty_like(x, dtype=float)
        # для каждого x найдём первую границу >= x
        xt = np.asarray(self.x_thresholds, dtype=float)
        yv = np.asarray(self.y_values, dtype=float)
        for i in range(len(x)):
            xi = x[i]
            j = np.searchsorted(xt, xi, side="left")
            if j >= len(xt):
                out[i] = float(yv[-1])
            else:
                out[i] = float(yv[j])
        return out

    def to_dict(self) -> Dict:
        return {"type": "isotonic", "x_thresholds": list(self.x_thresholds or []), "y_values": list(self.y_values or [])}

    @classmethod
    def from_dict(cls, d: Dict) -> "IsotonicCalibrator":
        return cls(
            x_thresholds=[float(x) for x in (d.get("x_thresholds", []) or [])],
            y_values=[float(y) for y in (d.get("y_values", []) or [])],
        )


# ----------------------------- high-level API -----------------------------

def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "platt",
) -> BaseCalibrator:
    method = str(method).lower().strip()
    if method == "platt":
        cal = PlattCalibrator()
        cal.fit(scores, labels)
        return cal
    if method == "isotonic":
        cal = IsotonicCalibrator()
        cal.fit(scores, labels)
        return cal
    raise ValueError(f"Неизвестный метод калибровки: {method}")


def evaluate_before_after(
    scores: np.ndarray,
    labels: np.ndarray,
    calibrator: Optional[BaseCalibrator],
    bins: int = 10,
) -> Dict[str, float]:
    """
    Возвращает метрики до/после: brier_before, brier_after, ece_before, ece_after.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    mask = np.isfinite(scores) & np.isfinite(labels)
    s = scores[mask]
    y = labels[mask]
    if s.size == 0:
        return {"brier_before": float("nan"), "brier_after": float("nan"), "ece_before": float("nan"), "ece_after": float("nan")}
    p_before = np.clip(s, 0.0, 1.0)
    b_before = brier_score(p_before, y)
    e_before = ece_score(p_before, y, bins=bins)
    if calibrator is None:
        return {"brier_before": b_before, "brier_after": float("nan"), "ece_before": e_before, "ece_after": float("nan")}
    p_after = np.clip(calibrator.predict_proba(s), 0.0, 1.0)
    b_after = brier_score(p_after, y)
    e_after = ece_score(p_after, y, bins=bins)
    return {"brier_before": b_before, "brier_after": b_after, "ece_before": e_before, "ece_after": e_after}

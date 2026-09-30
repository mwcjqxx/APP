from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost
from matplotlib import font_manager
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
ASSETS = BASE / 'assets'
MODEL_DIR = BASE / 'model'

SELECTED = ['Gender', 'HbA1c', 'BAP', 'Total_PINP', 'Age', 'Muscle_Mass', 'A/G', 'RSMI']

NSAMPLES = 512

C_RISE = '#e8003d'
C_FALL = '#1e88e5'
C_TEXT = '#22303C'
C_MUTED = '#5A6672'
C_GRID = '#DCE2E8'

def setup_style():
    for f in ('times.ttf', 'timesbd.ttf', 'timesi.ttf'):
        p = Path(r'C:\Windows\Fonts') / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 8,
        'axes.edgecolor': C_MUTED,
        'axes.labelcolor': C_TEXT,
        'text.color': C_TEXT,
        'xtick.color': C_MUTED,
        'ytick.color': C_MUTED,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

def load_assets():
    with open(ASSETS / 'artifacts.json', encoding='utf-8') as fh:
        return json.load(fh)

class BoosterModel:

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        dmat = xgboost.DMatrix(arr, feature_names=SELECTED)
        p = np.asarray(self.booster.predict(dmat), dtype=float).reshape(-1)
        return np.column_stack([1.0 - p, p])

def load_model():
    booster = xgboost.Booster()
    with open(MODEL_DIR / 'XGBoost.json', 'rb') as fh:
        booster.load_model(bytearray(fh.read()))
    return BoosterModel(booster)

def load_background():
    return pd.read_csv(ASSETS / 'background_internal_60.csv', encoding='utf-8')[SELECTED]

def make_explainer(model, background, nsamples=NSAMPLES):
    def predict_pos(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=SELECTED))[:, 1]

    return shap.KernelExplainer(predict_pos, background, nsamples=nsamples)

def to_frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([[float(values[c]) for c in SELECTED]], columns=SELECTED)

def predict_proba(model, values: dict) -> float:
    return float(model.predict_proba(to_frame(values))[:, 1][0])

def shap_local(explainer, values: dict, nsamples=NSAMPLES):
    x = to_frame(values)
    sv = np.asarray(explainer.shap_values(x.values, nsamples=nsamples, silent=True))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    sv = np.asarray(sv).reshape(-1)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    return pd.Series(sv, index=SELECTED), base, base + float(sv.sum())

def risk_band(prob: float, threshold: float) -> str:
    return 'High risk' if prob >= threshold else 'Low risk'

def _label(nm, v):
    if nm == 'Gender':
        return 'Gender = %d' % int(round(v))
    return '%s = %.2f' % (nm, v)

def fig_force_plot(sv: pd.Series, base: float, fx: float, values: dict,
                   figsize=(5.9, 3.55), fs=9.0, dpi=130, threshold=None):
    vals = np.asarray(sv.values, dtype=float)
    fvals = np.asarray([float(values[c]) for c in sv.index], dtype=float)
    order = np.argsort(-np.abs(vals))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.012, right=0.988,
                        top=0.80 if threshold is not None else 0.99, bottom=0.16)
    ax.set_autoscale_on(False)
    left = right = float(base)
    segs = []
    for i in order:
        v = vals[i]
        if v >= 0:
            segs.append((right, right + v, C_RISE, i))
            right += v
        else:
            segs.append((left + v, left, C_FALL, i))
            left += v
    lo, hi = min(left, right), max(left, right)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.05 * span, hi + 0.05 * span)
    Y_LO, Y_HI = -0.02, 2.04
    ax.set_ylim(Y_LO, Y_HI)
    x_lo, x_hi = ax.get_xlim()

    renderer = fig.canvas.get_renderer()

    def tw(s):
        t = ax.text(0.0, 0.0, s, fontsize=fs)
        w = t.get_window_extent(renderer=renderer).width / fig.dpi
        t.remove()
        return w

    av = figsize[0] * 0.976
    over, used = [], [[], [], [], []]
    LEVEL = (1.00, 1.22, 1.44, 1.66)
    for x0, x1, col, i in segs:
        ax.add_patch(Rectangle((x0, 0), x1 - x0, 0.85, facecolor=col,
                               edgecolor='white', linewidth=0.25))
        lab = _label(sv.index[i], fvals[i])
        w_in = tw(lab)
        if w_in <= (x1 - x0) / (x_hi - x_lo) * av - 0.035:
            ax.text((x0 + x1) / 2.0, 0.425, lab, ha='center', va='center',
                    fontsize=fs, color='white')
        else:
            over.append(((x0 + x1) / 2.0, lab, w_in, i))

    def place(xc, lab, w_in):
        w = w_in / av * (x_hi - x_lo)
        pad = 0.10 / av * (x_hi - x_lo)
        half = w / 2.0
        x_orig = xc
        xc = min(max(xc, x_lo + half + 0.01), x_hi - half - 0.01)
        a, b = xc - half - pad, xc + half + pad
        for L, yv in enumerate(LEVEL):
            if all(b < c or a > d for c, d in used[L]):
                used[L].append((a, b))
                ax.plot([x_orig, xc], [0.85, yv - 0.05], color='#9a9a9a', lw=0.5, zorder=1)
                ax.text(xc, yv, lab, ha='center', va='bottom', fontsize=fs,
                        color='#333333')
                return True
        return False

    for xc, lab, w_in, i in sorted(over, key=lambda t: t[0]):
        if place(xc, lab, w_in):
            continue
        short = str(sv.index[i])
        place(xc, short, tw(short))

    ax.set_yticks([])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(C_GRID)
    ax.set_xlabel('Predicted probability', labelpad=1.5)
    ax.tick_params(axis='x', pad=1.5)

    ytxt = (1.84 - Y_LO) / (Y_HI - Y_LO)
    ax.text(0.006, ytxt, 'E[f(X)] = %.2f' % base, transform=ax.transAxes,
            fontsize=fs, color=C_MUTED, ha='left', va='bottom')
    ax.text(0.5, ytxt, 'red: higher risk    blue: lower risk', transform=ax.transAxes,
            fontsize=fs, color=C_MUTED, ha='center', va='bottom')
    ax.text(0.994, ytxt, 'f(x) = %.2f' % fx, transform=ax.transAxes,
            fontsize=fs, color=C_MUTED, ha='right', va='bottom')

    if threshold is not None:
        tag = 'High risk' if fx >= threshold else 'Low risk'
        ax.set_title('%s | predicted risk = %.2f' % (tag, fx),
                     fontsize=fs + 1.0, color=C_TEXT, pad=9)
    return fig

def fig_waterfall(sv: pd.Series, base: float, fx: float, values: dict,
                  figsize=(7.0, 4.15), fs=8.8, dpi=130, threshold=None):
    vals = np.asarray(sv.values, dtype=float)
    fvals = np.asarray([float(values[c]) for c in sv.index], dtype=float)
    order = np.argsort(-np.abs(vals))
    names = [str(sv.index[i]) for i in order]
    v = vals[order]
    fv = fvals[order]
    n = len(v)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0.30, right=0.975,
                        top=0.865 if threshold is not None else 0.955,
                        bottom=0.105)
    ax.set_autoscale_on(False)

    cum = [float(base)]
    for k in range(n):
        cum.append(cum[-1] + float(v[k]))

    refs = [float(base), float(fx), 0.0] + cum
    lo, hi = min(refs), max(refs)
    pad = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.13 * pad, hi + 0.20 * pad)
    ax.set_ylim(n + 1.62, -0.62)

    for k in range(n):
        y = k + 1
        w = float(v[k])
        x0 = cum[k]
        ax.barh(y, w, left=x0, height=0.62, zorder=3,
                color=C_RISE if w >= 0 else C_FALL,
                edgecolor='white', linewidth=0.4)
        if w >= 0:
            ax.text(x0 + w, y, '  +%.3f' % w, ha='left', va='center',
                    fontsize=fs - 0.4, color='#4a5560', zorder=4)
        else:
            ax.text(x0 + w, y, '%.3f  ' % w, ha='right', va='center',
                    fontsize=fs - 0.4, color='#4a5560', zorder=4)

    ax.plot([float(base)], [0.0], marker='s', markersize=4.6,
            color='#7c8b99', zorder=4)
    ax.barh(n + 1, float(fx), left=0.0, height=0.62, color='#1f4e79',
            edgecolor='white', linewidth=0.4, zorder=3)
    ax.text(float(fx), n + 1, '  %.2f' % float(fx), ha='left',
            va='center', fontsize=fs - 0.4, color='#4a5560', zorder=4)

    ax.axvline(float(base), color='#9aa7b4', lw=0.7, ls=(0, (3, 2.4)), zorder=2)

    ax.set_yticks(np.arange(n + 2))
    ax.set_yticklabels(['E[f(X)]'] + [_label(names[k], fv[k]) for k in range(n)]
                       + ['f(x)'], fontsize=fs - 0.2)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xlabel('Predicted probability', labelpad=2)
    ax.tick_params(axis='x', labelsize=fs - 1.2, pad=1.5)
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(C_GRID)
    ax.set_axisbelow(True)

    ax.text(0.5, 1.012, 'red: higher risk    blue: lower risk', transform=ax.transAxes,
            fontsize=fs - 1.4, color=C_MUTED, ha='center', va='bottom')

    if threshold is not None:
        tag = 'High risk' if fx >= threshold else 'Low risk'
        ax.set_title('%s | predicted risk = %.2f' % (tag, fx),
                     fontsize=fs + 0.9, color=C_TEXT, pad=24)
    return fig

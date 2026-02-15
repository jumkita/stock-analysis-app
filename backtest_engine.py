# -*- coding: utf-8 -*-
"""
シグナル有効性検証用バックテストエンジン
各売買シグナルのパフォーマンスを検証し、勝てるシグナルの選別を支援する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class BacktestEngine:
    """
    テクニカルシグナルの有効性を検証するバックテストエンジン。
    vectorized operation を活用し、複数シグナルの一括検証を高速に行う。
    """

    def run(
        self,
        df: pd.DataFrame,
        signal_columns: list[str],
        holding_period_days: int = 5,
        stop_loss_pct: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        各シグナル列についてバックテストを実行し、指標をまとめたDataFrameを返す。

        Parameters
        ----------
        df : pd.DataFrame
            OHLCVデータ。必須列: Open, High, Low, Close
        signal_columns : list[str]
            検証対象のシグナル列名（True/Falseのブール列）
        holding_period_days : int, default 5
            保有期間（営業日）
        stop_loss_pct : float, optional
            損切りライン。例: 0.05 なら -5% で損切り。None なら無効

        Returns
        -------
        pd.DataFrame
            Signal Name, Total Trades, Win Rate, Profit Factor, Avg Return,
            Max Drawdown, Expectancy を列とする結果
        """
        required = ["Open", "High", "Low", "Close"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"DataFrame に必須列 '{c}' がありません")

        results = []
        for col in signal_columns:
            if col not in df.columns:
                continue
            metrics = self._backtest_single_signal(
                df, col, holding_period_days, stop_loss_pct
            )
            if metrics is not None:
                metrics["Signal Name"] = col
                results.append(metrics)

        if not results:
            return pd.DataFrame(
                columns=[
                    "Signal Name", "Total Trades", "Win Rate", "Profit Factor",
                    "Avg Return", "Max Drawdown", "Expectancy",
                ]
            )

        return pd.DataFrame(results)[
            [
                "Signal Name", "Total Trades", "Win Rate", "Profit Factor",
                "Avg Return", "Max Drawdown", "Expectancy",
            ]
        ]

    def _backtest_single_signal(
        self,
        df: pd.DataFrame,
        signal_col: str,
        holding_period_days: int,
        stop_loss_pct: Optional[float],
    ) -> Optional[dict]:
        """単一シグナル列のバックテストを実行。"""
        mask = df[signal_col].fillna(False).astype(bool)
        entry_indices = np.where(mask)[0]

        if len(entry_indices) == 0:
            return None

        n = len(df)
        close = df["Close"].values.astype(float)
        low = df["Low"].values.astype(float)

        entry_prices = close[entry_indices]
        exit_indices = np.minimum(entry_indices + holding_period_days, n - 1)

        if stop_loss_pct is not None and stop_loss_pct > 0:
            stop_prices = entry_prices * (1 - stop_loss_pct)
            returns_list = []
            for k, idx in enumerate(entry_indices):
                window_end = min(idx + holding_period_days + 1, n)
                low_window = low[idx + 1 : window_end]
                hit_mask = low_window < stop_prices[k]
                if np.any(hit_mask):
                    ret = (stop_prices[k] - entry_prices[k]) / entry_prices[k]
                else:
                    ret = (close[exit_indices[k]] - entry_prices[k]) / entry_prices[k]
                returns_list.append(ret)
            returns = np.array(returns_list)
        else:
            exit_prices = close[exit_indices]
            returns = (exit_prices - entry_prices) / entry_prices
        return self._compute_metrics(returns)

    def _compute_metrics(self, returns: np.ndarray) -> dict:
        """リターン配列から指標を計算。"""
        n = len(returns)
        if n == 0:
            return None

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        total_profit = np.sum(wins) if len(wins) > 0 else 0.0
        total_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-12
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

        win_rate = len(wins) / n
        avg_return_pct = np.mean(returns) * 100

        cum_ret = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = running_max - cum_ret
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        loss_rate = 1 - win_rate
        expectancy = avg_win * win_rate + avg_loss * loss_rate

        return {
            "Total Trades": n,
            "Win Rate": round(win_rate, 4),
            "Profit Factor": round(profit_factor, 4),
            "Avg Return": round(avg_return_pct, 4),
            "Max Drawdown": round(max_dd, 4),
            "Expectancy": round(expectancy, 6),
        }


# ========== 可視化用ヘルパー（app.py からの呼び出し例） ==========
#
# ```python
# from backtest_engine import BacktestEngine, plot_backtest_results
#
# engine = BacktestEngine()
# res = engine.run(df, signal_columns=["Buy_RedThreeSoldiers", "Buy_MorningStar"], holding_period_days=5, stop_loss_pct=0.05)
# st.dataframe(res)
# plot_backtest_results(res, metric="Win Rate", kind="bar")
# plot_backtest_results(res, kind="heatmap")
# ```


def plot_backtest_results(
    result_df: pd.DataFrame,
    metric: str = "Win Rate",
    kind: str = "bar",
):
    """
    バックテスト結果を可視化するPlotly Figureを返す。
    app側で st.plotly_chart(plot_backtest_results(...), ...) により表示可能。

    Parameters
    ----------
    result_df : pd.DataFrame
        BacktestEngine.run() の戻り値
    metric : str
        表示する指標（Win Rate, Profit Factor, Avg Return など）
    kind : str
        "bar" または "heatmap"

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    import plotly.express as px

    if result_df.empty:
        return None

    if metric not in result_df.columns:
        metric = "Win Rate"

    if kind == "bar":
        fig = px.bar(
            result_df,
            x="Signal Name",
            y=metric,
            title=f"シグナル別 {metric}",
            labels={"Signal Name": "シグナル", metric: metric},
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    else:
        metrics_for_heat = ["Win Rate", "Profit Factor", "Avg Return", "Expectancy"]
        cols = [c for c in metrics_for_heat if c in result_df.columns]
        if not cols:
            return None
        plot_df = result_df.set_index("Signal Name")[cols]
        fig = px.imshow(
            plot_df.T,
            labels=dict(x="Signal", y="Metric", color="Value"),
            x=plot_df.index.tolist(),
            y=cols,
            aspect="auto",
            color_continuous_scale="RdYlGn",
        )
        return fig


# ========== 使用例 ==========

if __name__ == "__main__":
    # サンプル: 既存の detect_all_patterns 結果をシグナル列に変換してバックテスト
    import sys
    sys.path.insert(0, ".")
    from logic import fetch_ohlcv, detect_all_patterns

    ticker = "7203.T"
    df = fetch_ohlcv(ticker, period="1y", interval="1d")
    if df is None or len(df) < 30:
        print("データ取得失敗")
    else:
        patterns = detect_all_patterns(df)
        # パターン結果を True/False 列に変換（簡易）
        for i, name, side in patterns:
            if side == "buy":
                col = f"Buy_{name}"
                if col not in df.columns:
                    df[col] = False
                df.loc[df.index[i], col] = True
        signal_cols = [c for c in df.columns if c.startswith("Buy_")]
        if signal_cols:
            engine = BacktestEngine()
            res = engine.run(df, signal_cols, holding_period_days=5, stop_loss_pct=0.05)
            print(res)
        else:
            print("シグナル列がありません")

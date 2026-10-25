"""
Perishable Stock Optimizer
--------------------------
Decides how much of each perishable product to order every day so that
waste (expired stock) and lost sales (stock-outs) are both kept low.

Pipeline
  1. Load (or generate) daily sales data per product
  2. Forecast demand with a Gradient Boosting model
  3. Order up to  (forecast demand over lead time + 1 day) + z * forecast error
  4. Choose the safety factor z per product by back-testing a FIFO inventory
     simulation with expiry on the last 60 days of the training period
  5. Compare against a naive "50% buffer" policy on unseen days, save results

Usage
  python perishable_stock_optimizer.py                 # uses generated sample data
  python perishable_stock_optimizer.py --csv sales.csv # your own data
      CSV columns: date, product, units_sold
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# --------------------------------------------------------------------------
# 1. Settings
#    shelf_life : days a unit stays sellable after it arrives
#    cost/price : per-unit purchase cost and selling price
#    salvage    : value recovered from an expired unit
#    base       : average daily demand (sample data only)
# --------------------------------------------------------------------------
PRODUCTS = {
    "Milk":     dict(shelf_life=6,  cost=1.00, price=1.60, salvage=0.00, base=120),
    "Bread":    dict(shelf_life=3,  cost=0.80, price=1.50, salvage=0.10, base=90),
    "Yogurt":   dict(shelf_life=14, cost=0.60, price=1.10, salvage=0.00, base=70),
    "Tomatoes": dict(shelf_life=5,  cost=1.20, price=2.20, salvage=0.20, base=60),
    "Chicken":  dict(shelf_life=4,  cost=3.50, price=5.50, salvage=0.00, base=45),
}
DEFAULT_PARAMS = dict(shelf_life=5, cost=1.00, price=1.60, salvage=0.00)

LEAD_TIME = 2          # an order placed on day t arrives on day t + 2
TRAIN_DAYS = 270       # first 270 days train the model, the rest are simulated
TUNE_DAYS = 60         # last 60 training days are used to pick the safety factor z
Z_GRID = np.arange(-0.5, 3.01, 0.25)
SEED = 42


# --------------------------------------------------------------------------
# 2. Data
# --------------------------------------------------------------------------
def generate_sales_data(days=365, seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    weekly = np.array([0.90, 0.88, 0.92, 1.00, 1.15, 1.30, 1.20])  # Mon..Sun
    rows = []
    for name, p in PRODUCTS.items():
        trend = 1 + 0.0004 * np.arange(days)
        mean = p["base"] * weekly[dates.dayofweek] * trend
        mean = mean * rng.lognormal(0, 0.12, days)           # day-to-day demand shocks
        units = np.clip(np.round(rng.poisson(mean)), 0, None).astype(int)
        rows.append(pd.DataFrame({"date": dates, "product": name, "units_sold": units}))
    return pd.concat(rows, ignore_index=True)


def load_data(csv_path):
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        df = generate_sales_data()
        df.to_csv("sales_data.csv", index=False)
        print("Generated sample data -> sales_data.csv")
    return df.sort_values(["product", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Demand forecasting
#    A forecast for day d made at the start of day t (= d - k, k = 0..LEAD_TIME)
#    may only use sales up to day t - 1.
# --------------------------------------------------------------------------
class DemandForecaster:
    def __init__(self, units, dates, train_days):
        self.units = np.asarray(units, dtype=float)
        self.dow = pd.DatetimeIndex(dates).dayofweek.values
        self.train_days = train_days
        self.sigma = {k: 1.0 for k in range(LEAD_TIME + 1)}
        self._cache = {}
        self._fit()

    def _features(self, d, k):
        t = d - k
        u = self.units
        return [self.dow[d], u[d - 7], u[d - 14], u[t - 7:t].mean(), k]

    def _rows(self, start, end):
        X, y, ks = [], [], []
        for d in range(start, end):
            for k in range(LEAD_TIME + 1):
                X.append(self._features(d, k))
                y.append(self.units[d])
                ks.append(k)
        return np.array(X), np.array(y), np.array(ks)

    @staticmethod
    def _new_model():
        return GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=SEED)

    def _fit(self):
        X, y, ks = self._rows(15, self.train_days)
        split = int(len(X) * 0.8)
        # hold-out residuals give an honest estimate of forecast error
        probe = self._new_model().fit(X[:split], y[:split])
        resid = y[split:] - probe.predict(X[split:])
        for k in range(LEAD_TIME + 1):
            r = resid[ks[split:] == k]
            self.sigma[k] = float(np.std(r)) if len(r) > 1 else 1.0
        self.model = self._new_model().fit(X, y)

    def predict(self, d, k):
        key = (d, k)
        if key not in self._cache:
            self._cache[key] = max(0.0, float(self.model.predict([self._features(d, k)])[0]))
        return self._cache[key]


# --------------------------------------------------------------------------
# 4. Ordering policies
# --------------------------------------------------------------------------
def optimized_policy(forecaster, z):
    sigma = math.sqrt(sum(s ** 2 for s in forecaster.sigma.values()))

    def policy(t, position, units):
        # expected demand until an order placed today could be followed by the next delivery
        mu = sum(forecaster.predict(min(t + k, len(units) - 1), k) for k in range(LEAD_TIME + 1))
        target = mu + z * sigma
        return max(0, round(target - position))

    return policy


def naive_policy():
    """Baseline: keep 50% more than the 7-day average demand on hand."""
    def policy(t, position, units):
        avg = units[max(0, t - 7):t].mean()
        target = 1.5 * avg * (LEAD_TIME + 1)
        return max(0, round(target - position))

    return policy


# --------------------------------------------------------------------------
# 5. FIFO inventory simulation with expiry
# --------------------------------------------------------------------------
def simulate(units, start, policy, p, end=None):
    end = end or len(units)
    batches = [[units[:start].mean() * 2, p["shelf_life"] - 1 + start]]  # [qty, last sellable day]
    pipeline = {}                                                         # arrival_day -> qty
    ordered = sold = wasted = lost = 0.0
    log = []

    for t in range(start, end):
        if t in pipeline:                                    # delivery arrives, fresh
            batches.append([pipeline.pop(t), t + p["shelf_life"] - 1])

        on_hand = sum(b[0] for b in batches)
        position = on_hand + sum(pipeline.values())
        qty = policy(t, position, units)
        if qty > 0:
            pipeline[t + LEAD_TIME] = pipeline.get(t + LEAD_TIME, 0) + qty
            ordered += qty

        demand = units[t]                                     # sell oldest stock first
        batches.sort(key=lambda b: b[1])
        remaining = demand
        for b in batches:
            take = min(b[0], remaining)
            b[0] -= take
            remaining -= take
            if remaining <= 0:
                break
        sold += demand - remaining
        lost += remaining

        expired = sum(b[0] for b in batches if b[1] <= t)     # end of day: remove expired
        wasted += expired
        batches = [b for b in batches if b[1] > t and b[0] > 0]
        log.append((t, on_hand, demand, qty, expired))

    revenue = p["price"] * sold + p["salvage"] * wasted
    profit = revenue - p["cost"] * ordered
    return dict(
        ordered=ordered, sold=sold, wasted=wasted, lost=lost,
        waste_pct=100 * wasted / max(ordered, 1),
        fill_rate=100 * sold / max(sold + lost, 1),
        profit=profit,
    ), pd.DataFrame(log, columns=["day", "on_hand", "demand", "order", "expired"])


def tune_safety_factor(units, dates, train_days, p):
    """Pick the z that earns the most profit in a back-test just before the test period."""
    tune_start = train_days - TUNE_DAYS
    fc = DemandForecaster(units, dates, tune_start)          # sees only data before tune_start
    best_z, best_profit = 0.0, -np.inf
    for z in Z_GRID:
        m, _ = simulate(units, tune_start, optimized_policy(fc, z), p, end=train_days)
        if m["profit"] > best_profit:
            best_z, best_profit = float(z), m["profit"]
    return best_z


# --------------------------------------------------------------------------
# 6. Run everything
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with columns: date, product, units_sold")
    args = ap.parse_args()

    df = load_data(args.csv)
    results, traces = [], {}

    for product, g in df.groupby("product"):
        p = {**DEFAULT_PARAMS, **PRODUCTS.get(product, {})}
        units = g["units_sold"].to_numpy(dtype=float)
        dates = g["date"].reset_index(drop=True)
        train_days = min(TRAIN_DAYS, len(units) - 30)

        z = tune_safety_factor(units, dates, train_days, p)
        fc = DemandForecaster(units, dates, train_days)

        for name, policy in (("Naive", naive_policy()), ("Optimized", optimized_policy(fc, z))):
            metrics, log = simulate(units, train_days, policy, p)
            results.append({"product": product, "policy": name,
                            "z": z if name == "Optimized" else np.nan, **metrics})
            traces[(product, name)] = log

    res = pd.DataFrame(results).round(2)
    res.to_csv("results.csv", index=False)

    print("\n=== Results (simulated test period, unseen days) ===")
    print(res.to_string(index=False))

    summary = res.groupby("policy")[["ordered", "wasted", "lost", "profit"]].sum()
    print("\n=== Totals across all products ===")
    print(summary.round(1).to_string())
    gain = summary.loc["Optimized", "profit"] - summary.loc["Naive", "profit"]
    print(f"\nProfit difference (Optimized - Naive): {gain:,.1f}")

    # ---- charts ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["waste_pct", "fill_rate", "profit"],
                              ["Waste (% of ordered)", "Fill rate (%)", "Profit"]):
        pivot = res.pivot(index="product", columns="policy", values=col)
        pivot[["Naive", "Optimized"]].plot.bar(ax=ax, rot=30)
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    plt.close()

    first = df["product"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in ("Naive", "Optimized"):
        log = traces[(first, name)]
        ax.plot(log["day"], log["on_hand"], label=f"{name} stock")
    ax.plot(log["day"], log["demand"], color="gray", alpha=0.5, label="Demand")
    ax.set_title(f"{first}: stock on hand vs demand")
    ax.set_xlabel("Day")
    ax.legend()
    plt.tight_layout()
    plt.savefig("stock_trace.png", dpi=120)
    plt.close()
    print("\nSaved: results.csv, comparison.png, stock_trace.png")


if __name__ == "__main__":
    main()

"""
simulate_event.py — Hypothetical Rain Event Simulator
======================================================
Produces physically meaningful projections by:
  1. Computing rainfall-to-runoff response curves from historical training data
     (how much does each upstream gauge rise per inch of rain at 6h/12h/24h).
  2. Perturbing both precipitation accumulation features AND upstream gauge
     inputs so the XGBoost models actually respond to the simulated rainfall.
  3. Cascading the effect into lagged features for longer horizons.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_features_from_model():
    """Return the feature list expected by any existing model."""
    for candidate in [
        'model_watts_ok_flow_6h.pkl',
        'model_hwy_16_flow_6h.pkl',
        'model_hwy_59_flow_est_6h.pkl',
    ]:
        if os.path.exists(candidate):
            m = joblib.load(candidate)
            if hasattr(m, 'feature_names_in_'):
                return list(m.feature_names_in_)
    return []


def compute_rain_response_curves(df):
    """
    Analyses historical training data to find: for each inch of rain that fell
    over the basin, how much did the upstream gauges rise at +6h, +12h, +24h?

    Returns
    -------
    dict  { gauge_col: { horizon_hours: ft_or_cfs_per_inch, ... }, ... }
    """
    response = {}
    df = df.ffill().bfill()

    # Use 24-hour accumulated Fayetteville precip as the basin rain signal
    rain_col = 'precip_fayetteville_24h'
    if rain_col not in df.columns:
        # fallback: try single-hour column
        rain_col = 'precip_fayetteville' if 'precip_fayetteville' in df.columns else None
    if rain_col is None:
        return {}

    rain_inches = df[rain_col] / 25.4   # mm → inches

    for gauge in ('savoy_height', 'osage_creek_flow'):
        if gauge not in df.columns:
            continue

        # --- FALLBACK DEFAULTS ---
        # Savoy Height typically rises ~0.5ft per inch of intense rain in this basin.
        # Osage Creek typically surges ~150-200 CFS per inch.
        fallbacks = {
            'savoy_height': {6: 0.15, 12: 0.40, 24: 0.55},
            'osage_creek_flow': {6: 45.0, 12: 120.0, 24: 180.0}
        }

        lag_response = {}
        for horizon in (6, 12, 24):
            # Future change in gauge level relative to the row at which rain is measured
            future_change = df[gauge].shift(-horizon) - df[gauge]

            # Relaxed mask: rain > 0.05" and we only need 5 samples
            mask = (rain_inches > 0.05) & future_change.notna() & rain_inches.notna()
            
            computed_val = 0.0
            if mask.sum() >= 5:
                r = rain_inches[mask].values
                d = future_change[mask].values
                ratios = d / np.where(r > 0, r, np.nan)
                ratios = ratios[np.isfinite(ratios)]
                if len(ratios) >= 5:
                    p15, p85 = np.percentile(ratios, [15, 85])
                    clean = ratios[(ratios >= p15) & (ratios <= p85)]
                    if len(clean) > 0:
                        computed_val = float(np.median(clean))

            # Use fallback if computed value is zero or nonsensical (negative during rain)
            if computed_val <= 0:
                lag_response[horizon] = fallbacks[gauge][horizon]
            else:
                # Blend the two: 50% data-driven, 50% physical fallback for stability
                lag_response[horizon] = (computed_val * 0.5) + (fallbacks[gauge][horizon] * 0.5)

        response[gauge] = lag_response
        print(f"  📐 Response curve for {gauge}: {lag_response}")

    return response


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def simulate_rain(added_inches, verbose=False):
    """
    Simulate the impact of `added_inches` of additional rainfall on all gauges.

    Parameters
    ----------
    added_inches : float  Extra rainfall (inches) spread over the next 24 h.
    verbose      : bool   Print intermediate debug info.

    Returns
    -------
    dict  { gauge_key: { 'projected_6h': float, 'projected_12h': float,
                         'projected_24h': float }, ... }
    """
    if not os.path.exists('master_training_data.csv'):
        return {"error": "master_training_data.csv not found"}

    # ------------------------------------------------------------------ load
    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True)
    df = df.ffill().bfill()
    current_row = df.tail(1).copy()

    features = _load_features_from_model()
    if not features:
        return {"error": "No trained model files found. Run predict_all.py first."}

    added_mm = added_inches * 25.4

    # ------------------------------------------------- rainfall-runoff curves
    response_curves = compute_rain_response_curves(df)

    # ---------------------------------------- gauges and horizons to predict
    gauges   = ['hwy_16_flow', 'hwy_59_flow_est', 'lake_francis_height', 'watts_ok_flow']
    horizons = [6, 12, 24]

    results = {}

    for gauge in gauges:
        results[gauge] = {}
        for h in horizons:
            model_path = f'model_{gauge}_{h}h.pkl'
            if not os.path.exists(model_path):
                if verbose:
                    print(f"  [skip] {model_path} not found")
                continue

            model = joblib.load(model_path)
            model_features = (
                list(model.feature_names_in_)
                if hasattr(model, 'feature_names_in_')
                else features
            )

            # ---- Build a simulation row for this specific horizon ----------
            sim_row = current_row[
                [c for c in current_row.columns if c in model_features]
            ].copy()

            # -- 1. Bump all precipitation accumulation features (mm-based) --
            for col in model_features:
                if 'precip' in col and col in sim_row.columns:
                    sim_row[col] = sim_row[col] + added_mm

            # -- 2. Apply runoff response to upstream gauge inputs -----------
            # Rain falling NOW won't raise Savoy for ~3-6 hours.
            # We scale the effect by how far into the future we're predicting.
            # h=6  → small upstream rise already started
            # h=12 → moderate runoff pulse arriving
            # h=24 → full runoff effect visible
            propagation_scale = {6: 0.35, 12: 0.75, 24: 1.0}
            scale = propagation_scale.get(h, 1.0)

            for upstream_gauge, curve in response_curves.items():
                # Delta at this horizon based on the response curve
                base_delta = curve.get(h, curve.get(24, 0.0))
                delta = base_delta * added_inches * scale

                if verbose:
                    print(f"    Δ{upstream_gauge} at +{h}h = {delta:.3f} "
                          f"(response={base_delta:.3f}, scale={scale})")

                # Perturb the current gauge value
                if upstream_gauge in sim_row.columns:
                    sim_row[upstream_gauge] = sim_row[upstream_gauge] + delta

                # Cascade into lagged features proportionally
                # (the rain event already partially affected earlier readings)
                lag_fractions = {
                    f'{upstream_gauge}_3h_ago':  0.20 * scale,
                    f'{upstream_gauge}_6h_ago':  0.40 * scale,
                    f'{upstream_gauge}_12h_ago': 0.65 * scale if h >= 12 else 0.0,
                    f'{upstream_gauge}_24h_ago': 0.85 * scale if h >= 24 else 0.0,
                }
                for lag_col, frac in lag_fractions.items():
                    if lag_col in sim_row.columns and frac > 0:
                        sim_row[lag_col] = sim_row[lag_col] + base_delta * added_inches * frac

                # Perturb trend features
                for trend_col in [
                    f'{upstream_gauge}_trend_6h',
                    f'{upstream_gauge}_trend_24h',
                ]:
                    if trend_col in sim_row.columns:
                        sim_row[trend_col] = sim_row[trend_col] + delta * 0.5

            # -- 3. Predict -----------------------------------------------
            try:
                # Make sure columns are in the exact order the model expects
                input_df = pd.DataFrame(
                    [sim_row.reindex(columns=model_features).iloc[0].values],
                    columns=model_features
                ).fillna(0)

                pred = model.predict(input_df)[0]
                results[gauge][f'projected_{h}h'] = round(float(pred), 2)

                if verbose:
                    print(f"  ✅ {gauge} +{h}h ({added_inches:.1f}\"): {pred:.2f}")
            except Exception as e:
                if verbose:
                    print(f"  ❌ {gauge} +{h}h error: {e}")
                results[gauge][f'projected_{h}h'] = None

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import xgboost as xgb  # ensure xgb is available when loaded via joblib

    if len(sys.argv) > 1:
        try:
            inches = float(sys.argv[1])
            print(f"\n🌧  Simulating {inches}\" of additional rainfall...\n")
            result = simulate_rain(inches, verbose=True)
            print("\n📊 Results:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
    else:
        print("Usage: python simulate_event.py <inches_of_rain>")
        print("Example: python simulate_event.py 1.5")

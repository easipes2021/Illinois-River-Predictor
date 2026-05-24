"""
Accuracy Tracker for the Illinois River Predictor.

Three jobs:
  1. log_predictions()     – Append current forecasts to prediction_log.json
  2. score_past_predictions() – Compare old predictions against actual observed values
  3. generate_accuracy_scores() – Write accuracy_scores.json for the dashboard
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd

LOG_FILE = 'prediction_log.json'
SCORES_FILE = 'accuracy_scores.json'
TRAINING_DATA = 'master_training_data.csv'
MAX_LOG_AGE_DAYS = 30

GAUGE_NAMES = {
    'hwy_16_flow': 'Hwy 16 (Siloam)',
    'hwy_59_flow_est': 'Hwy 59 (AR Bridge)',
    'lake_francis_height': 'Lake Francis Level',
    'watts_ok_flow': 'Watts Bridge (OK)'
}

HORIZONS = [6, 12, 24]


def _load_log():
    """Load the prediction log from disk."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("⚠️  Corrupted prediction_log.json — starting fresh.")
            return []
    return []


def _save_log(entries):
    """Save the prediction log to disk."""
    with open(LOG_FILE, 'w') as f:
        json.dump(entries, f, indent=2)


def _prune_old_entries(entries):
    """Remove entries older than MAX_LOG_AGE_DAYS."""
    cutoff = (datetime.utcnow() - timedelta(days=MAX_LOG_AGE_DAYS)).isoformat()
    return [e for e in entries if e.get('prediction_made_at', '') >= cutoff]


# ──────────────────────────────────────────────
# Job 1: Log Predictions
# ──────────────────────────────────────────────

def log_predictions(forecast_results):
    """
    Append current predictions to the rolling log.
    
    Args:
        forecast_results: The dict written to forecasts.json by forecast_now.py
    """
    entries = _load_log()

    made_at = forecast_results.get('timestamp')
    if not made_at:
        print("⚠️  No timestamp in forecast_results — skipping log.")
        return

    for gauge_key in GAUGE_NAMES:
        gauge_data = forecast_results.get(gauge_key)
        if not gauge_data:
            continue

        current_val = gauge_data.get('current')

        for h in HORIZONS:
            projected = gauge_data.get(f'projected_{h}h')
            if projected is None:
                continue

            # Calculate the target time from the prediction timestamp
            try:
                base_dt = datetime.fromisoformat(made_at)
                target_dt = base_dt + timedelta(hours=h)
                target_iso = target_dt.isoformat()
            except (ValueError, TypeError):
                continue

            entry = {
                'prediction_made_at': made_at,
                'gauge': gauge_key,
                'horizon': f'{h}h',
                'forecast_target_time': target_iso,
                'predicted_value': projected,
                'current_value_at_prediction': current_val,
                'actual_value': None,
                'absolute_error': None,
                'percentage_error': None,
                'scored': False
            }
            entries.append(entry)

    # Prune old entries
    entries = _prune_old_entries(entries)
    _save_log(entries)
    print(f"📝 Logged {len(GAUGE_NAMES) * len(HORIZONS)} predictions to {LOG_FILE}")


# ──────────────────────────────────────────────
# Job 2: Score Past Predictions
# ──────────────────────────────────────────────

def score_past_predictions():
    """
    Scan unscored entries whose target time has passed,
    look up the actual value, and compute errors.
    """
    entries = _load_log()
    if not entries:
        print("📊 No predictions to score yet.")
        return

    if not os.path.exists(TRAINING_DATA):
        print("⚠️  master_training_data.csv not found — cannot score predictions.")
        return

    df = pd.read_csv(TRAINING_DATA, index_col=0, parse_dates=True)
    if df.empty:
        print("⚠️  master_training_data.csv is empty — cannot score.")
        return

    now_utc = datetime.utcnow()
    scored_count = 0

    for entry in entries:
        if entry.get('scored'):
            continue

        # Check if the forecast target time has passed
        try:
            raw_target = entry['forecast_target_time']
            # Strip timezone info to get naive datetime matching the CSV index
            raw_target = raw_target.replace('+00:00', '').replace('Z', '')
            # Handle offset like +05:00 etc
            if '+' in raw_target:
                raw_target = raw_target[:raw_target.rfind('+')]
            target_time = datetime.fromisoformat(raw_target)
        except (ValueError, KeyError):
            continue

        if target_time > now_utc:
            continue  # Target time hasn't arrived yet

        gauge_key = entry.get('gauge')
        if gauge_key not in df.columns:
            continue

        # Find the closest observation to the target time
        # Allow a 1-hour tolerance window
        time_diff = abs(df.index - target_time)
        min_diff_idx = time_diff.argmin()
        min_diff_seconds = time_diff[min_diff_idx].total_seconds()

        if min_diff_seconds > 3600:  # More than 1 hour off — no close match
            continue

        actual = df.iloc[min_diff_idx][gauge_key]
        if pd.isna(actual):
            continue

        predicted = entry['predicted_value']
        actual = float(actual)
        abs_error = abs(predicted - actual)
        pct_error = (abs_error / actual * 100) if actual != 0 else 0.0

        entry['actual_value'] = round(actual, 2)
        entry['absolute_error'] = round(abs_error, 2)
        entry['percentage_error'] = round(pct_error, 2)
        entry['scored'] = True
        scored_count += 1

    _save_log(entries)
    total_scored = sum(1 for e in entries if e.get('scored'))
    print(f"📊 Scored {scored_count} new predictions ({total_scored} total scored in log)")


# ──────────────────────────────────────────────
# Job 3: Generate Accuracy Scores
# ──────────────────────────────────────────────

def _compute_confidence(mape):
    """Convert MAPE to a 0-100 confidence score."""
    return max(0, min(100, round(100 - (mape * 5))))


def _confidence_label(score):
    """Map score to a human-readable label."""
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Good'
    elif score >= 50:
        return 'Fair'
    else:
        return 'Poor'


def _compute_trend(errors):
    """Compare recent errors to prior errors to determine trend."""
    if len(errors) < 20:
        return 'insufficient_data'

    recent_10 = errors[-10:]
    prior_10 = errors[-20:-10]

    recent_avg = sum(recent_10) / len(recent_10)
    prior_avg = sum(prior_10) / len(prior_10)

    diff = recent_avg - prior_avg
    if diff < -1:
        return 'improving'
    elif diff > 1:
        return 'declining'
    else:
        return 'stable'


def generate_accuracy_scores():
    """
    Read the scored prediction log and generate accuracy_scores.json
    for the dashboard to consume.
    """
    entries = _load_log()
    scored = [e for e in entries if e.get('scored')]

    result = {
        'last_updated': datetime.utcnow().isoformat() + '+00:00',
        'total_predictions_logged': len(entries),
        'total_scored': len(scored),
        'min_required': 10,
        'gauges': {}
    }

    for gauge_key, gauge_name in GAUGE_NAMES.items():
        gauge_entries = [e for e in scored if e['gauge'] == gauge_key]

        horizons_data = {}
        all_pct_errors = []

        for h in HORIZONS:
            h_key = f'{h}h'
            h_entries = [e for e in gauge_entries if e['horizon'] == h_key]

            # Use last 48 scored predictions for metrics
            recent = h_entries[-48:] if len(h_entries) > 48 else h_entries

            if len(recent) < 10:
                horizons_data[h_key] = {
                    'mae': None,
                    'mape': None,
                    'confidence_score': None,
                    'confidence_label': 'Collecting data',
                    'sample_count': len(recent),
                    'recent_trend': 'insufficient_data',
                    'last_10_errors': [e.get('percentage_error', 0) for e in recent[-10:]]
                }
                continue

            abs_errors = [e['absolute_error'] for e in recent if e.get('absolute_error') is not None]
            pct_errors = [e['percentage_error'] for e in recent if e.get('percentage_error') is not None]

            mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
            mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0
            score = _compute_confidence(mape)
            trend = _compute_trend(pct_errors)

            all_pct_errors.extend(pct_errors)

            horizons_data[h_key] = {
                'mae': round(mae, 2),
                'mape': round(mape, 2),
                'confidence_score': score,
                'confidence_label': _confidence_label(score),
                'sample_count': len(recent),
                'recent_trend': trend,
                'last_10_errors': [round(e, 2) for e in pct_errors[-10:]]
            }

        # Overall confidence = weighted average (6h weighted most)
        scored_horizons = {k: v for k, v in horizons_data.items() if v.get('confidence_score') is not None}
        if scored_horizons:
            weights = {'6h': 3, '12h': 2, '24h': 1}
            total_weight = sum(weights.get(k, 1) for k in scored_horizons)
            overall = sum(v['confidence_score'] * weights.get(k, 1) for k, v in scored_horizons.items()) / total_weight
            overall = round(overall)
        else:
            overall = None

        # Recent predictions for the mini-chart (last 10 per horizon)
        recent_predictions = []
        for e in gauge_entries[-30:]:
            recent_predictions.append({
                'target_time': e.get('forecast_target_time'),
                'horizon': e.get('horizon'),
                'predicted': e.get('predicted_value'),
                'actual': e.get('actual_value')
            })

        result['gauges'][gauge_key] = {
            'name': gauge_name,
            'horizons': horizons_data,
            'overall_confidence': overall,
            'overall_label': _confidence_label(overall) if overall is not None else 'Collecting data',
            'recent_predictions': recent_predictions
        }

    with open(SCORES_FILE, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Accuracy scores written to {SCORES_FILE}")


# ──────────────────────────────────────────────
# Convenience: Run all three jobs
# ──────────────────────────────────────────────

def run_full_accuracy_pipeline(forecast_results):
    """Run all three accuracy tracking jobs in sequence."""
    log_predictions(forecast_results)
    score_past_predictions()
    generate_accuracy_scores()


if __name__ == '__main__':
    # When run standalone, just score and generate (don't log — no new forecast)
    print("🔍 Running accuracy scoring on existing log...")
    score_past_predictions()
    generate_accuracy_scores()

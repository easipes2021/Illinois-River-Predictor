#!/usr/bin/env python3
"""Generate a daily accuracy report for the Illinois River Predictor."""

import json
import os
from datetime import datetime
from collections import defaultdict

def grade_emoji(label):
    """Return emoji for a confidence label."""
    return {
        'Excellent': '🟢',
        'Good': '🟢',
        'Fair': '🟡',
        'Poor': '🔴',
        'Collecting data': '⏳',
    }.get(label, '⚪')

def trend_arrow(early, recent):
    """Return trend arrow comparing early vs recent errors."""
    delta = recent - early
    if delta < -2:
        return '↗ improving'
    elif delta > 2:
        return '↘ declining'
    else:
        return '→ stable'

def analyze_model(scores_file, log_file, model_name):
    """Analyze a model's accuracy scores and prediction log."""
    with open(scores_file) as f:
        scores = json.load(f)
    
    with open(log_file) as f:
        log = json.load(f)
    
    scored = [e for e in log if e.get('scored')]
    groups = defaultdict(list)
    for e in scored:
        groups[(e['gauge'], e['horizon'])].append(e['percentage_error'])
    
    lines = []
    lines.append(f"### {model_name}")
    lines.append(f"- Predictions Logged: **{scores['total_predictions_logged']}** | Scored: **{scores['total_scored']}**")
    lines.append(f"- Last Updated: {scores['last_updated'][:19]} UTC")
    lines.append("")
    
    # Performance table
    lines.append("| Gauge | 6h | 12h | 24h |")
    lines.append("|---|---|---|---|")
    
    for gauge_key, info in scores['gauges'].items():
        name = info['name']
        cells = []
        for hz in ['6h', '12h', '24h']:
            hd = info['horizons'].get(hz, {})
            mape = hd.get('mape')
            label = hd.get('confidence_label', 'N/A')
            emoji = grade_emoji(label)
            if mape is not None:
                cells.append(f"{emoji} {mape:.1f}%")
            else:
                cells.append(f"⏳ TBD")
        lines.append(f"| **{name}** | {cells[0]} | {cells[1]} | {cells[2]} |")
    
    lines.append("")
    
    # Trend table
    lines.append("**Trend (Early 10 → Recent 10):**")
    lines.append("")
    lines.append("| Gauge | 6h | 12h | 24h |")
    lines.append("|---|---|---|---|")
    
    gauge_names = {k: v['name'] for k, v in scores['gauges'].items()}
    for gauge_key in scores['gauges']:
        name = gauge_names[gauge_key]
        cells = []
        for hz in ['6h', '12h', '24h']:
            errs = groups.get((gauge_key, hz), [])
            if len(errs) >= 10:
                early = sum(errs[:10]) / 10
                recent = sum(errs[-10:]) / 10
                arrow = trend_arrow(early, recent)
                cells.append(f"{early:.1f}% → {recent:.1f}% {arrow}")
            elif len(errs) > 0:
                avg = sum(errs) / len(errs)
                cells.append(f"{avg:.1f}% (n={len(errs)})")
            else:
                cells.append("—")
        lines.append(f"| **{name}** | {cells[0]} | {cells[1]} | {cells[2]} |")
    
    lines.append("")
    return lines, scores, groups

def generate_report():
    """Generate the full daily report as markdown."""
    now = datetime.utcnow()
    date_str = now.strftime("%B %d, %Y")
    
    report = []
    report.append(f"# 🌊 Illinois River Predictor — Daily Report")
    report.append(f"**Date:** {date_str}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Main Model
    main_lines, main_scores, main_groups = analyze_model(
        'accuracy_scores.json', 'prediction_log.json', '📊 Main Model'
    )
    report.extend(main_lines)
    
    report.append("---")
    report.append("")
    
    # Beta Model
    beta_lines, beta_scores, beta_groups = analyze_model(
        'accuracy_scores_beta.json', 'prediction_log_beta.json', '🧪 Beta Model (Watershed-Enhanced)'
    )
    report.extend(beta_lines)
    
    report.append("---")
    report.append("")
    
    # Head-to-head at 6h
    report.append("### 🏆 Head-to-Head: Main vs Beta (6h)")
    report.append("")
    report.append("| Gauge | Main | Beta | Winner |")
    report.append("|---|---|---|---|")
    
    for gauge_key in main_scores['gauges']:
        name = main_scores['gauges'][gauge_key]['name']
        main_mape = main_scores['gauges'][gauge_key]['horizons'].get('6h', {}).get('mape')
        beta_mape = beta_scores['gauges'].get(gauge_key, {}).get('horizons', {}).get('6h', {}).get('mape')
        
        if main_mape is not None and beta_mape is not None:
            if main_mape < beta_mape:
                winner = "🏆 Main"
                m_str = f"**{main_mape:.1f}%**"
                b_str = f"{beta_mape:.1f}%"
            else:
                winner = "🏆 Beta"
                m_str = f"{main_mape:.1f}%"
                b_str = f"**{beta_mape:.1f}%**"
            report.append(f"| **{name}** | {m_str} | {b_str} | {winner} |")
        else:
            report.append(f"| **{name}** | {'N/A' if main_mape is None else f'{main_mape:.1f}%'} | {'N/A' if beta_mape is None else f'{beta_mape:.1f}%'} | — |")
    
    report.append("")
    report.append("---")
    report.append("")
    
    # Alerts
    report.append("### ⚠️ Alerts")
    report.append("")
    
    alerts = []
    for gauge_key, info in main_scores['gauges'].items():
        name = info['name']
        for hz in ['6h', '12h', '24h']:
            errs = main_groups.get((gauge_key, hz), [])
            if len(errs) >= 10:
                early = sum(errs[:10]) / 10
                recent = sum(errs[-10:]) / 10
                # Alert if 6h is declining significantly
                if hz == '6h' and recent > early + 5:
                    alerts.append(f"⚠️ **{name} 6h declining:** {early:.1f}% → {recent:.1f}% — monitor closely")
                # Alert if any horizon has very high recent error
                if recent > 100:
                    alerts.append(f"🔴 **{name} {hz} very high error:** recent avg {recent:.1f}%")
    
    if alerts:
        for a in alerts:
            report.append(f"- {a}")
    else:
        report.append("- ✅ No critical alerts. Models are operating within expected parameters.")
    
    report.append("")
    report.append("---")
    report.append(f"*Auto-generated by Illinois River Predictor at {now.strftime('%Y-%m-%d %H:%M')} UTC*")
    
    return '\n'.join(report)


if __name__ == '__main__':
    report = generate_report()
    
    # Write to file for the GitHub Action to pick up
    with open('daily_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Daily report generated: daily_report.md")
    print(report)

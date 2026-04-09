# 30-Day Excel Integration TODO

Completed: ✅ | Pending: ⏳

## Plan Implementation Steps

✅ **Step 1: Create this TODO.md** - Tracking progress

✅ **Step 2: Update CSS/Styling** ✓

- Complete source CSS integrated

✅ **Step 3: Enhance 30-Day Gurobi Loop** ✓

- Random baseline + per-sev tracking + full daily_results

✅ **Step 4: Add Savings Banner & KPI Row** ✓

- Aggregates + banner + 6 KPIs added

✅ **Step 5: Chart Row 1 - Daily Performance** ✓

- Col1/C2: Daily jobs + backlog charts added

✅ **Step 6: Per-Severity Backlog Charts** ✓

- Col1: Stacked daily backlog bar
- Col2: Donut pie + summary table

⏳ **Step 6: Per-Severity Backlog Charts**

- Col1: Daily stacked backlog bar (Critical→Very Low)
- Col2: 30-day donut pie + per-sev summary table

⏳ **Step 7: Fibre vs Copper Split Bar**

- Horizontal grouped bar: 30-day backlog per sev (fibre/blue vs copper/green)

⏳ **Step 8: Demand & Fulfillment Charts**

- Col1: Daily demand line vs stacked fibre/copper completed
- Col2: Monthly fibre/copper job mix pie

⏳ **Step 9: Cost Savings Charts**

- Col1: Cumulative savings line + annotation
- Col2: Savings breakdown table

⏳ **Step 10: Backlog Age & Insights**

- Col1: Backlog age stacked bar (new/day-1/day-2+)
- Col2: Dynamic insights cards (high backlog alerts, util warnings, etc.)

⏳ **Step 11: Daily Detail Table**

- Expandable styled table (17 cols: day, demands, jobs, extra, backlog w/ sev badges, util, daily saving)

⏳ **Step 12: Test with Sample Excel**

- `streamlit run technician_forecast_final_complete_fixed.py`
- Upload `~$FINAL_forecast_results_with_5_new_cols.xlsx`
- Verify: all charts render, savings calculate, per-sev tracking, Quick tab unchanged

⏳ **Step 13: attempt_completion**

- Confirm Quick tab preserved, 30-day now matches source features

**Notes:**

- Quick Manual tab: UNCHANGED
- Test file: ~$FINAL_forecast_results_with_5_new_cols.xlsx
- Single file: technician_forecast_final_complete_fixed.py

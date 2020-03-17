# Subjects to analyze
subjects=(6 10 11 12 13 15 16 19 20 21 23 27 28 29 34 37 41 42 44 45 46 47 48 49 51 52 53 54 57 99)

# Compute time-series
python compute_time_series.py "${subjects[@]}"

# Compute instabilities
python compute_instabilities.py "${subjects[@]}"

# Statistical analysis
ipython statistical_analysis.py "${subjects[@]}"

"""The gate: picks which checked findings become review comments, and measures the result.

policy.py holds the rules (tier, confidence, one comment per place, a cap per PR), curve.py
estimates precision and recall at every cutoff from a scored ungated run, and commands.py adds
`pr-reviewer gate ...`.
"""

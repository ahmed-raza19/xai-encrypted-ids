"""xaiids: measuring the fidelity and computational cost of post-hoc explanations
for intrusion detection on encryption-constrained (metadata-only) traffic features.

Module map (Phase 2 methodology step -> module):
  Steps 2-4  data collection, preprocessing, visibility tiers  -> data.py, tiers.py
  Step 5/6   detectors and explainers                          -> models.py, explainers.py
  Step 5     budget-aware selective XAI trigger                -> trigger.py, stream.py
  Step 8     detection / fidelity / consistency / cost metrics -> metrics.py, profiling.py
  Step 9     statistical analysis                              -> stats.py
"""

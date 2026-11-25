# Results

Each experiment writes a folder:

```
results/<experiment_name>/
  config.yaml      # frozen copy of the run config
  metrics.json     # accuracy, macro-F1, per-class F1
  history.json     # train/val curves
  confusion.png    # optional, after evaluation
```

Do not commit large checkpoints. Commit `metrics.json` once runs are final.

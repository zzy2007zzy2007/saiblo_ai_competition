# Checklist

- [ ] `collect_data.py` runs end-to-end: produces `.npz` with correct shapes
- [ ] `train_bc.py` trains successfully and outputs a valid BC checkpoint
- [ ] BC checkpoint loads into ES trainer via `--load-bc`
- [ ] `--load-bc` correctly initializes `self.mean` and resets velocity

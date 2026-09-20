# bayesian_die_swell

Load observations before building the ROM:

```python
model.load_data(filename="path/to/curve4_y.txt", u_avg_obs=0.1, x_max=5.0)
model.build_rom()
```

The default `x_max=5.0` keeps only points with `x <= 5` for plotting,
inference, and ROM training. The mask is applied before noise calibration
and before the ROM's POD/GPR fit. Observation thinning is applied within
the retained range. Training curves must share the observation file's
spatial columns. Set `x_max=None` for the full curve; rebuild the ROM after
changing the retained range.

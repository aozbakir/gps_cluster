# TODO

## Algorithm

- [ ] Increase `n_restarts` to 3,000 to fully reproduce Savage (2018) — currently 100 for practical runtime
- [ ] Track down the missing 469th station (paper has 469, our CSV has 468)
- [ ] Implement the collinearity / error-ellipsoid overlap criterion for optimal k (paper Figs 3, 4)
- [ ] Implement cluster consolidation step (merge contiguous clusters with similar Euler vectors — paper Fig 5)
- [ ] Add uncertainty estimation for Euler poles (95% confidence ellipsoids)

## Figures

- [ ] Add linear fit (red dashed line) to ω-space Fig 8, matching paper Figs 3 & 4
- [ ] Add 95% confidence ellipsoids to ω-space plot
- [ ] Add block-boundary overlays to Fig 7 (CMTL, NKTZ, OKTL lines)
- [ ] Add relative velocity arrows across plate boundaries (paper Fig 7)
- [ ] Fig 5: compare Euler pole locations with paper's reported values

## Infrastructure

- [ ] Add GitHub remote and configure CI push
- [ ] Add `requirements.txt` export for conda-incompatible environments
- [ ] Pin cartopy version in `viz` extras (API changes between 0.21 and 0.23)

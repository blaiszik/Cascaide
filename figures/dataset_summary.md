# Cascaide dataset coverage

- **7,100 cascades**, Tungsten (W), 1073 K, minimized post-cascade defect structures
- PKA energy **0.014–299.8 keV** (continuous), Frenkel pairs **0–1107** (mean 97.9)
- N_vac == N_SIA for every cascade (0 Frenkel-conservation violations); total point defects = 2x pairs
- supercell scales with energy (sc50 → sc280) so the box contains the cascade
- power-law fit (E>1 keV): N_pairs ~ E^0.90  (log-log r=0.946)

| energy campaign | cascades | supercell | min | median | mean | max |
|---|---|---|---|---|---|---|
| 0-10keV | 600 | sc50 | 0 | 7 | 7.2 | 21 |
| 10-30keV | 600 | sc64 | 3 | 16 | 16.8 | 39 |
| 30-50keV | 600 | sc80 | 9 | 28 | 28.5 | 61 |
| 50-70keV | 600 | sc100 | 17 | 42 | 42.6 | 91 |
| 70-100keV | 900 | sc120 | 16 | 62 | 63.7 | 129 |
| 100-150keV | 1500 | sc160 | 37 | 96 | 101.4 | 454 |
| 150-200keV | 1500 | sc200 | 52 | 145 | 156.9 | 533 |
| 200-250keV | 400 | sc240 | 109 | 192 | 209.1 | 520 |
| 250-300keV | 400 | sc280 | 141 | 246 | 273.7 | 1107 |

# ML Assignment 2 — Reproducible experiment scaffold

Student: Abhishek Kapar  
Roll number: M26ecm511  
Random seed: `26511` (numeric characters of roll number, in order)

## Run
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_experiments.py
```

The first run downloads the Olivetti Faces dataset through scikit-learn. The script saves the split to `outputs/split.json` and reuses it on subsequent runs. Results and plots are written to `outputs/` and `outputs/figures/`.

## Important notes before submission
- Run the script and inspect every JSON result. Do not write numerical claims until they match the generated files.
- The assignment's Q2 asks for a plot of weighted training error, learner weights, maximum sample weight and total weight carried by the 10 heaviest training images over rounds. Extend the SAMME loop to log those per-round quantities if your instructor expects all four traces explicitly; the current script records learner errors and weights but not the complete per-round sample-weight history.
- Inspect `q1d_clean_masked.json`: the current vote-count diagnostic is stored for the ensemble; the tree itself has a single predicted class. For a strong report, print the actual tree class and the bagging vote histogram for each selected example.
- Q3 follows the stated resampling scheme: within each identity, sample six entries with replacement from that identity's six original training examples. The clean test set remains fixed.
- Verify all package versions and exact reproduction commands before committing. Commit your final report, scripts, split, configurations, logs and figures, then copy the real commit hash to page 1 of the PDF.
- This is a scaffold for reproducible experiments, not a claim that the assignment has already been executed or that its measured results are known.

# ML Assignment 2: Bagging and Boosting

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Seed](https://img.shields.io/badge/Random%20Seed-26511-green)

| | |
|---|---|
| **Student** | Abhishek Kapar |
| **Roll Number** | M26ECM511 |
| **Random Seed** | 26511 |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Main Files](#main-files)
4. [Getting Started](#getting-started)
5. [Results and Report](#results-and-report)
6. [Reproducibility](#reproducibility)

---

## Project Overview

This project studies **decision tree classification**, **bagging**, **SAMME boosting**, and **bias–variance analysis** using the **Olivetti Faces** dataset.

The experiments are implemented in Python using scikit-learn and related libraries. Results are saved as JSON files, and plots are generated to compare model performance.

---

## Repository Structure

```text
ML-Assignment-2-Bagging-Boosting/
├── README.md
├── config.json
├── run_experiments.py
├── make_report.py
├── requirements.txt
├── reproduce.sh
├── RUN_IN_COLAB.md
├── report_template.md
└── outputs/
    ├── report.pdf
    ├── report.tex
    ├── split.json
    ├── q1a_tree_grid.json
    ├── q1b_bagging.json
    ├── q1c_ensemble_diagnostics.json
    ├── q1d_clean_masked.json
    ├── q2a_samme_toy.json
    ├── q2b_boosting_grid.json
    ├── q2c_comparison.json
    ├── q2d_clean_masked.json
    ├── q3_bias_variance.json
    └── figures/
        ├── q1_disagreement.png
        ├── q1_oob_accuracy.png
        ├── q1_vote_margin.png
        ├── q2_learner_weights.png
        ├── q2_max_sample_weight.png
        ├── q2_top10_weight_mass.png
        ├── q2_training_accuracy.png
        ├── q2_weighted_training_error.png
        ├── q3_bias_squared.png
        └── q3_variance.png
```

---

## Main Files

| File | Purpose |
|------|---------|
| [`run_experiments.py`](run_experiments.py) | Runs the experiments and generates results and plots |
| [`config.json`](config.json) | Stores the experiment configuration |
| [`requirements.txt`](requirements.txt) | Lists the required Python packages |
| [`make_report.py`](make_report.py) | Helps generate the report |
| [`reproduce.sh`](reproduce.sh) | Provides commands to reproduce the experiments |
| [`RUN_IN_COLAB.md`](RUN_IN_COLAB.md) | Explains how to run the project in Google Colab |
| [`report_template.md`](report_template.md) | Provides a report template |
| [`outputs/report.pdf`](outputs/report.pdf) | Final assignment report |
| [`outputs/`](outputs/) | Experimental results in JSON format |
| [`outputs/figures/`](outputs/figures/) | Generated plots |

---

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 4. Run the experiments

```bash
python run_experiments.py
```

> **Note:** The first run downloads the Olivetti Faces dataset if needed. The script then saves results and generated plots to the configured output directories.

**Using Google Colab?** See [`RUN_IN_COLAB.md`](RUN_IN_COLAB.md).

---

## Results and Report

The repository contains the JSON results, generated plots, and the final PDF report. These files can be inspected to review the experimental setup, numerical results, and model comparisons.

📄 **Final report:** [`outputs/report.pdf`](outputs/report.pdf)

---

## Reproducibility

The project uses a fixed random seed of **`26511`**. Install the dependencies and follow the steps in [Getting Started](#getting-started) to reproduce the experiments.

> Results may vary slightly across software versions or environments.

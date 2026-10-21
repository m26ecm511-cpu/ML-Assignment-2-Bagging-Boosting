ML Assignment 2: Bagging and Boosting

Student: Abhishek Kapar
Roll Number: M26ecm511
Random Seed: 26511

---

1. Project Overview

This project studies decision tree classification, bagging, SAMME boosting, and bias–variance analysis using the Olivetti Faces dataset.

The experiments are implemented in Python using scikit-learn and related libraries. The results are saved as JSON files, and plots are generated to compare model performance.

2. Repository Structure

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

3. Main Files

File| Purpose
"run_experiments.py"| Runs the experiments and generates results and plots.
"config.json"| Stores the experiment configuration.
"requirements.txt"| Lists the required Python packages.
"make_report.py"| Helps generate the report.
"reproduce.sh"| Provides commands to reproduce the experiments.
"RUN_IN_COLAB.md"| Explains how to run the project in Google Colab.
"report_template.md"| Provides a report template.
"outputs/report.pdf"| Contains the final assignment report.
"outputs/"| Stores experimental results in JSON format.
"outputs/figures/"| Stores the generated plots.

4. How to Run the Project

Step 1: Create a virtual environment

python -m venv .venv

Step 2: Activate the environment

Windows:

.venv\Scripts\activate

Linux or macOS:

source .venv/bin/activate

Step 3: Install the required packages

python -m pip install -r requirements.txt

Step 4: Run the experiments

python run_experiments.py

The first run downloads the Olivetti Faces dataset if needed. The script then saves experimental results and generated plots in the configured output directories.

For Google Colab instructions, see ""RUN_IN_COLAB.md"" (RUN_IN_COLAB.md).

5. Results and Report

The repository contains the JSON results, generated plots, and final PDF report. These files can be inspected to review the experimental setup, numerical results, and model comparisons.

Final report: ""outputs/report.pdf"" (outputs/report.pdf)

6. Reproducibility

The project uses a fixed random seed of "26511". Install the dependencies and follow the run instructions above to reproduce the experiments. Results may vary slightly across software versions or environments.

ML Assignment 2: Bagging and Boosting

---

Student Information

Detail| Information
Name| Abhishek Kapar
Roll Number| M26ecm511
Random Seed| 26511

---

1. About the Assignment

This project implements and evaluates decision tree classification, bagging, SAMME boosting, and bias–variance analysis using the Olivetti Faces dataset.

The experiments are implemented in Python using scikit-learn and related libraries. The results are saved as JSON files, and the generated plots are used to compare model performance.

2. Repository Structure

ML-Assignment-2-Bagging-Boosting/
│
├── README.md
├── config.json
├── run_experiments.py
├── make_report.py
├── requirements.txt
├── reproduce.sh
├── RUN_IN_COLAB.md
├── report_template.md
│
└── outputs/
    ├── report.pdf
    ├── report.tex
    ├── split.json
    ├── [experiment result JSON files]
    │
    └── figures/
        ├── README.md
        └── [generated PNG plots]

3. Files and Their Purpose

File or folder| Purpose
"run_experiments.py"| Runs the experiments
"config.json"| Stores experiment configuration
"requirements.txt"| Lists Python dependencies
"make_report.py"| Helps generate the report
"reproduce.sh"| Provides a reproduction script
"RUN_IN_COLAB.md"| Explains how to run the project in Google Colab
"outputs/"| Stores the report and experimental results
"outputs/figures/"| Stores the generated plots

4. How to Run

Step 1: Create a virtual environment

python -m venv .venv

Step 2: Activate it

Windows:

.venv\Scripts\activate

Step 3: Install the dependencies

python -m pip install -r requirements.txt

Step 4: Run the experiments

python run_experiments.py

The first run downloads the Olivetti Faces dataset if required. The script saves experiment results and generated plots in the output directories configured by the project.

5. Results and Report

The experiment results are stored as JSON files. The plots provide visual comparisons of the models and their behaviour.

Final report: "View the PDF report" (outputs/report.pdf)

- "Experiment results and figures" (outputs/)
- "Generated plots" (outputs/figures/)

6. Reproducibility

The project uses a fixed random seed to support reproducible experiments. Install the listed dependencies and run the experiment script using the steps above.

For the Google Colab workflow, refer to "RUN_IN_COLAB.md".

---

End of README

# Run the assignment in Google Colab (if local dataset download is blocked)

1. Open https://colab.research.google.com/ and create a new Python notebook.
2. Upload `ML_Assignment2_Reproducible_Starter.zip` in the Files panel.
3. Run this cell:

```python
!unzip -q ML_Assignment2_Reproducible_Starter.zip -d ml_assignment2
%cd ml_assignment2
!python -m pip install -r requirements.txt
!python run_experiments.py
!python make_report.py
```

4. Download the complete generated output folder and project files as a ZIP:

```python
%cd /content
!zip -qr ML_Assignment2_results.zip ml_assignment2
from google.colab import files
files.download('/content/ML_Assignment2_results.zip')
```

The first run needs internet access to download the official scikit-learn Olivetti Faces dataset. The experiments may take several minutes because Q3 fits many trees. Keep `outputs/split.json` unchanged between runs. The report builder requires a LaTeX installation; if Colab does not have `pdflatex`, it will generate `outputs/report.tex`, which can be compiled in Overleaf after uploading the `outputs/figures` directory and changing the figure paths as needed.

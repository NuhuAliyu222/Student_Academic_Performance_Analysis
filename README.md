# Student Academic Performance Analysis

A Python exploratory data analysis project using Pandas, NumPy, Matplotlib and Seaborn.

## Dataset disclosure
The included `student_performance_demo.csv` is synthetic demonstration data. It does not contain real student records and does not represent a real school or population. Use a properly sourced dataset for real-world conclusions.

## Project contents
- `data/`: synthetic student dataset
- `notebooks/`: step-by-step Jupyter Notebook
- `src/`: reusable analysis script
- `visualizations/`: generated charts (created when you run the analysis)

## Setup with Anaconda
```bash
conda create -n student-analysis python=3.12 -y
conda activate student-analysis
pip install -r requirements.txt
jupyter notebook
```
Open `notebooks/Student_Academic_Performance_Analysis.ipynb` and run all cells.

To run the script:
```bash
python src/analyze_student_performance.py
```

## Skills demonstrated
Data inspection, missing-value treatment, duplicate removal, descriptive statistics, grouping, score-band creation, exploratory analysis, correlation and visualisation.

## Interpretation reminder
Correlation does not prove causation. Explain the limitations of the data and avoid drawing conclusions about real students from this synthetic sample.

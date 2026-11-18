from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance_demo.csv"
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Original shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
numeric_cols = ["study_time_hours_per_week", "attendance_percent", "absences", "previous_score", "final_score"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
for col in ["tutoring", "internet_access", "parent_education", "gender"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df["performance_band"] = pd.cut(df["final_score"], bins=[-0.1, 49.9, 69.9, 84.9, 100],
    labels=["Needs support", "Satisfactory", "Good", "Excellent"])

group_summary = df.groupby("gender", observed=False)["final_score"].agg(["mean", "median", "count"])
group_summary.to_csv(OUT / "score_summary_by_gender.csv")
print("\nOverall score summary:\n", df["final_score"].describe())
print("\nPerformance bands:\n", df["performance_band"].value_counts())
print("\nAverage scores by gender:\n", group_summary.round(2))

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["final_score"], bins=20)
ax.set_title("Distribution of Final Scores")
ax.set_xlabel("Final score")
ax.set_ylabel("Number of students")
fig.tight_layout()
fig.savefig(OUT / "final_score_distribution.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["study_time_hours_per_week"], df["final_score"], alpha=0.4)
coef = np.polyfit(df["study_time_hours_per_week"], df["final_score"], 1)
x_line = np.linspace(df["study_time_hours_per_week"].min(), df["study_time_hours_per_week"].max(), 100)
ax.plot(x_line, np.polyval(coef, x_line))
ax.set_title("Study Time and Final Score")
ax.set_xlabel("Study time (hours per week)")
ax.set_ylabel("Final score")
fig.tight_layout()
fig.savefig(OUT / "study_time_vs_score.png", dpi=160)
plt.close(fig)

tutoring_groups = [df.loc[df["tutoring"] == value, "final_score"] for value in ["No", "Yes"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(tutoring_groups, labels=["No", "Yes"])
ax.set_title("Final Scores by Tutoring Status")
ax.set_xlabel("Receives tutoring")
ax.set_ylabel("Final score")
fig.tight_layout()
fig.savefig(OUT / "scores_by_tutoring.png", dpi=160)
plt.close(fig)

corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
ax.set_yticks(range(len(corr.index)), corr.index)
for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
ax.set_title("Correlation Between Numeric Student Variables")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "student_correlation_heatmap.png", dpi=160)
plt.close(fig)

band_counts = df["performance_band"].value_counts().reindex(["Needs support", "Satisfactory", "Good", "Excellent"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(band_counts.index.astype(str), band_counts.values)
ax.set_title("Number of Students in Each Performance Band")
ax.set_xlabel("Performance band")
ax.set_ylabel("Number of students")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(OUT / "performance_band_counts.png", dpi=160)
plt.close(fig)

print("\nCharts saved to:", OUT)
print("NOTE: Synthetic demonstration data; associations are not causal evidence.")

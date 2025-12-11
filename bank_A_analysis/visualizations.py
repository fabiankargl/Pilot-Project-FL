import pandas as pd
import matplotlib.pyplot as plt
import os

plots_dir = "plots"

# Load data
df = pd.read_csv('../data/BankA.csv')

# 1. Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["age"], bins=8)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(plots_dir, "age_distribution.png"))
plt.close()

# 2. Hours per Week Distribution
plt.figure(figsize=(8,5))
plt.hist(df["hours-per-week"], bins=8)
plt.title("Hours per Week Distribution")
plt.xlabel("Hours per Week")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(plots_dir, "hours_per_week_distribution.png"))
plt.close()

# 3. Education Level Count
plt.figure(figsize=(8,5))
df["education"].value_counts().plot(kind="bar")
plt.title("Count of Education Levels")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig(os.path.join(plots_dir, "education_count.png"))
plt.close()

# 4. Gender Count
plt.figure(figsize=(8,5))
df['gender'].value_counts().plot(kind='bar')
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(plots_dir, "gender_count.png"))
plt.close()

# 5. Average Age by Education Level
plt.figure(figsize=(8,5))
df.groupby("education")["age"].mean().plot(kind="bar")
plt.title("Average Age by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Average Age")
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig(os.path.join(plots_dir, "avg_age_by_education.png"))
plt.close()

# 6. Average Hours Worked per Week by Education Level
plt.figure(figsize=(8,5))
df.groupby("education")["hours-per-week"].mean().plot(kind="bar")
plt.title("Average Hours Worked per Week by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Hours per Week")
plt.grid(True, linestyle="--", alpha=0.4)
plt.savefig(os.path.join(plots_dir, "avg_hours_by_education.png"))
plt.close()

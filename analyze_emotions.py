import pandas as pd
import matplotlib.pyplot as plt

# Load emotion log
df = pd.read_csv("emotions_log.csv", names=["time", "name", "emotion"])

# Group by person and emotion
grouped = df.groupby(["name", "emotion"]).size().reset_index(name="count")

# Plot per-person pie charts
for person in grouped["name"].unique():
    person_data = grouped[grouped["name"] == person]
    plt.figure(figsize=(5,5))
    plt.pie(person_data["count"], labels=person_data["emotion"], autopct="%1.1f%%")
    plt.title(f"Emotion distribution for {person}")
    plt.show()

# Plot overall distribution 
overall = df["emotion"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(overall, labels=overall.index, autopct="%1.1f%%", colors=plt.cm.Paired.colors)
plt.title("Overall Emotion Distribution")
plt.show()

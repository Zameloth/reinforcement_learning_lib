import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("output_mc_es.csv", header=None, names=["state", "action", "q_value"])
df["state"] = df["state"].astype(int)
df["action"] = df["action"].astype(int)
df["q_value"] = df["q_value"].astype(float)

pivot = df.pivot(index="state", columns="action", values="q_value")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, cmap="coolwarm_r", fmt=".2f")  
plt.title("Q-values Monte Carlo Exploring Starts")
plt.xlabel("Action")
plt.ylabel("State")
plt.tight_layout()
plt.show()

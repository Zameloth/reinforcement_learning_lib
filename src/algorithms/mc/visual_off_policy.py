import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lecture des Q-valeurs
df = pd.read_csv("q_values_off_policy.csv", names=["state", "action", "q_value"])

# Pivot pour la heatmap
pivot = df.pivot(index="state", columns="action", values="q_value")

# Heatmap inversée (si tu veux le rouge pour le négatif)
sns.heatmap(pivot, annot=True, cmap="coolwarm_r", fmt=".2f")

plt.title("Q-values Heatmap - Off Policy MC Control")
plt.xlabel("Action")
plt.ylabel("State")
plt.tight_layout()
plt.show()

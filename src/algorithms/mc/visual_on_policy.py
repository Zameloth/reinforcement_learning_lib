import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("output_on_policy.csv", names=["state", "action", "q_value"])

# Création d'une matrice Q-table pour affichage clair
q_table = df.pivot(index="state", columns="action", values="q_value")

# Affichage avec Matplotlib
plt.figure(figsize=(10, 6))
plt.title("Q-values learned by On-Policy First Visit MC")
plt.xlabel("Actions")
plt.ylabel("States")
plt.imshow(q_table, cmap="viridis", aspect="auto")
plt.colorbar(label="Q-value")
plt.xticks(range(q_table.columns.shape[0]))
plt.yticks(range(q_table.index.shape[0]))
plt.grid(False)
plt.show()

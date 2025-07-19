import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("output_off_policy.csv", names=["state", "action", "q_value"])

# Créer une table Q
q_table = df.pivot(index="state", columns="action", values="q_value").fillna(0)

# Déduire la meilleure action par état
best_actions = q_table.idxmax(axis=1)

# Plot flèches
plt.figure(figsize=(5, 5))
for state, action in best_actions.items():
    if action == 0:
        plt.text(0.5, state, '←', fontsize=20, ha='center')
    elif action == 1:
        plt.text(0.5, state, '→', fontsize=20, ha='center')

plt.ylim(-0.5, 4.5)
plt.xlim(0, 1)
plt.xticks([])
plt.yticks(range(len(q_table.index)))
plt.title("Politique Apprise Off-Policy (meilleure action par état)")
plt.xlabel("← gauche / → droite")
plt.ylabel("État")
plt.grid(True)
plt.show()

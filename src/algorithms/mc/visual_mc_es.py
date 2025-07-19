"""""
# visual_mc_es.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("output_mc_es.csv", names=["state", "action", "q_value"])
pivot = df.pivot(index="state", columns="action", values="q_value")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("Monte Carlo ES - Q-values")
plt.xlabel("Action")
plt.ylabel("État")
plt.tight_layout()
plt.show()

"""""
import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV (généré par le test Rust)
df = pd.read_csv("output_mc_es.csv", names=["state", "action", "q_value"])

# Déterminer la meilleure action pour chaque état
best_actions = df.groupby("state").apply(
    lambda group: group.loc[group["q_value"].idxmax()]
).reset_index(drop=True)

# Mapper les actions vers des symboles (adapte si ton env a + de 2 actions)
action_symbols = {
    0: "←",  # Par exemple : gauche
    1: "→",  # Par exemple : droite
    2: "↑",  # Si ton env a une 3e action
}

best_actions["symbol"] = best_actions["action"].map(action_symbols)

# Plotting
plt.figure(figsize=(8, 6))
plt.title("Politique Apprise (Meilleure action par état)")
plt.yticks(best_actions["state"])
plt.xticks([])

for _, row in best_actions.iterrows():
    plt.text(0.5, row["state"], row["symbol"], fontsize=14, ha='center', va='center')

plt.xlabel("← gauche, → droite (action optimale)")
plt.ylabel("État")
plt.grid(False)
plt.show()

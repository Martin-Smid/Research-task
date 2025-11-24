import pandas as pd
import matplotlib.pyplot as plt

filename = 'resources/data/max_values.csv'

# Načti s MultiIndex ve sloupcích (dva řádky záhlaví), ignoruj komentáře
data = pd.read_csv(filename, comment='#', header=[0, 1], index_col=0)

# Vytvoř seznam legendárních popisků (N a spin)
labels = [f"N = {n.replace('N', '')}, spin = {s.replace('s=', '')}"
          for (n, s) in data.columns]

# Normalizuj každou složku
for col in data.columns:
    first_val = data[col].iloc[0]
    data[col] = (data[col] / first_val) ** 0.25

# Plotting
plt.figure(figsize=(10, 6))

for col, label in zip(data.columns, labels):
    plt.plot(data.index, data[col], label=label)

# Osa a legenda
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Normalized Max Values", fontsize=18)
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.tight_layout()
plt.show()
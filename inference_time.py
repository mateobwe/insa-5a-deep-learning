import matplotlib.pyplot as plt 
import os
import pandas 

MODELS_DIR="C:\\Users\\mateo\\Documents\\INSA\\5A\\DeepLearning\\insa-5a-deep-learning\\model"

models = [
    {
        "name": d,
        "path": f"{MODELS_DIR}/{d}/results.csv",
        "data": pandas.read_csv(f"{MODELS_DIR}/{d}/results.csv")
    }
    for d in os.listdir(MODELS_DIR)
    if d.endswith("ep20")
]

if len(models) == 0:
    raise RuntimeError("No models with 'ep20' ending found.")

fig, ax = plt.subplots(figsize=(10, 6))

for model in models:
    dataframe = model["data"]
    if "time" in dataframe.columns:
        ax.plot(dataframe["epoch"], dataframe["time"], label=model["name"])

ax.set_title("Inference Time per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Time/Epoch (s)")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

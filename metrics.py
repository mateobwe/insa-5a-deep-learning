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
    if d != "100e_yolo11n_img360"
]

if len(models) == 0:
    raise RuntimeError("No models with results.csv found.")

metric_map = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)"
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (title, column) in zip(axes, metric_map.items()):
    for model in models:
        dataframe = model["data"]
        if column in dataframe.columns:
            ax.plot(dataframe["epoch"], dataframe[column], label=model["name"])
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(column)
    ax.grid(True)
    ax.legend()

fig.suptitle("Training Metrics for All Models", fontsize=16)
plt.tight_layout()
plt.show()

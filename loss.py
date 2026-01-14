import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot training losses')
parser.add_argument('csv_path', help='Path to the results.csv file')
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.csv_path)

# Create figure with subplots
fig, axes = plt.subplots(figsize=(14, 10))

# Plot training losses
axes.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
axes.plot(df['epoch'], df['train/cls_loss'], label='CLS Loss', marker='s')
axes.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
axes.set_xlabel('Epoch')
axes.set_ylabel('Loss')
axes.set_title('Training Losses')
axes.legend()
axes.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

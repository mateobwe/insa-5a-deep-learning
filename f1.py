import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot F1, Precision and Recall')
parser.add_argument('csv_path', help='Path to the results.csv file')
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.csv_path)
# Calculate F1 score
df['f1'] = 2 * ((df['metrics/precision(B)'] * df['metrics/recall(B)']) / (df['metrics/precision(B)'] + df['metrics/recall(B)']))

# Create figure with subplots
fig, axes = plt.subplots(figsize=(14, 10))

# Plot F1, Precision and Recall
axes.plot(df['epoch'], df['f1'], label='F1 Score')
axes.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
axes.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
axes.set_xlabel('Epoch')
axes.set_ylabel('Score')
axes.set_title('F1, Precision and Recall')
axes.legend()
axes.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_f1_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

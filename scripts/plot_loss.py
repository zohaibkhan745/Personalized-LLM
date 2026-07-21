import json
import matplotlib.pyplot as plt
import os

def load_eval_data(json_path):
    if not os.path.exists(json_path):
        return [], []
    with open(json_path, 'r') as f:
        data = json.load(f)
    epochs = []
    losses = []
    for entry in data.get('log_history', []):
        if 'eval_loss' in entry and 'epoch' in entry:
            epochs.append(entry['epoch'])
            losses.append(entry['eval_loss'])
    return epochs, losses

def plot_combined_validation_loss(models_info, output_path):
    plt.figure(figsize=(12, 7))
    colors = ['gray', 'orange', 'red', 'green']
    
    for i, (name, path) in enumerate(models_info):
        epochs, losses = load_eval_data(path)
        if epochs:
            plt.plot(epochs, losses, label=name, color=colors[i % len(colors)], linewidth=2.5, marker='o')

    plt.title("Master Project Progress: Validation Loss Across All Versions", fontsize=16)
    plt.xlabel("Epochs Trained", fontsize=14)
    plt.ylabel("Validation Loss (Lower is Better)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined plot: {output_path}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("eval/results", exist_ok=True)
    
    models = [
        ("V1 (Baseline | 2 Epochs)", "llama3-math-tutor-adapter/checkpoint-51/trainer_state.json"),
        ("V2 (Bugfix | 2 Epochs)", "llama3-math-tutor-adapter-v2/checkpoint-34/trainer_state.json"),
        ("V3 (Overfit | 15 Epochs)", "llama3-math-tutor-adapter-v3/checkpoint-255/trainer_state.json"),
        ("V4 (Optimized + Early Stop)", "llama3-math-tutor-adapter-v4/checkpoint-80/trainer_state.json")
    ]
    
    plot_combined_validation_loss(models, "eval/results/combined_loss_curve.png")

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

def plot_trainer_state(json_path, output_image_path, title):
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    epochs_train = []
    loss_train = []
    
    epochs_eval = []
    loss_eval = []

    for entry in data.get('log_history', []):
        if 'loss' in entry and 'epoch' in entry:
            epochs_train.append(entry['epoch'])
            loss_train.append(entry['loss'])
        
        elif 'eval_loss' in entry and 'epoch' in entry:
            epochs_eval.append(entry['epoch'])
            loss_eval.append(entry['eval_loss'])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, loss_train, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(epochs_eval, loss_eval, label='Validation Loss', color='red', linewidth=2, marker='o')

    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Saved plot: {output_image_path}")
    plt.close()

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
    
    plot_trainer_state(models[0][1], "eval/results/v1_loss_curve.png", "V1 Loss Curve (Baseline)")
    plot_trainer_state(models[1][1], "eval/results/v2_loss_curve.png", "V2 Loss Curve (Bugfix)")
    plot_trainer_state(models[2][1], "eval/results/v3_loss_curve.png", "V3 Loss Curve (The Overfit Model)")
    plot_trainer_state(models[3][1], "eval/results/v4_loss_curve.png", "V4 Loss Curve (Early Stopping)")

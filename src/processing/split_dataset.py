import json
import random
import re
from collections import defaultdict
from pathlib import Path

def extract_traits(content: str) -> dict:
    """Extracts the learner traits from the user prompt."""
    traits = {}
    for key in ["Learning Style", "Confidence", "Level", "Pedagogical Strategy"]:
        match = re.search(fr"- {key}:\s*(.+)", content)
        if match:
            traits[key] = match.group(1).strip()
        else:
            traits[key] = "Unknown"
    return traits

def print_distribution(dataset_name: str, dataset: list):
    """Calculates and prints distribution percentages for the given dataset."""
    counts = {
        "Learning Style": defaultdict(int),
        "Confidence": defaultdict(int),
        "Level": defaultdict(int),
        "Pedagogical Strategy": defaultdict(int)
    }
    
    for item in dataset:
        traits = extract_traits(item['messages'][1]['content'])
        for k, v in traits.items():
            counts[k][v] += 1
            
    print(f"--- {dataset_name} Distribution ({len(dataset)} examples) ---")
    for trait, dist in counts.items():
        print(f"  {trait}:")
        for k, c in dist.items():
            pct = round((c / len(dataset)) * 100, 1)
            print(f"    - {k}: {c} ({pct}%)")
    print("\n")

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    input_file = PROJECT_ROOT / "data" / "generated" / "train.jsonl"
    train_out = PROJECT_ROOT / "data" / "clean" / "train.jsonl"
    val_out = PROJECT_ROOT / "data" / "clean" / "validation.jsonl"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    # 1. Random shuffle with fixed seed
    random.seed(42)
    random.shuffle(data)
    
    # 2. Extract compound key and sort to group similar entries 
    # (stable sort preserves random shuffle inside identical combined keys)
    def sort_key(ex):
        t = extract_traits(ex['messages'][1]['content'])
        return (t['Learning Style'], t['Level'], t['Confidence'], t['Pedagogical Strategy'])
        
    stratified_data = sorted(data, key=sort_key)
    
    # 3. Stratified split (every 10th item goes to validation, representing exactly 10%)
    validation = stratified_data[::10]
    train = [item for i, item in enumerate(stratified_data) if i % 10 != 0]
    
    # Re-shuffle the final splits to avoid ordered structures leaking into training epochs
    random.shuffle(train)
    random.shuffle(validation)

    # 4. Save the files
    with open(train_out, 'w', encoding='utf-8') as f:
        for item in train:
            f.write(json.dumps(item) + '\n')
            
    with open(val_out, 'w', encoding='utf-8') as f:
        for item in validation:
            f.write(json.dumps(item) + '\n')

    # 5. Overlap verification
    # Using the assistant response or the whole string hash since it was deduplicated
    train_hashes = set(json.dumps(ex) for ex in train)
    val_hashes = set(json.dumps(ex) for ex in validation)
    overlap = train_hashes.intersection(val_hashes)
    
    print("========================================")
    print("DATASET SPLIT COMPLETE")
    print("========================================")
    print(f"Total Original: {len(data)}")
    print(f"Train Split:    {len(train)}")
    print(f"Val Split:      {len(validation)}\n")
    
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping examples between train and val!")
    else:
        print("Verification: No overlapping examples found between Train and Val sets. ✅\n")

    # 6. Print Statistics
    print_distribution("Total Dataset", data)
    print_distribution("Training Set", train)
    print_distribution("Validation Set", validation)

if __name__ == "__main__":
    main()

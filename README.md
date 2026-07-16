# Personalized Math Tutor - LLM Fine-Tuning Pipeline

This repository contains the data generation, fine-tuning, and evaluation pipelines for building a personalized math tutoring language model powered by LLaMA 3. 

The primary goal of this repository is to train an LLM capable of adapting its tutoring methodology (e.g., visual analogies, Socratic questioning, concise logic) to the individual learning style, confidence, and detail preference of the student.

## 📂 Repository Structure

The project has been elegantly separated into core modular domains:

- **`train.py`**: The definitive QLoRA entry point utilizing `trl.SFTTrainer`, handling the 4-bit quantization config and LoRA target projections.
- **`report.md`**: The structured supervisor-level executive summary detailing model parameters, hyperparams, and outcomes.
- **`data/`**: The core datasets.
  - `clean/`: Contains `train.jsonl` and `validation.jsonl` using the standard system/user/assistant conversational arrays.
- **`src/`**: Data engineering and processing toolkit.
  - `generation/`: Object-Oriented architectures used to synthetically spawn tutoring datasets mimicking varying learner profiles.
  - `processing/`: Splitting pipelines to divide datasets rigorously.
- **`eval/`**: The formal evaluation harness to quantitatively verify math capabilities.
  - `eval_scripts/`:
    - `evaluate_metrics.py`: Computes deterministic metrics over held-out domains using pure mathematical algebraic logic parsers.
    - `audit_math_correctness.py`: Scrubs the source training dataset with `SymPy` tracking to prevent hallucinations within the training bounds.
  - `results/`: Contains the generated PDF (`evaluation_report.pdf`), CSV summaries, and raw JSON extraction structures mapping the base vs. finetune performance.

## 🚀 Key Highlights & Metrics

A quantitative automated test confirmed our adapter correctly generalized the requested system instructions directly tracking student profiles (vastly out-expanding the base model's logic chains). Algebraic parsing with `SymPy` correctly scored exact math reasoning logic against held-out tests across standard equations, systems, ratios, and probability.
 
> **Refer to `report.md` and `eval/results/evaluation_report.pdf` for the complete final metrics rundown.**

---
*Built incrementally targeting Meta-Llama-3-8B-Instruct using PyTorch, Transformers, PEFT, and BitsAndBytes on a local RTX-4090 accelerator.*

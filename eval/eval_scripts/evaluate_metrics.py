"""
evaluate_metrics.py — Three-way comparison: base, v1 adapter, v2 adapter.

Loads the base model ONCE and hot-swaps LoRA adapters via PeftModel to save
VRAM. Runs the same 20 deterministic held-out problems for every variant.

Outputs (written to ./eval/results/ relative to project root):
  - evaluation_results_raw.json
  - evaluation_metrics_summary.csv
  - evaluation_report.pdf

Usage (run from project root):
    python3 eval/eval_scripts/evaluate_metrics.py
"""

import contextlib
import csv
import json
import os
import re
import time

import torch
from datetime import datetime
from peft import PeftModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sympy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "eval", "results")

RAW_JSON_PATH  = os.path.join(RESULTS_DIR, "evaluation_results_raw.json")
CSV_PATH       = os.path.join(RESULTS_DIR, "evaluation_metrics_summary.csv")
PDF_PATH       = os.path.join(RESULTS_DIR, "evaluation_report.pdf")

BASE_MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_V1_PATH = os.path.join(PROJECT_ROOT, "llama3-math-tutor-adapter")
ADAPTER_V2_PATH = os.path.join(PROJECT_ROOT, "llama3-math-tutor-adapter-v2")
ADAPTER_V3_PATH = os.path.join(PROJECT_ROOT, "llama3-math-tutor-adapter-v3")


# ---------------------------------------------------------------------------
# 20 deterministic held-out problems
# ---------------------------------------------------------------------------
def get_test_set():
    problems = [
        {"id":  1, "topic": "linear_equations",    "text": "Solve for x: 5x - 4 = 21",                                                                                       "gt": "5",               "profile": {"learning_style": "Visual",      "confidence": "Low",          "detail_preference": "detailed"}},
        {"id":  2, "topic": "linear_equations",    "text": "Solve for x: 3(x - 2) = 12",                                                                                     "gt": "6",               "profile": {"learning_style": "Analytical",  "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id":  3, "topic": "linear_equations",    "text": "Solve for y: 2y + 5 = 2y + 10",                                                                                  "gt": "no solution",     "profile": {"learning_style": "Conceptual",  "confidence": "High",         "detail_preference": "detailed"}},
        {"id":  4, "topic": "quadratic_factoring", "text": "Factor: x^2 - 9",                                                                                                "gt": "(x - 3)*(x + 3)", "profile": {"learning_style": "Reflective",  "confidence": "Low",          "detail_preference": "concise"}},
        {"id":  5, "topic": "quadratic_factoring", "text": "Factor: x^2 + 7x + 10",                                                                                          "gt": "(x + 2)*(x + 5)", "profile": {"learning_style": "Visual",      "confidence": "High",         "detail_preference": "detailed"}},
        {"id":  6, "topic": "quadratic_factoring", "text": "Factor: 2x^2 + 4x",                                                                                              "gt": "2*x*(x + 2)",     "profile": {"learning_style": "Analytical",  "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id":  7, "topic": "ratios",              "text": "A recipe uses 2 cups flour to 3 cups sugar. If you use 8 cups flour, how many cups of sugar do you need?",        "gt": "12",              "profile": {"learning_style": "Conceptual",  "confidence": "Low",          "detail_preference": "detailed"}},
        {"id":  8, "topic": "ratios",              "text": "A car travels 150 miles in 3 hours. At this same rate, how far will it travel in 5 hours?",                       "gt": "250",             "profile": {"learning_style": "Reflective",  "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id":  9, "topic": "ratios",              "text": "If 4 painters can paint a house in 10 days, how long will it take 5 painters to do the same job?",               "gt": "8",               "profile": {"learning_style": "Visual",      "confidence": "High",         "detail_preference": "detailed"}},
        {"id": 10, "topic": "systems",             "text": "Solve for x: y = 2x and x + y = 15",                                                                             "gt": "5",               "profile": {"learning_style": "Analytical",  "confidence": "Low",          "detail_preference": "concise"}},
        {"id": 11, "topic": "systems",             "text": "Solve for y: x - y = 2 and 2x + y = 13",                                                                         "gt": "3",               "profile": {"learning_style": "Conceptual",  "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 12, "topic": "systems",             "text": "Solve for x: 3x + 2y = 12 and y = x + 1",                                                                        "gt": "2",               "profile": {"learning_style": "Reflective",  "confidence": "High",         "detail_preference": "concise"}},
        {"id": 13, "topic": "probability",         "text": "A standard 6-sided die is rolled. What is the probability of rolling a prime number? Give a simplified fraction.", "gt": "1/2",             "profile": {"learning_style": "Visual",      "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 14, "topic": "probability",         "text": "A bag has 3 red marbles and 2 blue marbles. What is the probability of drawing a blue marble? Give a simplified fraction.", "gt": "2/5",   "profile": {"learning_style": "Analytical",  "confidence": "High",         "detail_preference": "concise"}},
        {"id": 15, "topic": "probability",         "text": "A fair coin is flipped 3 times. What is the probability of getting all heads? Give a simplified fraction.",       "gt": "1/8",             "profile": {"learning_style": "Conceptual",  "confidence": "Low",          "detail_preference": "detailed"}},
        {"id": 16, "topic": "statistics",          "text": "Find the median of the following dataset: 3, 1, 4, 1, 5",                                                         "gt": "3",               "profile": {"learning_style": "Reflective",  "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id": 17, "topic": "statistics",          "text": "Find the arithmetic mean of the following dataset: 10, 20, 30",                                                   "gt": "20",              "profile": {"learning_style": "Visual",      "confidence": "High",         "detail_preference": "detailed"}},
        {"id": 18, "topic": "statistics",          "text": "Find the range of the following dataset: 5, 2, 9, 4, 7",                                                          "gt": "7",               "profile": {"learning_style": "Analytical",  "confidence": "Low",          "detail_preference": "concise"}},
        {"id": 19, "topic": "exponential",         "text": "Solve for x: 2^x = 32",                                                                                           "gt": "5",               "profile": {"learning_style": "Conceptual",  "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 20, "topic": "exponential",         "text": "Solve for x: 3^(x-1) = 9",                                                                                       "gt": "3",               "profile": {"learning_style": "Reflective",  "confidence": "High",         "detail_preference": "concise"}},
    ]
    return problems


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------
def check_correctness(extracted, gt):
    if not extracted:
        return "UNPARSEABLE"

    extracted_clean = extracted.lower().replace(",", "").strip()
    gt_clean = gt.lower().strip()

    if gt_clean == "no solution":
        return "CORRECT" if "no solution" in extracted_clean else "INCORRECT"

    if "=" in extracted_clean:
        extracted_clean = extracted_clean.split("=")[-1].strip()

    transformations = standard_transformations + (
        implicit_multiplication_application, convert_xor
    )
    try:
        ex_expr = parse_expr(extracted_clean, transformations=transformations)
        gt_expr = parse_expr(gt_clean,        transformations=transformations)
        if sympy.simplify(ex_expr - gt_expr) == 0:
            return "CORRECT"
    except Exception:
        pass

    if re.fullmatch(r"\b" + re.escape(gt_clean) + r"\b", extracted_clean):
        return "CORRECT"
    if "".join(extracted_clean.split()) == "".join(gt_clean.split()):
        return "CORRECT"

    return "INCORRECT"


def extract_answer(text, gt=None):
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    end_text_lines = " ".join(lines[-3:])

    patterns = [
        r"(?i)the answer is\s*([^.]+)",
        r"(?i)therefore,?\s*(?:we have)?\s*([^.]+)",
        r"(?i)solution is\s*([^.]+)",
        r"\b(?:x|y)\s*=\s*([^\s.]+)",
        r"([-+]?\d*\.?\d+(?:/\d+)?)\s*$",
    ]
    for pat in patterns:
        match = re.search(pat, end_text_lines)
        if match:
            return match.group(1).replace('"', "").strip()

    if gt and gt.lower() in end_text_lines.lower():
        return gt

    return lines[-1]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_response(model, tokenizer, prompt_messages):
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2,
            do_sample=False,          # greedy / deterministic
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def generate_pdf_report(summary_metrics, pdf_path, training_metrics=None):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    heading2 = styles["Heading2"]
    normal   = styles["Normal"]
    elements = []

    # ── Title ──────────────────────────────────────────────────────────────
    elements.append(Paragraph(
        "<b>Model Evaluation Report — Personalized Math Tutor (3-Way Comparison)</b>",
        styles["Title"]
    ))
    elements.append(Paragraph(
        f"{datetime.now().strftime('%Y-%m-%d')}  |  20 held-out problems  |  "
        f"Greedy decoding (temperature=0.2, do_sample=False)  |  SymPy-verified grading",
        normal
    ))
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    elements.append(Spacer(1, 10))

    # ── Training Telemetry Table (If Available) ─────────────────────────────
    if training_metrics:
        elements.append(Paragraph("<b>Training Telemetry (Loss & Meta)</b>", heading2))
        train_data = [["Model", "Eval Loss", "Epochs", "Mean Token Acc %"]]
        for v in [m["model_name"] for m in summary_metrics]:
            if v == "base":
                train_data.append(["base", "N/A (Base)", "N/A", "N/A"])
            elif v in training_metrics:
                tm = training_metrics[v]
                loss = round(tm.get("eval_loss", 0), 4)
                epochs = tm.get("epoch", "N/A")
                acc = round(tm.get("eval_mean_token_accuracy", 0) * 100, 2)
                train_data.append([str(v), str(loss), str(epochs), f"{acc}%"])
            else:
                train_data.append([str(v), "Missing Details", "Missing", "Missing"])

        t_train = Table(train_data, colWidths=[90, 100, 90, 120])
        t_train.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2a9d8f")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.whitesmoke),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  11),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  10),
            ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f3f3f3")),
            ("GRID",          (0, 0), (-1, -1), 0.8, colors.black),
        ]))
        elements.append(t_train)
        elements.append(Spacer(1, 20))

    # ── Overall Summary Table ───────────────────────────────────────────────
    elements.append(Paragraph("<b>Overall Summary</b>", heading2))
    summary_data = [["Model", "Overall Accuracy (%)", "Unparseable Count", "Avg Response Length"]]
    for row in summary_metrics:
        summary_data.append([
            str(row["model_name"]),
            f"{row['overall_accuracy_%']}%",
            str(row["unparseable_count"]),
            f"{row['avg_response_length']} chars",
        ])
    t_sum = Table(summary_data, colWidths=[90, 120, 120, 130])
    t_sum.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.whitesmoke),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  11),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  10),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f3f3f3")),
        ("GRID",          (0, 0), (-1, -1), 0.8, colors.black),
    ]))
    elements.append(t_sum)
    elements.append(Spacer(1, 20))

    # ── Per-Topic Accuracy Table ────────────────────────────────────────────
    elements.append(Paragraph("<b>Per-Topic Accuracy Breakdown (%)</b>", heading2))
    topics = [
        k.replace("acc_", "").replace("_%", "")
        for k in summary_metrics[0].keys()
        if k.startswith("acc_") and not k.startswith("conf_")
    ]

    topic_data = [["Model"] + [t.replace("_", " ").title() for t in topics]]
    for row in summary_metrics:
        t_vals = [f"{row.get('acc_' + t + '_%', 0)}%" for t in topics]
        topic_data.append([row["model_name"]] + t_vals)

    col_w = max(46, int(430 / max(len(topics), 1)))
    t_topic = Table(topic_data, colWidths=[70] + [col_w] * len(topics))
    t_topic.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1f497d")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.whitesmoke),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#eaf2fb")),
        ("GRID",          (0, 0), (-1, -1), 0.8, colors.black),
    ]))
    elements.append(t_topic)
    elements.append(Spacer(1, 24))

    # ── Programmatic Summary Section ────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Comparative Analysis Summary</b>", heading2))

    sorted_by_acc   = sorted(summary_metrics, key=lambda x: x["overall_accuracy_%"], reverse=True)
    best            = sorted_by_acc[0]
    worst           = sorted_by_acc[-1]
    best_name       = best["model_name"]
    worst_name      = worst["model_name"]
    best_acc        = best["overall_accuracy_%"]
    worst_acc       = worst["overall_accuracy_%"]

    base_row        = next((m for m in summary_metrics if m["model_name"] == "base"), None)
    v1_row          = next((m for m in summary_metrics if m["model_name"] == "v1"),   None)
    v2_row          = next((m for m in summary_metrics if m["model_name"] == "v2"),   None)

    insights = []

    # Best / Worst overall
    insights.append(
        f"<b>Highest overall accuracy:</b> <b>{best_name}</b> at {best_acc}%."
    )
    insights.append(
        f"<b>Lowest overall accuracy:</b> <b>{worst_name}</b> at {worst_acc}%."
    )

    # v1 vs v2 delta
    if v1_row and v2_row:
        delta_v = round(v2_row["overall_accuracy_%"] - v1_row["overall_accuracy_%"], 2)
        direction = "improvement" if delta_v > 0 else ("no change" if delta_v == 0 else "regression")
        sign = "+" if delta_v >= 0 else ""
        insights.append(
            f"<b>v2 vs v1 accuracy delta:</b> {sign}{delta_v}% "
            f"({'v2 improved over v1' if delta_v > 0 else 'v2 regressed vs v1' if delta_v < 0 else 'identical'})."
        )

    # Base deltas
    if base_row:
        for candidate in [v1_row, v2_row]:
            if candidate is None:
                continue
            delta = round(candidate["overall_accuracy_%"] - base_row["overall_accuracy_%"], 2)
            sign = "+" if delta >= 0 else ""
            direction = "improvement" if delta > 0 else ("no change" if delta == 0 else "drop")
            insights.append(
                f"<b>{candidate['model_name']} vs base:</b> {sign}{delta}% ({direction} over the base model)."
            )

    elements.append(Spacer(1, 6))
    for line in insights:
        elements.append(Paragraph(f"• {line}", normal))
        elements.append(Spacer(1, 4))

    # Per-topic best / worst ------------------------------------------------
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Per-Topic Best / Worst Model</b>", heading2))

    topic_comparison_data = [["Topic", "Best Model", "Best Acc (%)", "Worst Model", "Worst Acc (%)"]]
    for t in topics:
        key = f"acc_{t}_%"
        scored = [(m["model_name"], m.get(key, 0.0)) for m in summary_metrics]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        best_t  = scored_sorted[0]
        worst_t = scored_sorted[-1]
        topic_comparison_data.append([
            t.replace("_", " ").title(),
            best_t[0],  f"{best_t[1]}%",
            worst_t[0], f"{worst_t[1]}%",
        ])

    t_per_topic = Table(topic_comparison_data, colWidths=[110, 75, 75, 95, 75])
    t_per_topic.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2e4057")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.whitesmoke),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#f0f4f8")),
        ("GRID",          (0, 0), (-1, -1), 0.8, colors.black),
    ]))
    elements.append(t_per_topic)
    elements.append(Spacer(1, 20))

    # Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated by evaluate_metrics.py  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ParagraphStyle("footer", parent=normal, fontSize=8, textColor=colors.grey)
    ))

    doc.build(elements)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.chdir(PROJECT_ROOT)  # ensure relative adapter paths resolve correctly

    # ── Discover which variants to test ────────────────────────────────────
    variants = ["base"]
    adapter_paths = {
        "v1": ADAPTER_V1_PATH,
        "v2": ADAPTER_V2_PATH,
        "v3": ADAPTER_V3_PATH,
    }
    for v_name, v_path in adapter_paths.items():
        if os.path.isdir(v_path):
            variants.append(v_name)
        else:
            print(f"[NOTE] {v_name} adapter not found at {v_path} — skipping.")

    print(f"\nVariants to evaluate: {variants}")

    # ── Load training telemetry ──────────────────────────────────────────
    training_metrics = {}
    for v_name in variants:
        if v_name == "base": continue
        results_json = os.path.join(adapter_paths[v_name], "training_results.json")
        if os.path.isfile(results_json):
            try:
                with open(results_json, "r") as f:
                    training_metrics[v_name] = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not read training metrics for {v_name}: {e}")

    # ── Load base model once ────────────────────────────────────────────────
    print("\nLoading 4-bit quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {BASE_MODEL_ID} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Hot-swap LoRA adapters (load once per adapter into PeftModel) ───────
    model = base_model
    adapter_names = [v for v in variants if v != "base"]

    if adapter_names:
        first = adapter_names[0]
        first_path = adapter_paths[first]
        print(f"Initialising PeftModel with adapter '{first}' from {first_path} ...")
        model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first)

        for other in adapter_names[1:]:
            other_path = adapter_paths[other]
            print(f"Loading adapter '{other}' from {other_path} ...")
            model.load_adapter(other_path, adapter_name=other)

    model.eval()
    test_suite = get_test_set()

    system_tutor = "You are a personalized math tutor that adapts explanations to each learner."
    system_base  = "You are a helpful math tutor. Solve the problem step by step."

    all_results    = []
    summary_metrics = []

    print(f"\nSTARTING EVALUATION on {len(test_suite)} held-out problems across variants: {variants}\n")

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"  Testing variant: {variant.upper()}")
        print(f"{'='*60}\n")

        # Choose inference context (disable adapters for base, set adapter for v1/v2)
        if variant == "base":
            if isinstance(model, PeftModel):
                ctx = model.disable_adapter()
            else:
                ctx = contextlib.nullcontext()
        else:
            if isinstance(model, PeftModel):
                model.set_adapter(variant)
            ctx = contextlib.nullcontext()

        metrics = {
            "variant": variant,
            "correct": 0, "incorrect": 0, "unparseable": 0,
            "topic_correct": {}, "topic_total": {},
            "conf_correct":  {}, "conf_total":  {},
            "total_chars": 0,
        }

        with ctx:
            for idx, prob in enumerate(test_suite, start=1):
                if variant == "base":
                    messages = [
                        {"role": "system", "content": system_base},
                        {"role": "user",   "content": f"Problem: {prob['text']}"},
                    ]
                else:
                    user_content = (
                        f"Student Profile:\n"
                        f"- Learning Style: {prob['profile']['learning_style']}\n"
                        f"- Confidence: {prob['profile']['confidence']}\n"
                        f"- Detail Preference: {prob['profile']['detail_preference']}\n\n"
                        f"Problem:\n{prob['text']}"
                    )
                    messages = [
                        {"role": "system", "content": system_tutor},
                        {"role": "user",   "content": user_content},
                    ]

                output_text  = generate_response(model, tokenizer, messages)
                extracted    = extract_answer(output_text, prob["gt"])
                status       = check_correctness(extracted, prob["gt"])

                is_unparseable = (not extracted) or (status == "UNPARSEABLE")
                is_suspicious  = False
                if not is_unparseable and status != "CORRECT":
                    is_gt_numeric      = bool(re.fullmatch(r"[-+]?\d+(?:/\d+)?", str(prob["gt"]).replace(" ", "")))
                    has_letters_or_vars = bool(re.search(r"[a-zA-Z=]", extracted))
                    if is_gt_numeric and has_letters_or_vars:
                        is_suspicious = True

                if is_unparseable:
                    print(f"  [UNPARSEABLE] {variant} | Prob {prob['id']} | GT={prob['gt']} | Ext='{extracted}'")
                    status = "UNPARSEABLE"
                elif is_suspicious:
                    print(f"  [SUSPICIOUS ] {variant} | Prob {prob['id']} | GT={prob['gt']} | Ext='{extracted}'")

                all_results.append({
                    "model_variant":    variant,
                    "problem_id":       prob["id"],
                    "topic":            prob["topic"],
                    "ground_truth":     prob["gt"],
                    "extracted_answer": extracted,
                    "status":           status,
                    "full_response":    output_text,
                })

                if status == "CORRECT":
                    metrics["correct"]    += 1
                elif status == "INCORRECT":
                    metrics["incorrect"]  += 1
                else:
                    metrics["unparseable"] += 1

                topic = prob["topic"]
                conf  = prob["profile"]["confidence"]
                metrics["topic_total"][topic]  = metrics["topic_total"].get(topic, 0) + 1
                metrics["conf_total"][conf]    = metrics["conf_total"].get(conf, 0) + 1
                if status == "CORRECT":
                    metrics["topic_correct"][topic] = metrics["topic_correct"].get(topic, 0) + 1
                    metrics["conf_correct"][conf]   = metrics["conf_correct"].get(conf, 0) + 1

                metrics["total_chars"] += len(output_text)
                print(f"  [{variant}] {idx:>2}/{len(test_suite)} → {status:<12} | GT: {prob['gt']:<18} | Ext: {extracted}")

        # Compile flat summary row
        total_probs = len(test_suite)
        row = {
            "model_name":         variant,
            "overall_accuracy_%": round((metrics["correct"] / total_probs) * 100, 2),
            "unparseable_count":  metrics["unparseable"],
            "avg_response_length": round(metrics["total_chars"] / total_probs, 1),
        }
        for t in metrics["topic_total"]:
            t_acc = (metrics["topic_correct"].get(t, 0) / metrics["topic_total"][t]) * 100
            row[f"acc_{t}_%"] = round(t_acc, 1)
        for c in metrics["conf_total"]:
            c_acc = (metrics["conf_correct"].get(c, 0) / metrics["conf_total"][c]) * 100
            row[f"conf_acc_{c}_%"] = round(c_acc, 1)

        summary_metrics.append(row)

    # ── Save outputs ────────────────────────────────────────────────────────
    print(f"\nSaving raw results → {RAW_JSON_PATH}")
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saving CSV summary → {CSV_PATH}")
    if summary_metrics:
        keys = summary_metrics[0].keys()
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_metrics)

    print(f"Generating PDF report → {PDF_PATH}")
    if summary_metrics:
        generate_pdf_report(summary_metrics, PDF_PATH, training_metrics)

    # ── Final console comparison table ─────────────────────────────────────
    topics = [
        k.replace("acc_", "").replace("_%", "")
        for k in summary_metrics[0].keys()
        if k.startswith("acc_") and not k.startswith("conf_")
    ]

    topic_headers = [t.replace("_", " ").title() for t in topics]

    col_widths = [10, 16, 12] + [14] * len(topics) + [12]
    sep_line = "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 4)

    def fmt_row(*vals):
        parts = []
        for v, w in zip(vals, col_widths):
            parts.append(str(v).ljust(w))
        return " | ".join(parts)

    print("\n")
    print("=" * len(sep_line))
    print("  FINAL EVALUATION COMPARISON TABLE")
    print("=" * len(sep_line))
    print(sep_line)
    header_vals = ["Model", "Overall Acc %", "Unparseable"] + topic_headers + ["Avg Length"]
    print(fmt_row(*header_vals))
    print(sep_line)
    for srow in summary_metrics:
        t_vals = [f"{srow.get('acc_' + t + '_%', 0)}%" for t in topics]
        print(fmt_row(
            srow["model_name"],
            f"{srow['overall_accuracy_%']}%",
            srow["unparseable_count"],
            *t_vals,
            f"{srow['avg_response_length']}c",
        ))
    print(sep_line)

    # ── Best / Worst summary lines ──────────────────────────────────────────
    sorted_all  = sorted(summary_metrics, key=lambda x: x["overall_accuracy_%"], reverse=True)
    best_model  = sorted_all[0]
    worst_model = sorted_all[-1]

    # v1 vs v2 delta
    v1_row = next((m for m in summary_metrics if m["model_name"] == "v1"), None)
    v2_row = next((m for m in summary_metrics if m["model_name"] == "v2"), None)
    if v1_row and v2_row:
        delta = round(v2_row["overall_accuracy_%"] - v1_row["overall_accuracy_%"], 2)
        sign  = "+" if delta >= 0 else ""
        print(f"\nv2 vs v1 accuracy delta: {sign}{delta}%  "
              f"({'v2 improved' if delta > 0 else 'v2 regressed' if delta < 0 else 'identical'})")

    print(f"\nBest model:  {best_model['model_name']}  at {best_model['overall_accuracy_%']}% accuracy")
    print(f"Worst model: {worst_model['model_name']}  at {worst_model['overall_accuracy_%']}% accuracy")
    print(f"\nAll output files saved to: {RESULTS_DIR}/")
    print("\nEvaluation completed successfully.\n")


if __name__ == "__main__":
    main()

import os
import json
import csv
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime

def get_test_set():
    # 20 held-out problems, well-distributed
    problems = [
        {"id": 1, "topic": "linear_equations", "text": "Solve for x: 5x - 4 = 21", "gt": "5", "profile": {"learning_style": "Visual", "confidence": "Low", "detail_preference": "detailed"}},
        {"id": 2, "topic": "linear_equations", "text": "Solve for x: 3(x - 2) = 12", "gt": "6", "profile": {"learning_style": "Analytical", "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id": 3, "topic": "linear_equations", "text": "Solve for y: 2y + 5 = 2y + 10", "gt": "no solution", "profile": {"learning_style": "Conceptual", "confidence": "High", "detail_preference": "detailed"}},
        
        {"id": 4, "topic": "quadratic_factoring", "text": "Factor: x^2 - 9", "gt": "(x - 3)*(x + 3)", "profile": {"learning_style": "Reflective", "confidence": "Low", "detail_preference": "concise"}},
        {"id": 5, "topic": "quadratic_factoring", "text": "Factor: x^2 + 7x + 10", "gt": "(x + 2)*(x + 5)", "profile": {"learning_style": "Visual", "confidence": "High", "detail_preference": "detailed"}},
        {"id": 6, "topic": "quadratic_factoring", "text": "Factor: 2x^2 + 4x", "gt": "2*x*(x + 2)", "profile": {"learning_style": "Analytical", "confidence": "Intermediate", "detail_preference": "concise"}},
        
        {"id": 7, "topic": "ratios", "text": "A recipe uses 2 cups flour to 3 cups sugar. If you use 8 cups flour, how many cups of sugar do you need?", "gt": "12", "profile": {"learning_style": "Conceptual", "confidence": "Low", "detail_preference": "detailed"}},
        {"id": 8, "topic": "ratios", "text": "A car travels 150 miles in 3 hours. At this same rate, how far will it travel in 5 hours?", "gt": "250", "profile": {"learning_style": "Reflective", "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id": 9, "topic": "ratios", "text": "If 4 painters can paint a house in 10 days, how long will it take 5 painters to do the same job? (Assume they work at the same rate)", "gt": "8", "profile": {"learning_style": "Visual", "confidence": "High", "detail_preference": "detailed"}},
        
        {"id": 10, "topic": "systems", "text": "Solve for x: y = 2x and x + y = 15", "gt": "5", "profile": {"learning_style": "Analytical", "confidence": "Low", "detail_preference": "concise"}},
        {"id": 11, "topic": "systems", "text": "Solve for y: x - y = 2 and 2x + y = 13", "gt": "3", "profile": {"learning_style": "Conceptual", "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 12, "topic": "systems", "text": "Solve for x: 3x + 2y = 12 and y = x + 1", "gt": "2", "profile": {"learning_style": "Reflective", "confidence": "High", "detail_preference": "concise"}},
        
        {"id": 13, "topic": "probability", "text": "A standard 6-sided die is rolled. What is the probability of rolling a prime number? Provide your answer as a simplified fraction.", "gt": "1/2", "profile": {"learning_style": "Visual", "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 14, "topic": "probability", "text": "A bag has 3 red marbles and 2 blue marbles. What is the probability of drawing a blue marble? Provide your answer as a simplified fraction.", "gt": "2/5", "profile": {"learning_style": "Analytical", "confidence": "High", "detail_preference": "concise"}},
        {"id": 15, "topic": "probability", "text": "A fair coin is flipped 3 times. What is the probability of getting all heads? Provide your answer as a simplified fraction.", "gt": "1/8", "profile": {"learning_style": "Conceptual", "confidence": "Low", "detail_preference": "detailed"}},
        
        {"id": 16, "topic": "statistics", "text": "Find the median of the following dataset: 3, 1, 4, 1, 5", "gt": "3", "profile": {"learning_style": "Reflective", "confidence": "Intermediate", "detail_preference": "concise"}},
        {"id": 17, "topic": "statistics", "text": "Find the arithmetic mean of the following dataset: 10, 20, 30", "gt": "20", "profile": {"learning_style": "Visual", "confidence": "High", "detail_preference": "detailed"}},
        {"id": 18, "topic": "statistics", "text": "Find the range of the following dataset: 5, 2, 9, 4, 7", "gt": "7", "profile": {"learning_style": "Analytical", "confidence": "Low", "detail_preference": "concise"}},
        
        {"id": 19, "topic": "exponential", "text": "Solve for x: 2^x = 32", "gt": "5", "profile": {"learning_style": "Conceptual", "confidence": "Intermediate", "detail_preference": "detailed"}},
        {"id": 20, "topic": "exponential", "text": "Solve for x: 3^(x-1) = 9", "gt": "3", "profile": {"learning_style": "Reflective", "confidence": "High", "detail_preference": "concise"}}
    ]
    return problems

def check_correctness(extracted, gt):
    if not extracted:
        return "UNPARSEABLE"
        
    extracted_clean = extracted.lower().replace(',', '').strip()
    gt_clean = gt.lower().strip()
    
    if gt_clean == "no solution":
        return "CORRECT" if "no solution" in extracted_clean else "INCORRECT"
        
    # Extract right side if it's formatted as "x = 5"
    if "=" in extracted_clean:
        extracted_clean = extracted_clean.split("=")[-1].strip()

    # Sympy transformations
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        ex_expr = parse_expr(extracted_clean, transformations=transformations)
        gt_expr = parse_expr(gt_clean, transformations=transformations)
        
        if sympy.simplify(ex_expr - gt_expr) == 0:
            return "CORRECT"
    except Exception:
        pass # Fallback to strict string check if parsing fails

    # Strict fallback for unparseable answers like simple strings/numbers with weird spacing
    # Use exact word match to avoid substring false positives (e.g., matching "1" in "15")
    # For fractions like "1/2", exact match ignoring spaces is good.
    if re.fullmatch(r'\b' + re.escape(gt_clean) + r'\b', extracted_clean):
         return "CORRECT"
    
    if "".join(extracted_clean.split()) == "".join(gt_clean.split()):
         return "CORRECT"

    return "INCORRECT"

def extract_answer(text, gt=None):
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()
        
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return ""
        
    end_text_lines = " ".join(lines[-3:]) # Last ~3 sentences/lines
    
    # Advanced RegEx Patterns
    patterns = [
        r"(?i)the answer is\s*([^.]+)",
        r"(?i)therefore,?\s*(?:we have)?\s*([^.]+)",
        r"(?i)solution is\s*([^.]+)",
        r"\b(?:x|y)\s*=\s*([^\s.]+)",
        r'([-+]?\d*\.?\d+(?:/\d+)?)\s*$' # Number right at the very end
    ]
    
    for pat in patterns:
        match = re.search(pat, end_text_lines)
        if match:
            return match.group(1).replace('"', '').strip()
            
    # Try exact hit of ground truth as fallback
    if gt and gt.lower() in end_text_lines.lower():
        return gt
        
    # As absolute last resort, return the last non-empty line
    return lines[-1]

def generate_response(model, tokenizer, prompt_messages):
    prompt = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2, 
            do_sample=False, # Greedy decoding
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def generate_pdf_report(summary_metrics):
    doc = SimpleDocTemplate("evaluation_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("<b>Model Evaluation Report — Personalized Math Tutor Fine-Tuning</b>", styles['Title']))
    elements.append(Paragraph(f"{datetime.now().strftime('%Y-%m-%d')} | 20 held-out problems, greedy decoding, SymPy-verified grading", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Summary Table
    elements.append(Paragraph("<b>Overall Summary</b>", styles['Heading2']))
    
    summary_data = [["Model", "Overall Accuracy (%)", "Unparseable Count", "Avg Response Length"]]
    for row in summary_metrics:
        summary_data.append([
            str(row["model_name"]),
            f"{row['overall_accuracy_%']}%",
            str(row["unparseable_count"]),
            f"{row['avg_response_length']} chars"
        ])
        
    t_summary = Table(summary_data, colWidths=[100, 120, 120, 130])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f3f3f3")),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 30))
    
    # Topic Table
    elements.append(Paragraph("<b>Per-Topic Accuracy Breakdown (%)</b>", styles['Heading2']))
    topics = [k.replace('acc_', '').replace('_%', '') for k in summary_metrics[0].keys() if k.startswith('acc_') and not k.startswith('conf_')]
    
    topic_data = [["Model"] + [t.replace("_", " ").title() for t in topics]]
    
    for row in summary_metrics:
        t_vals = [f"{row.get('acc_'+t+'_%', 0)}%" for t in topics]
        topic_data.append([row["model_name"]] + t_vals)
        
    num_topics = len(topics)
    col_width = 400 / num_topics if num_topics > 0 else 60
    t_topic = Table(topic_data, colWidths=[70] + [col_width]*num_topics)
    t_topic.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1f497d")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f3f3f3")),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t_topic)
    elements.append(Spacer(1, 30))
    
    # Programmatic Interpretation
    elements.append(Paragraph("<b>Evaluation Insights</b>", styles['Heading2']))
    
    sorted_models = sorted(summary_metrics, key=lambda x: x["overall_accuracy_%"], reverse=True)
    best_model = sorted_models[0]["model_name"]
    best_acc = sorted_models[0]["overall_accuracy_%"]
    
    base_metrics = next((m for m in summary_metrics if m["model_name"] == "base"), None)
    
    insights = []
    insights.append(f"The <b>{best_model}</b> model scored highest overall with an accuracy of {best_acc}%.")
    
    if base_metrics and len(sorted_models) > 1:
        for m in sorted_models:
            if m["model_name"] != "base":
                delta = m["overall_accuracy_%"] - base_metrics["overall_accuracy_%"]
                direction = "improvement" if delta > 0 else "drop"
                insights.append(f"Compared to the base model, <b>{m['model_name']}</b> showed a {abs(delta):.1f}% {direction} in overall accuracy.")
                
                topic_diffs = {}
                for t in topics:
                    t_key = f"acc_{t}_%"
                    t_diff = m.get(t_key, 0) - base_metrics.get(t_key, 0)
                    topic_diffs[t] = t_diff
                
                if topic_diffs:
                    best_topic = max(topic_diffs, key=topic_diffs.get)
                    worst_topic = min(topic_diffs, key=topic_diffs.get)
                    
                    t_insights = []
                    if topic_diffs[best_topic] > 0:
                        t_insights.append(f"It saw its biggest improvement critically tracking <i>{best_topic.replace('_', ' ')}</i> (+{topic_diffs[best_topic]:.1f}%)")
                    if topic_diffs[worst_topic] < 0:
                        t_insights.append(f"but experienced an accuracy drop in <i>{worst_topic.replace('_', ' ')}</i> ({topic_diffs[worst_topic]:.1f}%)")
                    if t_insights:
                        insights.append(" " + ", ".join(t_insights) + ".")
                    
    elements.append(Paragraph(" ".join(insights), styles['Normal']))
    doc.build(elements)

def main():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_v1_path = "./llama3-math-tutor-adapter"
    adapter_v2_path = "./llama3-math-tutor-adapter-v2"
    
    variants = ["base"]
    if os.path.exists(adapter_v1_path):
        variants.append("v1")
    else:
        print(f"Note: {adapter_v1_path} not found. Skipping v1 adapter.")
        
    if os.path.exists(adapter_v2_path):
        variants.append("v2")
    else:
        print(f"Note: {adapter_v2_path} not found. Skipping v2 adapter.")

    print("\nLoading quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model: {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = base_model
    # If we have any adapters, load them sequentially into a PeftModel
    if "v1" in variants or "v2" in variants:
        # Load the first found adapter to create the PeftModel wrapper
        first_adapter = "v1" if "v1" in variants else "v2"
        first_path = adapter_v1_path if first_adapter == "v1" else adapter_v2_path
        
        print(f"Initializing PeftModel with adapter '{first_adapter}' from {first_path}...")
        model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_adapter)
        
        # Load the second adapter if both are present
        if first_adapter == "v1" and "v2" in variants:
            print(f"Loading adapter 'v2' from {adapter_v2_path}...")
            model.load_adapter(adapter_v2_path, adapter_name="v2")

    model.eval()
    test_suite = get_test_set()
    
    all_results = []
    summary_metrics = []

    system_tutor = "You are a personalized math tutor that adapts explanations to each learner."
    system_base = "You are a helpful math tutor. Solve the problem step by step."

    print(f"\nSTARTING EVALUATION against {len(test_suite)} held-out problems...")
    
    for variant in variants:
        print(f"\n--- Testing variant: {variant.upper()} ---")
        
        # Switch adapters or disable them
        if variant == "base":
            if isinstance(model, PeftModel):
                context_manager = model.disable_adapter()
            else:
                import contextlib
                context_manager = contextlib.nullcontext()
        else:
            if isinstance(model, PeftModel):
                model.set_adapter(variant)
                import contextlib
                context_manager = contextlib.nullcontext()
            else:
                import contextlib
                context_manager = contextlib.nullcontext()
                
        metrics = {
            "variant": variant,
            "correct": 0, "incorrect": 0, "unparseable": 0,
            "topic_correct": {}, "topic_total": {},
            "conf_correct": {}, "conf_total": {},
            "total_chars": 0
        }
        
        with context_manager:
            for idx, prob in enumerate(test_suite, start=1):
                if variant == "base":
                    messages = [
                        {"role": "system", "content": system_base},
                        {"role": "user", "content": f"Problem: {prob['text']}"}
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
                        {"role": "user", "content": user_content}
                    ]
                
                output_text = generate_response(model, tokenizer, messages)
                extracted_ans = extract_answer(output_text, prob["gt"])
                status = check_correctness(extracted_ans, prob["gt"])
                
                is_unparseable = (not extracted_ans) or (status == "UNPARSEABLE")
                is_suspicious = False
                
                if not is_unparseable and status != "CORRECT":
                    is_gt_numeric = bool(re.fullmatch(r'[-+]?\d+(?:/\d+)?', str(prob['gt']).replace(' ', '')))
                    has_letters_or_vars = bool(re.search(r'[a-zA-Z=]', extracted_ans))
                    if is_gt_numeric and has_letters_or_vars:
                        is_suspicious = True

                if is_unparseable:
                    print(f"\n[UNPARSEABLE FLAG] Model: {variant} | Prob {prob['id']}")
                    print(f"--- RAW TEXT ---\n{output_text}\n-----------------")
                    print(f"Extracted: '{extracted_ans}' | Ground Truth: '{prob['gt']}'\n")
                    status = "UNPARSEABLE"
                elif is_suspicious:
                    print(f"\n[SUSPICIOUS FLAG] Type Mismatch - Model: {variant} | Prob {prob['id']}")
                    print(f"Extracted: '{extracted_ans}' | Ground Truth: '{prob['gt']}'\n")
                
                # Record detailed result
                all_results.append({
                    "model_variant": variant,
                    "problem_id": prob["id"],
                    "topic": prob["topic"],
                    "ground_truth": prob["gt"],
                    "extracted_answer": extracted_ans,
                    "status": status,
                    "full_response": output_text
                })
                
                # Update metrics
                if status == "CORRECT":
                    metrics["correct"] += 1
                elif status == "INCORRECT":
                    metrics["incorrect"] += 1
                else:
                    metrics["unparseable"] += 1
                    
                topic = prob["topic"]
                conf = prob["profile"]["confidence"]
                
                metrics["topic_total"][topic] = metrics["topic_total"].get(topic, 0) + 1
                if status == "CORRECT":
                    metrics["topic_correct"][topic] = metrics["topic_correct"].get(topic, 0) + 1
                    
                metrics["conf_total"][conf] = metrics["conf_total"].get(conf, 0) + 1
                if status == "CORRECT":
                    metrics["conf_correct"][conf] = metrics["conf_correct"].get(conf, 0) + 1
                    
                metrics["total_chars"] += len(output_text)
                print(f"[{variant}] Prob {idx}/{len(test_suite)} -> {status} (GT: {prob['gt']} | Ext: {extracted_ans})")

        # Compile flat summary for this variant
        total_probs = len(test_suite)
        row = {
            "model_name": variant,
            "overall_accuracy_%": round((metrics["correct"] / total_probs) * 100, 2),
            "unparseable_count": metrics["unparseable"],
            "avg_response_length": round(metrics["total_chars"] / total_probs, 1)
        }
        
        # Add topics
        for t in metrics["topic_total"]:
            t_acc = (metrics["topic_correct"].get(t, 0) / metrics["topic_total"][t]) * 100
            row[f"acc_{t}_%"] = round(t_acc, 1)
            
        # Add confidences
        for c in metrics["conf_total"]:
            c_acc = (metrics["conf_correct"].get(c, 0) / metrics["conf_total"][c]) * 100
            row[f"conf_acc_{c}_%"] = round(c_acc, 1)
            
        summary_metrics.append(row)

    print("\nSaving raw results to evaluation_results_raw.json...")
    with open("evaluation_results_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
        
    print("Saving summary to evaluation_metrics_summary.csv...")
    if summary_metrics:
        keys = summary_metrics[0].keys()
        with open("evaluation_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_metrics)
            
    print("Generating PDF report...")
    if summary_metrics:
        generate_pdf_report(summary_metrics)

    # Print Clean Final Screen Comparison
    print("\n================================= FINAL EVALUATION METRICS =================================")
    
    # Extract topics generically from first summary
    topics = [k.replace('acc_', '').replace('_%', '') for k in summary_metrics[0].keys() if k.startswith('acc_') and not k.startswith('conf_')]
    
    headers = ["Model", "Overall Acc (%)", "Unparseable"] + [f"{t.title()}" for t in topics] + ["Avg Length"]
    header_format = "{:<10} | {:<15} | {:<12} | " + " | ".join(["{:<14}"] * len(topics)) + " | {:<12}"
    
    print("-" * 128)
    print(header_format.format(*headers))
    print("-" * 128)
    
    for row in summary_metrics:
        t_vals = [f"{row.get('acc_'+t+'_%', 0)}%" for t in topics]
        print(header_format.format(
            row["model_name"],
            f"{row['overall_accuracy_%']}%",
            row["unparseable_count"],
            *t_vals,
            f"{row['avg_response_length']} chars"
        ))
    print("-" * 128)
    print("\nEvaluation completed successfully!\n")

if __name__ == "__main__":
    main()

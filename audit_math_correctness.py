import json
import os
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

def clean_math_text(text):
    text = text.replace('²', '**2').replace('³', '**3')
    return text

def parse_equation(eq_text):
    parts = eq_text.split('=')
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    if len(parts) == 1:
        return parse_expr(parts[0], transformations=transformations), None
    elif len(parts) == 2:
        lhs = parse_expr(parts[0], transformations=transformations)
        rhs = parse_expr(parts[1], transformations=transformations)
        return lhs, rhs
    return None, None

def verify_equivalence(problem_text, answer_text):
    prob_clean = clean_math_text(problem_text.strip())
    ans_clean = clean_math_text(answer_text.strip())
    
    if '±' in ans_clean:
        return None # Hard to parse cleanly, delegate to manual review
        
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    try:
        if prob_clean.lower().startswith('factor:'):
            expr_str = prob_clean.lower().replace('factor:', '').strip()
            prob_expr = parse_expr(expr_str, transformations=transformations)
            ans_expr = parse_expr(ans_clean, transformations=transformations)
            
            if sympy.simplify(prob_expr - ans_expr) == 0:
                return True
            else:
                return False
                
        elif prob_clean.lower().startswith('solve the system:'):
            lines = [line.strip() for line in prob_clean.split('\n') if line.strip() and not line.lower().startswith('solve')]
            eqs = []
            for line in lines:
                lhs, rhs = parse_equation(line)
                if lhs is not None and rhs is not None:
                    eqs.append(sympy.Eq(lhs, rhs))
            
            ans_dict = {}
            parts = ans_clean.split(',')
            for p in parts:
                if '=' in p:
                    v, val = p.split('=')
                    v_clean = v.split()[-1]
                    ans_dict[sympy.Symbol(v_clean)] = parse_expr(val.strip(), transformations=transformations)
                    
            if len(eqs) > 0 and len(ans_dict) > 0:
                for eq in eqs:
                    # Substitute solutions back into both sides of each equation
                    if sympy.simplify(eq.lhs.subs(ans_dict) - eq.rhs.subs(ans_dict)) != 0:
                        return False
                return True

        elif prob_clean.lower().startswith('solve'):
            eq_str = prob_clean.lower()
            eq_str = re.sub(r'^solve for [a-z]+:', '', eq_str).strip()
            eq_str = re.sub(r'^solve', '', eq_str).strip()
            
            lhs, rhs = parse_equation(eq_str)
            if lhs is None or rhs is None:
                return None
                
            prob_eq = sympy.Eq(lhs, rhs)
            
            if '=' in ans_clean:
                roots = []
                for part in ans_clean.split(','):
                    if '=' in part:
                        v, val = part.split('=')
                        val_expr = parse_expr(val.strip(), transformations=transformations)
                        v_clean = v.split()[-1]
                        sym_var = sympy.Symbol(v_clean)
                        roots.append((sym_var, val_expr))
                
                if roots:
                    for sym_var, val_expr in roots:
                        sub_lhs = prob_eq.lhs.subs(sym_var, val_expr)
                        sub_rhs = prob_eq.rhs.subs(sym_var, val_expr)
                        if sympy.simplify(sub_lhs - sub_rhs) != 0:
                            return False
                    return True
            else:
                # E.g. "Final Answer: 3"
                val_match = re.search(r'[-+]?\d*\.?\d+(?:/\d+)?', ans_clean)
                if val_match:
                    val_expr = parse_expr(val_match.group(0), transformations=transformations)
                    free_syms = prob_eq.free_symbols
                    if len(free_syms) == 1:
                        sym_var = list(free_syms)[0]
                        sub_lhs = prob_eq.lhs.subs(sym_var, val_expr)
                        sub_rhs = prob_eq.rhs.subs(sym_var, val_expr)
                        if sympy.simplify(sub_lhs - sub_rhs) != 0:
                            return False
                        return True

    except Exception as e:
        # If SymPy fails to parse text (like '12 liters blue'), delegate to manual
        return None
        
    return None

def main():
    jsonl_path = "data/clean/train.jsonl"
    
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    auto_correct = 0
    auto_wrong = 0
    manual_review = 0
    
    wrong_records = []
    manual_records = []

    print("Auditing math correctness in training data...")
    print("This may take a few seconds due to algebraic parsing...\n")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
                
            data = json.loads(line)
            messages = data.get("messages", [])
            
            problem_text = ""
            assistant_full = ""
            final_answer = ""
            
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    if "Problem:\n" in content:
                        problem_text = content.split("Problem:\n")[1].strip()
                elif msg["role"] == "assistant":
                    assistant_full = msg["content"]
                    if "Final Answer:" in assistant_full:
                        final_answer = assistant_full.split("Final Answer:")[1].strip()
                        
            if not problem_text or not final_answer:
                manual_review += 1
                continue
                
            status = verify_equivalence(problem_text, final_answer)
            
            if status is True:
                auto_correct += 1
            elif status is False:
                auto_wrong += 1
                wrong_records.append({
                    "idx": idx+1,
                    "problem": problem_text,
                    "assistant_full": assistant_full
                })
            else:
                manual_review += 1
                manual_records.append({
                    "idx": idx+1,
                    "problem": problem_text,
                    "answer": final_answer
                })
                
    print(f"=== AUDIT SUMMARY ===")
    print(f"Total Examples Checked: {auto_correct + auto_wrong + manual_review}")
    print(f"Auto-Verified CORRECT:  {auto_correct}")
    print(f"Auto-Verified WRONG:    {auto_wrong}")
    print(f"Needs Manual Review:    {manual_review}")
    print("======================\n")
    
    if auto_wrong > 0:
        print("=== WRONG ANSWERS ===\n")
        for wr in wrong_records:
            print(f"Example #{wr['idx']}")
            print(f"Problem: {wr['problem']}")
            print(f"Assistant Output:\n{wr['assistant_full']}")
            print("-" * 50)
            
    if manual_review > 0:
        print("\n=== NEEDS MANUAL REVIEW (Spot check these) ===")
        for mr in manual_records:
            print(f"Problem: {mr['problem']}")
            print(f"Stated Answer: {mr['answer']}")
            print("-")

if __name__ == "__main__":
    main()

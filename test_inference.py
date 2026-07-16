import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "./llama3-math-tutor-adapter"

    print("Loading quantization config...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model: {base_model_id}")
    # Load base model with 4-bit quantization just like during training
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map="auto"
    )

    print(f"Loading tokenizer from: {adapter_path}")
    # Load tokenizer directly from the adapter path to ensure it matches
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from: {adapter_path}")
    # Load adapter onto the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Define the 6 test cases with varied attributes and problems
    test_cases = [
        {
            "profile": "Visual, high confidence, concise",
            "problem": "Find the intersection point of the lines y = 2x - 3 and y = -x + 6."
        },
        {
            "profile": "Analytical, low confidence, detailed",
            "problem": "If 5 bakers can decorate 20 cakes in 4 hours, how many hours would it take 3 bakers to decorate 30 cakes?"
        },
        {
            "profile": "Reflective, intermediate level",
            "problem": "Factor the quadratic expression completely: 2x^2 + 7x - 15."
        },
        {
            "profile": "Conceptual, beginner level",
            "problem": "A car rental company charges $30 per day and an additional $0.15 per mile driven. If you rent the car for 3 days and the total bill comes out to $135, how many miles did you drive?"
        },
        {
            "profile": "Socratic pedagogical strategy, high confidence, detailed",
            "problem": "Solve for x: 3^(2x-1) = 27."
        },
        {
            "profile": "Direct instruction, beginner level, concise",
            "problem": "A bag contains 4 red marbles, 5 blue marbles, and 3 green marbles. What is the probability of randomly drawing a blue marble, returning it to the bag, and then drawing a red marble?"
        }
    ]

    system_message = "You are a personalized math tutor that adapts explanations to each learner."

    print("\nStarting generation for test cases...\n")
    for i, test in enumerate(test_cases, 1):
        print("=" * 80)
        print(f"TEST CASE {i}")
        print(f"Learner Profile: {test['profile']}")
        print(f"Problem: {test['problem']}")
        print("-" * 80)

        # Match the exact format used during training
        user_content = f"Student Profile: {test['profile']}\n\nProblem: {test['problem']}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        # Apply Llama 3 chat template
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Slice the output to only extract the newly generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print("GENERATED RESPONSE:")
        print(response.strip())
        print("=" * 80 + "\n")

    print("""
=== MANUAL REVIEW CHECKLIST ===
[ ] Is the math actually correct?
[ ] Does tone/depth match the stated confidence level?
[ ] Does detail level match "concise" vs "detailed"?
[ ] Does it reflect the stated learning style (visual = spatial language, analytical = formula-first, etc.)?
[ ] Is the response complete (not cut off/truncated)?
[ ] Does it avoid repeating the same encouragement phrase every time?
===============================
""")

if __name__ == "__main__":
    main()

import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

# ==============================================================================
# DATACLASSES
# ==============================================================================

@dataclass
class StudentProfile:
    learning_style: str
    level: str
    confidence: str
    detail_preference: str
    characteristic: str
    pedagogical_style: str

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

LEARNING_STYLES = ["Visual", "Analytical", "Conceptual", "Reflective"]
LEVELS = ["Beginner", "Intermediate", "Advanced"]
CONFIDENCE_LEVELS = ["Low", "High"]
DETAIL_PREFS = ["concise", "detailed"]
CHARACTERISTICS = ["Needs encouragement", "Makes arithmetic mistakes", "None"]
PEDAGOGICAL_STYLES = [
    "Worked Example", "Guided Discovery", "Socratic Questioning", 
    "Concept First", "Example First", "Hint-Based", "Error-Driven", "Verification First"
]

INSTRUCTION_TEMPLATES = [
    "Teach this problem.",
    "Help the student solve this.",
    "Explain this concept.",
    "Guide the learner.",
]

METHOD_RANKING = {
    "Visual": ["graphical", "area_model", "model_drawing", "algebraic"],
    "Analytical": ["algebraic", "substitution", "elimination", "factoring"],
    "Conceptual": ["balance_scale", "area_model", "model_drawing", "algebraic"],
    "Reflective": ["verification", "working_backwards", "algebraic"]
}

# ==============================================================================
# STRATEGY CLASSES
# ==============================================================================

class TeachingStrategy(ABC):
    def __init__(self, profile: StudentProfile, method_data: Dict, correct_answer: str):
        self.profile = profile
        self.method_data = method_data
        self.correct_answer = correct_answer
        self.steps = method_data.get('steps', [])
        self.explanation = method_data.get('explanation', '')
        self.mistakes = method_data.get('common_mistakes', [])
        self.concepts = method_data.get('key_concepts', [])
        
    def _apply_level_pacing(self, step_text: str, index: int, total: int) -> str:
        if self.profile.level == "Beginner":
            transitions = {0: "First, ", total-1: "Finally, "}
            t = transitions.get(index, random.choice(["Next, ", "Then, ", "Now, "]))
            return f"{t}{step_text.lower()}" if step_text else ""
        elif self.profile.level == "Advanced":
            return step_text
        return step_text

    def _inject_mistake(self, step_text: str, index: int, total: int) -> str:
        if self.mistakes and index == max(1, total // 2):
            mistake = self.mistakes[0].lower().rstrip('.')
            return step_text + f"\n[Tutor Note: At this exact step, many students accidentally {mistake}. Let's avoid that.]"
        return step_text

    def _inject_confidence(self, step_text: str, index: int, total: int) -> str:
        if self.profile.confidence == "Low" and index == 1:
            return "You are doing great so far. " + step_text
        elif self.profile.confidence == "High" and index == total - 2:
            return "Notice how this shortcut sets up the end. " + step_text
        return step_text

    def build_steps_block(self) -> str:
        if not self.steps:
            return ""
        formatted = []
        for i, step in enumerate(self.steps):
            s = self._apply_level_pacing(step, i, len(self.steps))
            s = self._inject_confidence(s, i, len(self.steps))
            s = self._inject_mistake(s, i, len(self.steps))
            formatted.append(f"{i+1}. {s}")
        return "\n".join(formatted)
        
    def build_concept_block(self) -> str:
        ret = f"Understanding the WHY: {self.explanation}"
        if self.profile.detail_preference == "detailed" and self.concepts:
            ret += "\nKey Ideas to connect: " + ", ".join(self.concepts)
        return ret
        
    def build_error_block(self) -> str:
        if self.mistakes:
            return f"Let's look at a common trap. Many students {self.mistakes[-1].lower().rstrip('.')}. This is mathematically invalid. Let's do it correctly."
        return "Let's make sure we build this on solid footing."
        
    def build_hint_block(self) -> str:
        return f"Hint: To start, think about utilizing the {self.method_data.get('method_name', 'standard')} method."

    def build_socratic_block(self) -> str:
        if not self.steps:
            return ""
        dialogue = []
        dialogue.append("Tutor: Let's figure this out together. What operation should we perform first?")
        for i, step in enumerate(self.steps):
            if i % 2 == 0:
                short_step = step[:30].lower() + "..." if len(step) > 30 else step.lower()
                dialogue.append(f"...\nStudent: Perhaps we should {short_step}?")
                dialogue.append(f"Tutor: Excellent reasoning. Specifically: {step}")
            else:
                s = self._apply_level_pacing(step, i, len(self.steps))
                dialogue.append(f"Tutor: Now moving forward... {s}")
        return "\n\n".join(dialogue)

    @abstractmethod
    def get_custom_blocks(self) -> Dict[str, str]:
        pass
        
    def generate_response(self) -> str:
        blocks = self.get_custom_blocks()
        p_style = self.profile.pedagogical_style
        
        # Style overrides
        if p_style == "Socratic Questioning":
            blocks["steps"] = self.build_socratic_block()
            
        components = []
        
        if p_style == "Error-Driven":
            components.append(self.build_error_block())
            components.append(blocks.get("steps", ""))
            
        elif p_style == "Concept First":
            components.append(blocks.get("concept", ""))
            components.append(blocks.get("steps", ""))
            
        elif p_style == "Guided Discovery":
            components.append(self.build_hint_block())
            components.append(blocks.get("steps", ""))
            components.append("Did you notice how that worked out?")
            
        elif p_style == "Verification First":
            components.append(f"The final outcome will be {self.correct_answer}. Let's prove why.")
            components.append(blocks.get("steps", ""))
            
        elif p_style == "Example First":
            components.append("Here is exactly how this is done:")
            components.append(blocks.get("steps", ""))
            components.append(blocks.get("concept", ""))
            
        else: # Worked Example
            components.append(blocks.get("steps", ""))
            components.append(blocks.get("concept", ""))

        if p_style != "Verification First":
            components.append(f"\nFinal Answer: {self.correct_answer}")
            
        return "\n\n".join([c for c in components if c])


class VisualTeacher(TeachingStrategy):
    def get_custom_blocks(self) -> Dict[str, str]:
        steps_text = self.build_steps_block()
        ans = str(self.correct_answer).lower()
        
        diagram = ""
        if "=" in ans:
            diagram = "[ Left Side ] === [ Right Side ]\n(Imagine maintaining the balance as we proceed)"
        elif "/" in ans or ":" in ans or "ratio" in ans:
            diagram = "████  vs  ████\n(Comparing the proportions)"
        elif "miles" in ans or "hour" in ans:
            diagram = "Distance: |--------|--------|--------| \n(Visualizing the number line timeline)"
        else:
            diagram = "[ Initial State ] ---> [ Transformation ] ---> [ Result ]\n(Mental progression box)"
            
        steps_text = diagram + "\n\n" + steps_text
        concept = self.build_concept_block().replace("Understanding", "Visualizing")
        return {"steps": steps_text, "concept": concept}


class AnalyticalTeacher(TeachingStrategy):
    def get_custom_blocks(self) -> Dict[str, str]:
        steps_text = f"Mathematical Workflow:\n{self.build_steps_block()}"
        concept_text = "Properties and Invariants: " + self.explanation
        if self.concepts and self.profile.detail_preference == "detailed":
            concept_text += f"\nRigorous Definitions: {', '.join(self.concepts)}"
        return {"steps": steps_text, "concept": concept_text}


class ConceptualTeacher(TeachingStrategy):
    def get_custom_blocks(self) -> Dict[str, str]:
        steps_text = self.build_steps_block()
        concept_text = f"The core 'Why': {self.explanation}"
        return {"steps": steps_text, "concept": concept_text}


class ReflectiveTeacher(TeachingStrategy):
    def get_custom_blocks(self) -> Dict[str, str]:
        steps_text = self.build_steps_block()
        concept_text = self.build_concept_block()
        steps_text += "\n\nReflection Question: Take a moment. Could another method have solved this more efficiently?"
        return {"steps": steps_text, "concept": concept_text}


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def get_strategy(profile: StudentProfile, method_data: Dict, answer: str) -> TeachingStrategy:
    mapping = {
        "Visual": VisualTeacher,
        "Analytical": AnalyticalTeacher,
        "Conceptual": ConceptualTeacher,
        "Reflective": ReflectiveTeacher
    }
    strategy_class = mapping.get(profile.learning_style, ConceptualTeacher)
    return strategy_class(profile, method_data, answer)

def load_dataset(filepath: Path) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_student_profile() -> StudentProfile:
    if random.random() < 0.90:
        level = random.choice(["Beginner", "Intermediate", "Advanced"])
        if level == "Beginner":
            conf = "Low"
            char = random.choice(["Needs encouragement", "Makes arithmetic mistakes"])
        elif level == "Intermediate":
            conf = random.choice(["Low", "High"])
            char = random.choice(CHARACTERISTICS)
        else:
            conf = "High"
            char = random.choice(["None", "Makes arithmetic mistakes"])
        
        return StudentProfile(
            learning_style=random.choice(LEARNING_STYLES),
            level=level,
            confidence=conf,
            detail_preference=random.choice(DETAIL_PREFS),
            characteristic=char,
            pedagogical_style=random.choice(PEDAGOGICAL_STYLES)
        )
    else: # 10% fully randomized
        return StudentProfile(
            learning_style=random.choice(LEARNING_STYLES),
            level=random.choice(LEVELS),
            confidence=random.choice(CONFIDENCE_LEVELS),
            detail_preference=random.choice(DETAIL_PREFS),
            characteristic=random.choice(CHARACTERISTICS),
            pedagogical_style=random.choice(PEDAGOGICAL_STYLES)
        )

def select_method(methods: Dict[str, Dict], profile: StudentProfile) -> Tuple[str, Dict]:
    preferred_methods = METHOD_RANKING.get(profile.learning_style, [])
    for method_key in preferred_methods:
        if method_key in methods:
            return method_key, methods[method_key]
    available_keys = list(methods.keys())
    if not available_keys:
        raise ValueError("No methods provided in problem!")
    fallback_key = random.choice(available_keys)
    return fallback_key, methods[fallback_key]

def generate_instruction(problem_statement: str, profile: StudentProfile) -> str:
    base_instruction = random.choice(INSTRUCTION_TEMPLATES)
    profile_str = (
        f"Student Profile:\n"
        f"- Learning Style: {profile.learning_style}\n"
        f"- Confidence: {profile.confidence}\n"
        f"- Level: {profile.level}\n"
        f"- Detail Preference: {profile.detail_preference}\n"
        f"- Pedagogical Strategy: {profile.pedagogical_style}\n"
    )
    if profile.characteristic != "None":
        profile_str += f"- Characteristic: {profile.characteristic}\n"

    return f"{profile_str}\n{base_instruction}\n\nProblem:\n{problem_statement}"

def convert_problem(problem: Dict, num_variations: int) -> List[Dict]:
    examples = []
    methods = {k: v for k, v in problem.items() if isinstance(v, dict) and 'method_name' in v}
    if not methods:
        return examples
        
    problem_statement = problem.get('problem_statement', '')
    correct_answer = problem.get('correct_answer', '')
    
    used_prompts = set()
    attempts = 0
    max_attempts = num_variations * 3
    
    while len(examples) < num_variations and attempts < max_attempts:
        attempts += 1
        
        profile = generate_student_profile()
        method_key, selected_method = select_method(methods, profile)
        instruction = generate_instruction(problem_statement, profile)
        
        if instruction in used_prompts:
            continue
            
        used_prompts.add(instruction)
        
        strategy = get_strategy(profile, selected_method, correct_answer)
        response = strategy.generate_response()
        
        chat_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a personalized math tutor that adapts explanations to each learner."
                },
                {
                    "role": "user",
                    "content": instruction
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        }
        examples.append(chat_example)
        
    return examples

def save_jsonl(examples: List[Dict], filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "MASTER_DATASET.json"))
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "generated" / "train.jsonl"))
    parser.add_argument("--variations", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    problems = load_dataset(input_path)
    print(f"Generating {args.variations} variations...")
    
    all_examples = []
    for problem in problems:
        all_examples.extend(convert_problem(problem, args.variations))

    random.shuffle(all_examples)
    save_jsonl(all_examples, output_path)
    
    print(f"Done! Saved {len(all_examples)} examples to {output_path.name}")

if __name__ == "__main__":
    main()

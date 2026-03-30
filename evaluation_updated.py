import re
from typing import List, Dict, Set

class ClinicalTrajectoryEvaluator:
    """
    Evaluates the reasoning process of the Doctor Agent across the entire multi-turn simulation.
    This goes beyond simple "Correct/Incorrect" to measure *how* the model thinks.
    """

    def __init__(self):
        # Define basic/cheap vs advanced/expensive tests for the Test Rationality metric
        self.basic_tests = {"cbc", "bmp", "vitals", "blood pressure", "temperature", "x-ray", "ecg", "ekg"}
        self.advanced_tests = {"mri", "ct", "ct scan", "biopsy", "endoscopy", "lumbar puncture", "pet scan"}

    def compute_diagnostic_stability(self, differential_trajectory: List[List[str]]) -> float:
        """
        Metric 1: Diagnostic Stability Score (0.0 to 1.0)
        Measures how wildly the LLM changes its mind between turns.
        Calculated using the average Jaccard Similarity between consecutive differential diagnoses.
        - High score (~1.0): The model logically refines a stable set of hypotheses.
        - Low score (~0.0): The model is hallucinating and wildly changing guesses every turn.
        """
        if len(differential_trajectory) < 2:
            return 1.0 # Perfectly stable if only 1 turn

        stability_scores =[]
        for i in range(1, len(differential_trajectory)):
            prev_set = set(d.lower().strip() for d in differential_trajectory[i-1])
            curr_set = set(d.lower().strip() for d in differential_trajectory[i])
            
            # Skip empty sets
            if not prev_set and not curr_set:
                stability_scores.append(1.0)
                continue
            if not prev_set or not curr_set:
                stability_scores.append(0.0)
                continue
            
            # Jaccard Similarity: (Intersection) / (Union)
            intersection = len(prev_set.intersection(curr_set))
            union = len(prev_set.union(curr_set))
            stability_scores.append(intersection / union if union > 0 else 0)

        # Return the average stability across the whole conversation
        return sum(stability_scores) / len(stability_scores)

    def compute_test_rationality(self, tests_ordered: List[str]) -> float:
        """
        Metric 2: Test Rationality Score (0.0 to 1.0)
        Checks if the LLM follows standard clinical protocols (cheap/non-invasive tests first).
        Penalizes the model if it orders an MRI or Biopsy before checking basic vitals or bloodwork.
        """
        if not tests_ordered:
            return 1.0 # No tests ordered, no irrationality
            
        found_advanced_first = False
        had_basic_prior = False

        for test in tests_ordered:
            test_lower = test.lower()
            
            # Check if this test is considered advanced
            is_advanced = any(adv in test_lower for adv in self.advanced_tests)
            is_basic = any(bsc in test_lower for bsc in self.basic_tests)
            
            if is_basic:
                had_basic_prior = True
                
            if is_advanced and not had_basic_prior:
                # The model ordered an expensive/invasive test before any basic workup!
                found_advanced_first = True
                break

        # Returns 0.0 if they jumped straight to a CT scan, 1.0 if they followed protocol.
        return 0.0 if found_advanced_first else 1.0

    def compute_information_efficiency(self, extracted_symptoms: List[str], turn_count: int) -> float:
        """
        Metric 3: Information Efficiency (IE)
        Measures: (Number of relevant clinical data points gathered) / (Total questions asked)
        A highly efficient doctor gets the right symptoms in 3 questions. 
        An inefficient (or hallucinating) model takes 15 questions to get the same data.
        """
        if turn_count == 0:
            return 0.0
        
        # IE = Data Points per Turn
        efficiency = len(extracted_symptoms) / turn_count
        return round(efficiency, 2)

    def generate_full_report(self, trajectory_data: dict) -> dict:
        """
        Takes the aggregated data from a single LangGraph run and returns the new metrics.
        """
        stability = self.compute_diagnostic_stability(trajectory_data.get("differential_trajectory",[]))
        rationality = self.compute_test_rationality(trajectory_data.get("tests_ordered",[]))
        efficiency = self.compute_information_efficiency(
            trajectory_data.get("extracted_symptoms",[]), 
            trajectory_data.get("turn_count", 1)
        )

        return {
            "Diagnostic_Stability": round(stability, 2),
            "Test_Rationality": rationality,
            "Information_Efficiency": efficiency
        }

if __name__ == "__main__":
    # Simulated data extracted from a LangGraph State history
    mock_langgraph_run = {
        "turn_count": 4,
        "differential_trajectory": [
            ["Headache", "Dehydration"],                       # Turn 1["Tension Headache", "Migraine", "Dehydration"],   # Turn 2 (Added detail, decent stability)
            ["Migraine", "Brain Tumor"],                       # Turn 3 (Dropped dehydration, added tumor)
            ["Migraine"]                                       # Turn 4 (Final narrowing)
        ],
        "tests_ordered": ["Blood Pressure", "CBC", "CT Scan"], # Rational: Checked BP and Blood before CT
        "extracted_symptoms":["severe headache", "blurry vision", "nausea"]
    }

    evaluator = ClinicalTrajectoryEvaluator()
    report = evaluator.generate_full_report(mock_langgraph_run)
    
    print("--- ADVANCED METRICS REPORT ---")
    for k, v in report.items():
        print(f"{k}: {v}")
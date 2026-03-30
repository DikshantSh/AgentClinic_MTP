import argparse
import anthropic
import openai, re, random, time, json, replicate, os
from typing import TypedDict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging as hf_logging

# --- LANGGRAPH IMPORTS ---
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    print("CRITICAL ERROR: LangGraph is not installed. Please run: pip install langgraph langchain-core")
    exit(1)

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

## replicate client ##
replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))

hf_logging.set_verbosity_error()

def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype="auto")
    return pipe

def query_model(model_str, prompt, system_prompt, model_class, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False, is_final=False):
    if model_str not in["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "HF_mistralai/Mixtral-8x7B-v0.1", "mixtral-8x7b", "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    
    for _ in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        try:
            if model_str == "gpt4":
                messages =[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt-4o-mini":
                messages =[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt3.5":
                messages =[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "claude3.5sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url, input={
                        "prompt":  prompt, 
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'HF_mistralai/Mixtral-8x7B-v0.1': 
                max_tokens = 200 if is_final else 80   
                answer = model_class.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens
                )
                answer = re.sub(r"\s+", " ", answer) 
                return answer
            return answer
        
        except Exception as e:
            print(f"Model {model_str} inference error: {e}. Retrying...")
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

class LocalServerLLMWrapper:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 75) -> str:
        messages =[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=0.7 
        )
        return output[0]['generated_text'].strip()

MAX_RECENT_CASES = 5

def _stringify_info(info):
    if isinstance(info, str): return info
    try: return json.dumps(info, ensure_ascii=True)
    except TypeError: return str(info)

def _clean_dialogue_response(text: str) -> str:
    if not isinstance(text, str): return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def summarize_case_for_recency(scenario):
    return "Presentation: {} | Diagnosis: {}".format(_stringify_info(scenario.patient_information()), _stringify_info(scenario.diagnosis_information()))

def extract_json_list(text: str) -> List[str]:
    """Robust extractor for open-source models that might fail strict JSON formatting."""
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(f"[{match.group(1)}]")
        except:
            items = match.group(1).split(',')
            return[item.strip().strip('"').strip("'") for item in items if item.strip()]
    return ["Awaiting further info"]

# ----------- DATASET CLASSES (Preserved from v1) -----------
class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    def patient_information(self): return self.patient_info
    def examiner_information(self): return self.examiner_info
    def exam_information(self):
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    def diagnosis_information(self): return self.diagnosis

class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios =[ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self): return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id): return self.sample_scenario() if id is None else self.scenarios[id]


# ----------- AGENT CLASSES (Preserved from v1) -----------
class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()

    def inference_patient(self, question, model_class) -> str:
        prompt = f"\nHere is a history of your dialogue: {self.agent_hist}\n Here was the doctor response: {question}Now please continue your dialogue\nPatient: "
        answer = query_model(self.backend, prompt, self.system_prompt(), model_class)
        answer = _clean_dialogue_response(answer)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are a patient in a clinic who only responds in the form of dialogue. Answer in 1-3 sentences. Never include <think> tags or analysis."
        symptoms = f"\n\nBelow is all of your information. {self.symptoms}. \n\n Do not reveal your exact disease."
        return base + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, recent_cases=None) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.recent_cases = list(recent_cases) if recent_cases else[]
        self.reset()

    def reflect_metacognition(self, latest_input, model_class) -> List[str]:
        """PHASE 2 FEATURE: Hidden reasoning node to update differential diagnosis"""
        sys_prompt = "You are an internal medical logic engine. Based on the history, list the top 3 differential diagnoses. Output ONLY a valid JSON array of strings. Example:[\"Asthma\", \"COPD\", \"Pneumonia\"]"
        prompt = f"History:\n{self.agent_hist}\nLatest Input:\n{latest_input}\n\nUpdate Differential Diagnosis:"
        
        raw_response = query_model(self.backend, prompt, sys_prompt, model_class, is_final=False)
        return extract_json_list(raw_response)

    def inference_doctor(self, question, model_class, is_final=False) -> str:
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        
        if is_final: prompt = f"\nHistory: {self.agent_hist}\nPatient: {question}\n\n⚠️ This is your FINAL question. Provide diagnosis NOW as: DIAGNOSIS READY: [diagnosis]\nDoctor: "
        else: prompt = f"\nHistory: {self.agent_hist}\nPatient: {question}\nContinue your dialogue\nDoctor: "
        
        answer = query_model(self.backend, prompt, self.system_prompt(), model_class, is_final=is_final)
        answer = _clean_dialogue_response(answer)

        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        base = f"You are Dr. Agent. Limit: {self.MAX_INFS} questions. Asked: {self.infs}. Request tests via 'REQUEST TEST: [test]'. Once confident, type 'DIAGNOSIS READY: [diagnosis]'. Speak aloud only."
        presentation = f"\n\nInfo: {self.presentation}"
        return base + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        self.agent_hist = ""
        self.backend = backend_str
        self.scenario = scenario
        self.reset()

    def inference_measurement(self, question, model_class) -> str:
        answer = query_model(self.backend, f"\nHistory: {self.agent_hist}\nDoctor request: {question}", self.system_prompt(), model_class)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        return f"You are a test reader. Format: 'RESULTS: [results]'. Info: {self.information}. If missing, say NORMAL READINGS."
    
    def add_hist(self, hist_str) -> None: self.agent_hist += hist_str + "\n\n"
    def reset(self) -> None: 
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

def compare_results(diagnosis, correct_diagnosis, moderator_llm, model_class):
    prompt = f"\nCorrect: {correct_diagnosis}\nDoctor: {diagnosis}\nAre these the same?"
    sys = "Determine if the doctor diagnosis matches the correct diagnosis. Respond ONLY with Yes or No."
    answer = query_model(moderator_llm, prompt, sys, model_class)
    return answer.lower()


# ----------- PHASE 2: LANGGRAPH STATE MACHINE -----------

class ClinicalState(TypedDict):
    last_input_to_doctor: str  # What the patient/lab just said
    last_doctor_message: str  # What the doctor just asked
    differential_diagnoses: List[str] # The doctor's secret internal thoughts
    turn_count: int # Question counter (max 10 or 20)
    diagnosis_ready: bool # Stop mechanism


def main(api_key, replicate_api_key, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, total_inferences, scenario_id=None):
    # GLOBAL_HF_ENGINE = LocalServerLLMWrapper(model_id="Intelligent-Internet/II-Medical-8B")
    # GLOBAL_HF_ENGINE = None # Assuming API usage for this test, uncomment above if using local
    GLOBAL_HF_ENGINE = LocalServerLLMWrapper(model_id="mistralai/Mistral-7B-Instruct-v0.2")
    openai.api_key = api_key
    if patient_llm in ["llama-3-70b-instruct"]: os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    scenario_loader = ScenarioLoaderMedQA()
    print(f"Loaded {scenario_loader.num_scenarios} scenarios from {dataset}")

    total_correct = 0
    total_presents = 0
    scenario_ids =[scenario_id] if scenario_id is not None else range(0, min(num_scenarios or 1, scenario_loader.num_scenarios))

    for _scenario_id in scenario_ids:
        print(f"\n{'='*50}\nStarting Scenario ID: {_scenario_id}\n{'='*50}")
        total_presents += 1
        scenario = scenario_loader.get_scenario(id=_scenario_id)
        
        # Instantiate Agents
        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(scenario=scenario, backend_str=patient_llm)
        doctor_agent = DoctorAgent(scenario=scenario, backend_str=doctor_llm, max_infs=total_inferences)

        # Defining LangGraph Nodes using closures 
        def doctor_reflection_node(state: ClinicalState) -> dict:
            diff = doctor_agent.reflect_metacognition(state["last_input_to_doctor"], GLOBAL_HF_ENGINE)
            print(f"\n[METACOGNITION] Updated Differential: {diff}")
            return {"differential_diagnoses": diff}

        def doctor_speaker_node(state: ClinicalState) -> dict:
            is_final = state["turn_count"] >= total_inferences - 1
            msg = doctor_agent.inference_doctor(state["last_input_to_doctor"], GLOBAL_HF_ENGINE, is_final=is_final)
            print(f"\nDoctor [Turn {state['turn_count']}]: {msg}")
            return {
                "last_doctor_message": msg,
                "turn_count": state["turn_count"] + 1,
                "diagnosis_ready": "DIAGNOSIS READY" in msg.upper()
            }

        def patient_node(state: ClinicalState) -> dict:
            msg = patient_agent.inference_patient(state["last_doctor_message"], GLOBAL_HF_ENGINE)
            meas_agent.add_hist(msg)
            print(f"\nPatient: {msg}")
            return {"last_input_to_doctor": msg}

        def measurement_node(state: ClinicalState) -> dict:
            msg = meas_agent.inference_measurement(state["last_doctor_message"], GLOBAL_HF_ENGINE)
            patient_agent.add_hist(msg)
            print(f"\nMeasurement: {msg}")
            return {"last_input_to_doctor": msg}

        def route_doctor(state: ClinicalState):
            if state["diagnosis_ready"] or state["turn_count"] >= total_inferences:
                return END
            if "REQUEST TEST" in state["last_doctor_message"].upper():
                return "measurement"
            return "patient"

        # Build Graph 
        workflow = StateGraph(ClinicalState)
        workflow.add_node("doctor_reflection", doctor_reflection_node)
        workflow.add_node("doctor_speaker", doctor_speaker_node)
        workflow.add_node("patient", patient_node)
        workflow.add_node("measurement", measurement_node)

        workflow.set_entry_point("doctor_reflection")
        workflow.add_edge("doctor_reflection", "doctor_speaker")
        workflow.add_conditional_edges("doctor_speaker", route_doctor)
        workflow.add_edge("patient", "doctor_reflection")
        workflow.add_edge("measurement", "doctor_reflection")

        app = workflow.compile()

        # Execute Graph
        # initial state to start the doctor with the examiner information and prompt to begin
        initial_state = {
            "last_input_to_doctor": "Please begin the examination. What would you like to ask or test?",
            "last_doctor_message": "",
            "differential_diagnoses":[],
            "turn_count": 0,
            "diagnosis_ready": False
        }

        final_state = app.invoke(initial_state)

        # --- Evaluation ---
        print("\nDoctor's diagnosis complete.")
        final_diagnosis_str = final_state["last_doctor_message"].split("DIAGNOSIS READY:")[-1].strip()
        
        comparison_result = compare_results(final_diagnosis_str, scenario.diagnosis_information(), moderator_llm, GLOBAL_HF_ENGINE)
        correctness = "yes" in str(comparison_result).lower()
        if correctness: total_correct += 1
        
        print(f"\nDoctor's Final Output: {final_diagnosis_str}")
        print(f"Ground Truth: {scenario.diagnosis_information()}")
        print(f"Result: {'CORRECT' if correctness else 'INCORRECT'} (Current Accuracy: {int((total_correct/total_presents)*100)}%)")

    print(f"\n\nFinal accuracy after {total_presents} scenarios: {int((total_correct/total_presents)*100)}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="YOUR_KEY")
    parser.add_argument('--replicate_api_key', type=str, default="")
    parser.add_argument('--doctor_llm', type=str, default='gpt3.5')
    parser.add_argument('--patient_llm', type=str, default='gpt3.5')
    parser.add_argument('--measurement_llm', type=str, default='gpt3.5')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA')
    parser.add_argument('--num_scenarios', type=int, default=1)
    parser.add_argument('--total_inferences', type=int, default=10)
    parser.add_argument('--scenario_id', type=int, default=None)
    
    args = parser.parse_args()

    main(
        args.openai_api_key, args.replicate_api_key, 
        args.doctor_llm, args.patient_llm, args.measurement_llm, args.moderator_llm, 
        args.num_scenarios, args.agent_dataset, args.total_inferences, args.scenario_id
    )
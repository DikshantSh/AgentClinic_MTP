import argparse
import anthropic
from transformers import pipeline
import openai, re, random, time, json, replicate, os
# from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, logging as hf_logging
import torch

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

## replicate client ##
replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))

hf_logging.set_verbosity_error()

def load_huggingface_model(model_name):
    # pipe = pipeline("text-generation", model=model_name, device_map="cuda:0", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response


def query_model(model_str, prompt, system_prompt, model_class, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False, is_final=False):
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "HF_mistralai/Mixtral-8x7B-v0.1", "mixtral-8x7b", "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    # print(f"model str: {model_str}")
    # print(f"tries: {tries}")
    for _ in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        try:
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                            "image_url": {
                                "url": "{}".format(scene.image_url),
                            },
                        },
                    ]},]
                if model_str == "gpt4v":
                    response = openai.ChatCompletion.create(
                            model="gpt-4-vision-preview",
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                        )
                elif model_str == "gpt-4o-mini":
                    response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                        )
                elif model_str == "gpt4":
                    response = openai.ChatCompletion.create(
                            model="gpt-4-turbo",
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                        )
                elif model_str == "gpt4o":
                    response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=0.05,
                            max_tokens=200,
                        )
                answer = response["choices"][0]["message"]["content"]
            if model_str == "gpt4":
                messages = [
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
            elif model_str == "gpt4v":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt-4o-mini":
                messages = [
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
            elif model_str == "o1-preview":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                        model="o1-preview-2024-09-12",
                        messages=messages,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt3.5":
                messages = [
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
            elif model_str == "gpt4o":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url, input={
                        "prompt":  prompt, 
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            # elif model_str == 'mixtral-8x7b': # HF_mistralai/Mixtral-8x7B-v0.1
            elif model_str == 'HF_mistralai/Mixtral-8x7B-v0.1': # HF_mistralai/Mixtral-8x7B-v0.1
                # Increase max_new_tokens for final diagnosis to ensure enough space
                max_tokens = 200 if is_final else 80   

                # output = replicate_client.run(
                #     mixtral_url, 
                #     input={"prompt": prompt, 
                #             "system_prompt": system_prompt,
                #             "max_new_tokens": 75})
                # answer = ''.join(output)
                # answer = re.sub(r"\s+", " ", answer)
                answer = model_class.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_tokens
                )
                answer = re.sub(r"\s+", " ", answer) 
                return answer
            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url, input={
                        "prompt":  prompt, 
                        "system_prompt": system_prompt,
                        "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif "HF_" in model_str:
                input_text = system_prompt + prompt 
                # print(f"here inside the HF_ elif: {input_text}")
                #if self.pipe is None:
                #    self.pipe = load_huggingface_model(self.backend.replace("HF_", ""))
                raise Exception("Sorry, fixing TODO :3") #inference_huggingface(input_text, self.pipe)
            return answer
        
        except Exception as e:
            print(f"Model {model_str} inference error: {e}. Retrying...")
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

## local LLM wrapper code ##
# This is a conceptual class. It needs to be initialized ONCE globally.



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
            # ... other pipeline args ...
        )

    def generate_response(self, prompt: str, system_prompt: str, max_new_tokens: int = 75) -> str:
        # 1. Format the prompt (Mixtral/Mistral uses a specific Chat Template)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # print(f"messages: {messages}")
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. Run the inference
        output = self.pipe(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False, # Crucial: only return the generated text
            do_sample=True,
            temperature=0.7 
        )
        
        # 3. Clean and return
        # print(f"output: {output[0]['generated_text'].strip()}")
        return output[0]['generated_text'].strip()

# GLOBAL INSTANTIATION (needs to be done once before query_model is called)
# Ensure the model ID matches what is in your command line, e.g., 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# GLOBAL_HF_ENGINE = LocalServerLLMWrapper(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
## end ##

MAX_RECENT_CASES = 5

def _stringify_info(info):
    if isinstance(info, str):
        return info
    try:
        return json.dumps(info, ensure_ascii=True)
    except TypeError:
        return str(info)


def _clean_dialogue_response(text: str) -> str:
    """Strip internal reasoning tags and keep only spoken dialogue."""
    if not isinstance(text, str):
        return text
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove other XML/HTML-style tags such as <Answer>
    text = re.sub(r"</?[^>]+>", "", text)
    # Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_case_for_recency(scenario):
    patient_snapshot = _stringify_info(scenario.patient_information())
    diagnosis_snapshot = _stringify_info(scenario.diagnosis_information())
    return "Presentation: {} | Diagnosis: {}".format(patient_snapshot, diagnosis_snapshot)


class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question, model_class) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt(), model_class)
        answer = _clean_dialogue_response(answer)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length. Speak as if aloud to the doctor. Never include analysis, bullet points, <think> tags, stage directions, or anything besides your spoken reply."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, recent_cases=None) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
        self.recent_cases = list(recent_cases) if recent_cases else []

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return self._recency_bias_prompt()
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def _recency_bias_prompt(self) -> str:
        if not self.recent_cases:
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        recent_text = "\nYour most recent patients were:\n"
        for idx, case in enumerate(self.recent_cases[-MAX_RECENT_CASES:], 1):
            recent_text += "{}. {}\n".format(idx, case)
        return recent_text + "These encounters strongly influence your current judgement and increase your tendency toward recency bias.\n"

    def inference_doctor(self, question, model_class, image_requested=False, is_final=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        
        # Create a more forceful prompt for final diagnosis
        if is_final:
            prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "\n\n⚠️ IMPORTANT: This is your FINAL question. You MUST provide a diagnosis NOW using the format: DIAGNOSIS READY: [diagnosis name]\n\nDoctor: "
        else:
            prompt = "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: "
        
        answer = query_model(self.backend, prompt, self.system_prompt(), model_class, image_requested=image_requested, scene=self.scenario, is_final=is_final)
        answer = _clean_dialogue_response(answer)

        if is_final and "DIAGNOSIS READY" not in answer.upper():
            answer = self._force_final_diagnosis(question, model_class)

        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        # Make the prompt more forceful when it's the last question
        if self.infs >= self.MAX_INFS - 1:
            base = "You are a doctor named Dr. Agent. This is your FINAL question - you MUST provide a diagnosis NOW. You have asked {} out of {} questions. You MUST respond with: DIAGNOSIS READY: [diagnosis name here]. Do not ask more questions. Provide your diagnosis immediately.".format(self.infs, self.MAX_INFS)
        else:
            base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        base += "\n\nSpeak only in natural conversational sentences directed at the patient. Never output analysis, internal thoughts, <think> tags, XML-style markers, or bullet points—only the dialogue you would say aloud."
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

    def _force_final_diagnosis(self, question, model_class) -> str:
        force_prompt = (
            "The dialogue between you (doctor) and the patient is:\n"
            f"{self.agent_hist}\n"
            f"The patient just replied: {question}\n"
            "You have reached the question limit and must now deliver your final diagnosis. "
            "Respond with one concise sentence that starts with 'DIAGNOSIS READY:' followed by your best diagnostic impression. "
            "Do not ask additional questions."
        )
        force_system_prompt = (
            "You are Dr. Agent finalizing a case. You already exhausted all permitted questions and must now state the final diagnosis. "
            "Output EXACTLY one sentence that begins with 'DIAGNOSIS READY:' and contains only the diagnosis name (optionally with a short qualifier)."
        )
        forced_answer = query_model(
            self.backend,
            force_prompt,
            force_system_prompt,
            model_class,
            is_final=True
        )
        forced_answer = _clean_dialogue_response(forced_answer)
        if "DIAGNOSIS READY" not in forced_answer.upper():
            forced_answer = "DIAGNOSIS READY: Unable to determine (insufficient information)."
        return forced_answer


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question, model_class) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt(), model_class)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe, model_class):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.", model_class)
    # print(answer)
    return answer.lower()


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None, scenario_id=None):
    GLOBAL_HF_ENGINE = LocalServerLLMWrapper(model_id="Intelligent-Internet/II-Medical-8B")
    # GLOBAL_HF_ENGINE = LocalServerLLMWrapper(model_id="mistralai/Mistral-7B-Instruct-v0.2")
    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load MedQA, MIMICIV or NEJM agent case scenarios
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    total_correct = 0
    total_presents = 0
    print("Loaded {} scenarios from {}".format(scenario_loader.num_scenarios, dataset))


    # Pipeline for huggingface models
    if "HF_" in moderator_llm:
        # pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
        pipe = load_huggingface_model(mixtral_url)
        print("Loaded huggingface model pipeline for moderator llm")
    else:
        pipe = None
    # Determine which scenario IDs to run
    if scenario_id is not None:
        scenario_ids = [scenario_id]
        print("Starting simulation with 1 scenario (ID: {})...".format(scenario_id))
    else:
        if num_scenarios is None:
            num_scenarios = scenario_loader.num_scenarios
        total_to_run = min(num_scenarios, scenario_loader.num_scenarios)
        scenario_ids = range(0, total_to_run)
        print("Starting simulation with {} scenarios...".format(total_to_run))
    recent_case_summaries = []
    # for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
    for _scenario_id in scenario_ids:
        scenario_start_time = time.time()
        print(f"scenario id: {_scenario_id}")
        total_presents += 1
        pi_dialogue = str()
        # Initialize scenarios (MedQA/NEJM)
        scenario =  scenario_loader.get_scenario(id=_scenario_id)
        # Initialize agents
        meas_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=measurement_llm)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario, 
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences, 
            img_request=img_request,
            recent_cases=recent_case_summaries)

        doctor_dialogue = ""
        
        for _inf_id in range(total_inferences):
            
            print(f"  inference id: {_inf_id}")
            # Check for medical image request
            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in doctor_dialogue
                else: imgs = True
            else: imgs = False
            # Check if final inference
            is_final_inference = (_inf_id == total_inferences - 1)
            if is_final_inference:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_doctor":
                # print("here in if")
                doctor_dialogue = input("\nQuestion for patient: ")
            else: 
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, GLOBAL_HF_ENGINE, image_requested=imgs, is_final=is_final_inference)
            
            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
            # Doctor has arrived at a diagnosis, check correctness
            if "DIAGNOSIS READY" in doctor_dialogue:
                print("Doctor's diagnosis complete.")
                # correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm, pipe, GLOBAL_HF_ENGINE) == "yes"
                # if "Yes" in correctness: total_correct += 1
                comparison_result = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm, pipe, GLOBAL_HF_ENGINE)
                print("Comparison result:", comparison_result)
                correctness = "yes" in str(comparison_result).lower()
                if correctness: total_correct += 1
                print("\nDoctor's diagnosis:", doctor_dialogue.split("DIAGNOSIS READY:")[-1].strip())
                print("Correct answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                break
            # Obtain medical exam from measurement reader
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue,GLOBAL_HF_ENGINE)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
            # Obtain response from patient
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue, GLOBAL_HF_ENGINE)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)
        # Prevent API timeouts
        # time.sleep(1.0)
        scenario_duration = time.time() - scenario_start_time
        print("Scenario {} duration: {:.2f}s".format(_scenario_id, scenario_duration))
        case_summary = summarize_case_for_recency(scenario)
        recent_case_summaries.append(case_summary)
        if len(recent_case_summaries) > MAX_RECENT_CASES:
            recent_case_summaries.pop(0)
        # print("Recent case summaries for recency bias:\n", "\n".join(recent_case_summaries))


    print("\n\nTotal correct inferences: {}".format(total_correct))
    print("Total scenarios presented: {}".format(total_presents))
    print("\nFinal accuracy after {} scenarios: {}%".format(total_presents, int((total_correct/total_presents)*100)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, default="", required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--measurement_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=10, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    parser.add_argument('--scenario_id', type=int, default=None, required=False, help='Run a single specific scenario ID (overrides num_scenarios)')
    
    args = parser.parse_args()

    main(
        args.openai_api_key,
        args.replicate_api_key,
        args.inf_type,
        args.doctor_bias,
        args.patient_bias,
        args.doctor_llm,
        args.patient_llm,
        args.measurement_llm,
        args.moderator_llm,
        args.num_scenarios,
        args.agent_dataset,
        args.doctor_image_request,
        args.total_inferences,
        args.anthropic_api_key,
        args.scenario_id,
    )

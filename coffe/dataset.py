from datasets import load_dataset, load_from_disk
import argparse
import os
import json
import ast
import sys
import time

from coffe.sanitize import CodeVisitor, CodeProcessor
from coffe.config import benchmarks

class Dataset(object):
    def __init__(self, name, data_path = None, testfile_path = None, full = True, train_description_path = None, tmp_dir = "./tmp") -> None:
        self.name = name
        if self.name not in benchmarks:
            raise ValueError(f"Cannot find the benchmark {self.name} in Coffe. Make sure you have registered it in config.py and reinstall Coffe.")
        self.index = -1
        self.tmp_dir = tmp_dir
        self.data_path = data_path
        if self.name in ["function", "file"]:
            self.dataset = json.load(open(os.path.join(data_path, "data.json"), "r"))
        elif not data_path:
            self.dataset = load_dataset(name, cache_dir = "./temp_datasets")
        else:
            self.dataset = load_from_disk(data_path)

        if self.name == "mbpp":
            testfile_path = os.path.join(data_path, "MbppPlus.jsonl")

        if testfile_path and name in ["mbpp", "openai_humaneval"]:
            raw_data = open(testfile_path, "r").read().splitlines()
            data = []
            for line in raw_data:
                data.append(json.loads(line))
            self.testcases = {}
            for instance in data:
                if self.name == "mbpp":
                    self.testcases[instance["task_id"].replace("HumanEval/", "").replace("Mbpp/", "")] = {"inputs": instance["base_input"], "outputs": None, "entry_point": instance["entry_point"]}
                else:
                    self.testcases[instance["task_id"].replace("HumanEval/", "").replace("Mbpp/", "")] = {"inputs": instance["base_input"], "outputs": None}
        else:
            self.testcases = {}

        if full:
            self.prompt2instance = self.get_prompt2instance()
            self.prompt2groundtruth = {}
            self.prompt2testcase = {}
            self.prompt2stressful = {}
            self.prompt2io = {}
            

        self.reset_index()

    def reset_index(self):
        self.index = -1

    def length(self):
        return len(self.dataset)

    def next(self):
        self.index += 1
        finish = False
        if self.index == len(self.dataset) - 1:
            finish = True

        instance = {}
        if self.name in ["openai_humaneval", "mbpp"] and len(self.testcases) > 0:
            for key in self.dataset[self.index]:
                instance[key] = self.dataset[self.index][key]
            if str(instance["task_id"]).replace("HumanEval/", "") in self.testcases:
                instance["testcases"] = self.testcases[str(instance["task_id"]).replace("HumanEval/", "")]
                if "entry_point" in instance["testcases"]:
                    instance["entry_point"] = instance["testcases"]["entry_point"]
            else:
                instance["testcases"] = None
        else:
            instance = self.dataset[self.index]


        return instance, finish

    def get_function_signature(self, instance):
        if self.name == "mbpp":
            code = instance["code"]
            lines = code.splitlines()
            if "entry_point" in instance:
                for line in lines:
                    if line.startswith("def") and instance["entry_point"] in line:
                        return line.replace("def", "").replace(":", "").strip()
            else:
                visitor = CodeVisitor(code)
                visitor.run()
                for line in lines:
                    if line.startswith("def") and visitor.funcs[-1] in line:
                        return line.replace("def", "").replace(":", "").strip()
        elif self.name in ["codeparrot/apps"]:
            code = instance["starter_code"]
            lines = code.splitlines()
            for line in lines:
                if line.startswith("def"):
                    return line.replace("def", "").replace(":", "").split("->")[0].strip()
        return None
    
    def get_prompt(self, instance):
        if self.name in ["function", "file"]:
            return instance["final_prompt"]
        prompt = ""
        if self.name == "openai_humaneval":
            prompt += instance["prompt"]
        if self.name in ["codeparrot/apps"]:
            prompt += "You are an expert Python programmer, and here is your task:\n"
            prompt += instance["problem"]
            if len(instance["starter_code"]) > 0:
                prompt += "\nPlease write a Python function {} for the task.\n```python".format(self.get_function_signature(instance))
            else:
                prompt += "\nDo not give explanations, only give the Python code.\nPython Solution:\n```python\n"
        if self.name == "mbpp":
            signature = self.get_function_signature(instance)
            prompt += "You are an expert Python programmer, and here is your task: {} Please write a Python function {} for the task.\n```python".format(instance["prompt"], signature if signature else "")
        if self.name == "deepmind/code_contests":
            prompt += "You are an expert Python programmer, and here is your task:\n"
            prompt += instance["description"]
            prompt += "\nDo not give explanations, only give the Python code.\nPython Solution:\n```python\n"
        
        prompt = prompt.strip()

        return prompt

    def get_chat(self, instance):
        return [{"role": "user", "content": self.get_prompt(instance)}]

    def get_prompt_for_current_instance(self):
        return self.get_prompt(self.dataset[self.index])
    
    def get_all_prompts(self, model = None, context_length = None):
        self.reset_index()
        prompts = []
        finish = False
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            prompts.append(prompt)

        new_prompts = []
        overlong_prompts = []

        if model != None and context_length != None:
            for p in prompts:
                if model.get_prompt_length(p) >= context_length:
                    overlong_prompts.append(p)
                else:
                    new_prompts.append(p)
        else:
            new_prompts += prompts
            
        self.reset_index()
        
        return new_prompts, overlong_prompts


    def get_prompt2instance(self):
        self.reset_index()
        finish = False
        prompt2instance = {}
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            prompt2instance[prompt] = instance

        self.reset_index()

        return prompt2instance
               

    def save_prompt2id(self, file_path = None):
        self.reset_index()
        finish = False
        prompt2id = {}
        while(not finish):
            instance, finish = self.next()
            prompt = self.get_prompt(instance)
            if self.name == "codeparrot/apps":
                prompt2id[prompt] = instance["problem_id"]
            elif self.name == "deepmind/code_contests":
                prompt2id[prompt] = instance["name"]
            elif self.name in ["mbpp", "openai_humaneval"]:
                prompt2id[prompt] = instance["task_id"]
        
        self.reset_index()

        if file_path != None:
            if os.path.exists(file_path):
                new_filepath = "{}_{}.json".format(file_path.replace(".json", ""), time.time())
                print("Warning! {} already exists and renamed to {} to avoid overwriting.".format(file_path, new_filepath))
                os.system("mv {} {}".format(file_path, new_filepath))
            with open(file_path, "w") as f:
                f.write(json.dumps(prompt2id, sort_keys=True, indent=4, separators=(',', ': ')))
        elif self.data_path != None:
            file_path = os.path.join(self.data_path, "prompt2id.json")
            if os.path.exists(file_path):
                new_filepath = "{}_{}.json".format(file_path.replace(".json", ""), time.time())
                print("Warning! {} already exists and renamed to {} to avoid overwriting.".format(file_path, new_filepath))
                os.system("mv {} {}".format(file_path, new_filepath))
            with open(file_path, "w") as f:
                f.write(json.dumps(prompt2id, sort_keys=True, indent=4, separators=(',', ': ')))
        else:
            raise ValueError("Cannot find the file path for prompt2id.")
        

    def load_testcases(self, file_path = None):
        if file_path != None:
            self.prompt2testcase = json.load(open(file_path, "r"))
        elif self.data_path != None:
            self.prompt2testcase = json.load(open(os.path.join(self.data_path, "testcases.json"), "r"))
        else:
            raise ValueError("Cannot find the file path for test cases.")

    def load_stressful_testcases(self, file_path = None):
        if file_path != None:
            self.prompt2stressful = json.load(open(file_path, "r"))
        elif self.data_path != None:
            self.prompt2stressful = json.load(open(os.path.join(self.data_path, "stressful_testcases.json"), "r"))
        else:
            raise ValueError("Cannot find the file path for stressful test cases.")

    def load_groundtruths(self, file_path = None):
        if file_path != None:
            data = json.load(open(file_path, "r"))
            self.prompt2groundtruth = data["prompt2groundtruth"]
            self.prompt2io = data["prompt2io"]
        elif self.data_path != None:
            data = json.load(open(os.path.join(self.data_path, "solutions.json"), "r"))
            self.prompt2groundtruth = data["prompt2groundtruth"]
            self.prompt2io = data["prompt2io"]
        else:
            raise ValueError("Cannot find the file path for ground truth solutions.")
        
    def load_best_groundtruths(self, file_path = None):
        if file_path != None:
            data = json.load(open(file_path, "r"))
            self.prompt2bestgroundtruth = data
        elif self.data_path != None:
            data = json.load(open(os.path.join(self.data_path, "best_solutions.json"), "r"))
            self.prompt2bestgroundtruth = data
        else:
            raise ValueError("Cannot find the file path for best ground truth solutions.")

    def print_info(self):
        self.load_testcases()
        self.load_groundtruths()
        self.load_stressful_testcases()

        prompts, overlong_prompts = self.get_all_prompts()
        empty_solution = 0
        empty_testcase = 0
        empty_stressful = 0
        total_solution = 0
        total_testcase = 0
        total_stressful = 0
        total_num = len(prompts + overlong_prompts)


        for prompt in (prompts + overlong_prompts):
            if prompt not in self.prompt2groundtruth or len(self.prompt2groundtruth[prompt]) == 0:
                empty_solution += 1
            else:
                total_solution += len(self.prompt2groundtruth[prompt])
            if prompt not in self.prompt2testcase or len(self.prompt2testcase[prompt]) == 0 or prompt not in self.prompt2groundtruth or len(self.prompt2groundtruth[prompt]) == 0:
                empty_testcase += 1
            else:
                total_testcase += len(self.prompt2testcase[prompt])
            if prompt not in self.prompt2stressful or len(self.prompt2stressful[prompt]) == 0  or prompt not in self.prompt2groundtruth or len(self.prompt2groundtruth[prompt]) == 0:
                empty_stressful += 1
            else:
                total_stressful += len(self.prompt2stressful[prompt])

        print("Dataset Name: {}\nTotal instance: {}\nInstance with solutions: {}\nInstance with testcases: {}\nInstance with stressful testcases: {}\nAverage solutions: {} ({}/{})\nAverage testcases: {} ({}/{})\nAverage stressful testcases: {} ({}/{})".format(self.name, total_num, total_num - empty_solution, total_num - empty_testcase, total_num - empty_stressful, total_solution / (total_num - empty_solution), total_solution, total_num - empty_solution, total_testcase / (total_num - empty_testcase), total_testcase, total_num - empty_testcase, total_stressful / (total_num - empty_stressful), total_stressful, total_num - empty_stressful))
        print("=" * 20)

    
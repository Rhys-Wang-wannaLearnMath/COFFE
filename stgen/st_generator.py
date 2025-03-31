import ast
import random
from tabnanny import verbose
from typing import List
import time
import os
import json
import gc
from termcolor import colored
from multiprocessing import Array, Value
import copy
from typing import Any, List
from tqdm import tqdm

from stgen.utils import make_request
from coffe.code_execution import untrusted_check
from coffe.sanitize import sanitize



class FuncSTGen:
    def __init__(self, inputs: List, entry_point: str, contract: str, verbose: bool = False):
        self.contract = contract
        self.entry_point = entry_point
        self.seed_pool: List[Any] = copy.deepcopy(inputs)
        self.new_inputs = []
        self.seed_hash = set([hash(str(x)) for x in self.seed_pool])
        self.instruction_messages = [
            "Please generate stressful test cases for given function. Please generate complex, time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
            "Please generate stressful test cases for given function. Please generate time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
            "Please generate stressful test cases for given function. Please generate difficult, time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
        ]
        self.iteration = 20
        self.seed_num = 5
        self.verbose = verbose

    def input_seed_selection(self) -> List:
        return random.sample(self.seed_pool, k=min(len(self.seed_pool), self.seed_num))
    
    def parse_output(self, output):
        output = output[0]
        output = output.replace("```", '')
        output = output.replace("json", '')
        if self.verbose:
            print("Original output:", output)
        output = json.loads(output) 
        return output
    
    def check_correctness(self, results: List) -> bool:
        if len(results) == 0:
            return False, "Empty results"
        for result in results:
            if result["status"] != 1:
                print(result["status_reason"])
                return False, result["input"]
        return True, ' '

    def generate_one(self, selected_inputs: List) -> List:
        message = f"We want to conduct stress test. Here is a function that we want to conduct the stress test:\n```\n{self.contract}\n```"
        
        str_inputs = "\n".join(str(input) for input in selected_inputs)

        message += f"\nThese are some example inputs used to test the function:\n```\n{str_inputs}\n```"
        message += "Learn the format of the example input, but do not learn their style, generate test cases as diverse as possible, as complex as possible. Do not generate any test cases of a scale exceeding 10**4. Generate less than 10 cases per time. Please consider the internal contraints of the function. \n"
        message += f"\n{random.choice(self.instruction_messages)}"
        message += """
Return format should be pure json format which can be handled by json.loads() function in python, for some cases, you can use python expression to replace extreme long list or dict, but be sure to let it be a string variable, instead of a direct list, because we will handle it using json.loads().:
Return: {
"""
        for input in selected_inputs:
            message += "    "
            message += f"\"input{selected_inputs.index(input)+1}\": \"{str(input)}\",\n"
        message +="""
    ...
}          
        """
        message += "\n```json"
        if self.verbose:
            print(message)
            print("")
        ret = make_request(message)
        ret_json = self.parse_output(ret)
        inputs = []
        for key, value in ret_json.items():
            inputs.append(value)
        new_inputs = []
        for i in range(len(inputs)):
            try:
                raw = eval(inputs[i]) # eval the input
                del raw
                new_inputs.append(inputs[i])
            except:
                continue
        return new_inputs

    def generate(self, num: int):
        self.iteration = num * 3
        while len(self.new_inputs) < num and self.iteration >= 0:
            if self.verbose:
                print(colored("Got new inputs:" + str(len(self.new_inputs)) + " needs:" + str(num), "green"))
            seeds = self.input_seed_selection()
            try:
                gc.collect()
                new_inputs = self.generate_one(seeds)
                for new_input in new_inputs:
                    if hash(str(new_input)) not in self.seed_hash:
                        execute_code = self.contract.replace(f"def {self.entry_point}", "def solution")
                        time_limits = [50]
                        testcase = {"input": eval(new_input), "output": None}
                        stat, results = untrusted_check(
                            io=False,
                            code=execute_code,
                            testcases=[testcase],
                            atol=None,
                            ref_time=time_limits,
                            fast_check=False,
                            check=False,
                            generator=False,
                            gt_time_limit_factor=1.0
                        )
                        correctness, reason = self.check_correctness(results)
                        if correctness:
                            if self.verbose:
                                print("new_input passed:", new_input)
                                print(' ')
                            self.seed_pool.append(new_input)
                            self.seed_hash.add(hash(str(new_input)))
                            self.new_inputs.append(new_input)
                        elif self.verbose:
                            print("new_input failed:", new_input)
                            print(f"reason: {reason}")
                self.iteration -= 1
            except Exception as e:
                if self.verbose:
                    print(e)
                self.iteration -= 1
        return self.new_inputs[:num]




class FileSTGen:
    def __init__(self, description:str, io: bool, inputs: List, contract: str, verbose: bool = False):
        self.io = io,
        self.description = description
        self.inputs = inputs
        self.seed_pool: List[Any] = copy.deepcopy(inputs)
        self.new_inputs = []
        self.seed_hash = set([hash(str(x)) for x in self.seed_pool])
        self.contract = contract
        self.instruction_messages = [
            "Please generate stressful test cases for given code. Please generate complex, time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
            "Please generate stressful test cases for given code. Please generate time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
            "Please generate stressful test cases for given code. Please generate difficult, time-consuming inputs to test the function. Please generate difficult inputs to test the function, as diverse as possible, as complex as possible.",
        ]
        self.iteration = 50
        self.seed_num = 10
        self.verbose = verbose

    def input_seed_selection(self) -> List:
        return random.sample(self.seed_pool, k=min(len(self.seed_pool), self.seed_num))
    
    def check_correctness(self, results: List) -> bool:
        if len(results) == 0:
            return False, "Empty results"
        for result in results:
            if result["status"] != 1:
                print(result["status_reason"])
                return False, result["input"]
        return True, ' '
    
    def generate_one(self, selected_inputs: List):
        str_inputs = "\n".join(str(input) for input in selected_inputs)
        message = f"We want to conduct stress test. Here is code that we want to conduct the stress test:\n```\n{self.contract}\n```"
        message += f"\n{random.choice(self.instruction_messages)}\n"
        message += f"Here is the description of the given code:\n```\n{self.description}\n```"
        message = message.replace("You are an expert Python programmer, and here is your task:\n", "").replace("```python", "")
        message += f"\nThese are some example inputs used to test the function:\n```\n{str_inputs}\n```"
        message += "Learn the format of the example input, but do not learn their style, generate test cases as diverse as possible, as complex as possible. Do not generate any test cases of a scale exceeding the contrain described in the problem description. Generate less than 10 cases per time. Please consider the internal contrain of the function. \n"
        message += "Please consider the assertions in the code as the constrains of generated test cases. \n"
        message += f"\n{random.choice(self.instruction_messages)}\n"
        message += """
        **INSTRUCTION**
        please write an input generator function generate_input() for this code (DO NOT generate outputs). No need to include the example usage, just the function.
        The input generator should take no parameters and return one single test input.
        The generated input should meet the input constraints of the problem description.
        The generated input must be stressful test input that can distinguish the time efficiency of different programs.
        Please reply with ONLY the code without any other content. 
        Wrap your input generator with ```.

        You can use the python library random if necessary, 
        here are some examples of how to use the library, which may be helpful:
        random.randint(1, 10)
        random.randrange(1,100,2)
        random.uniform(1.1,5.4)
        random.random()
        """
        if self.verbose:
            print(message)
            print("")
        if len(message) > 450000:
            return False
        ret = make_request(message)
        try:
            ret_code = sanitize(ret[0], "", codegen=False, global_code=True)
        except Exception as e:
            return False
        if self.verbose:
            print("ret_code:", ret_code)
        return ret_code
    
    def generate(self, num: int):
        self.iteration = num * 3
        while len(self.new_inputs) < num and self.iteration >= 0:
            if self.verbose:
                print(colored("Got new inputs:" + str(len(self.new_inputs)) + " needs:" + str(num), "green"))
            seeds = self.input_seed_selection()
            try:
                gc.collect()
                new_inputs = [self.generate_one(seeds)]
                if new_inputs[0] == False:
                    if self.verbose:
                        print("Overlong prompt.")
                    self.iteration -= 2
                    continue
                results = []
                time_limits = [50 for t in new_inputs]
                stat, results = untrusted_check(
                    io=self.io,
                    code=self.contract,
                    testcases=new_inputs,
                    atol=None,
                    ref_time=time_limits,
                    check = False,
                    generator=True,
                    gt_time_limit_factor=1.0
                )

                correctness, reason = self.check_correctness(results)

                if correctness:
                    if self.verbose:
                        print("new_input passed:", new_inputs[0])
                    self.seed_pool.append(new_inputs[0])
                    self.seed_hash.add(hash(new_inputs[0]))
                    self.new_inputs.append(new_inputs[0])
                elif self.verbose:
                    print("new_input failed:", new_inputs[0])
                    print("reason:", reason)
                self.iteration -= 1
            except Exception as e:
                self.iteration -= 1
                if self.verbose:
                    print(colored("Error in executing new inputs", "red"))
                    print(f"Error: {e}")
                self.iteration -= 1
                continue
        return self.new_inputs[:num]






def gen_func_sts(data_file, testcase_file, contract_file, output_file, verbose = False, num = 5):
    data = json.load(open(data_file, "r"))
    testcases = json.load(open(testcase_file, "r"))
    contracts = json.load(open(contract_file, "r"))

    stressful_testcases = {}

    for d in tqdm(data):
        if d["final_prompt"] not in testcases or d["final_prompt"] not in contracts:
            continue
        entry_point = d["entry_point"]
        test_inputs = [testcase["input"] for testcase in testcases[d["final_prompt"]]]
        generator = FuncSTGen(test_inputs, entry_point, contracts[d["final_prompt"]], verbose = verbose)
        st_inputs = generator.generate(num)
        stressful_testcases[d["final_prompt"]] = []
        for st_input in st_inputs:
            stressful_testcases[d["final_prompt"]].append({"input": st_input, "output": None})
    
    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(stressful_testcases, sort_keys=True, indent=4, separators=(',', ': ')))


def gen_file_sts(data_file, testcase_file, solution_file, contract_file, output_file, verbose = False, num = 5):
    data = json.load(open(data_file, "r"))
    testcases = json.load(open(testcase_file, "r"))
    solutions = json.load(open(solution_file, "r"))
    contracts = json.load(open(contract_file, "r"))

    stressful_testcases = {}

    for d in tqdm(data):
        if d["final_prompt"] not in testcases or d["final_prompt"] not in contracts or d["final_prompt"] not in solutions:
            continue
        test_inputs = [testcase["input"] for testcase in testcases[d["final_prompt"]]]
        solution, io = solutions[d["final_prompt"]]
        prompt = d["final_prompt"].replace("You are an expert Python programmer, and here is your task:\n", "").replace("\n```python", "")
        generator = FileSTGen(prompt, io, test_inputs, contracts[d["final_prompt"]], verbose = verbose)
        st_inputs = generator.generate(num)
        stressful_testcases[d["final_prompt"]] = st_inputs
    
    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(stressful_testcases, sort_keys=True, indent=4, separators=(',', ': ')))


if __name__ == "__main__":
    #gen_func_sts("Coffe/datasets/function/data.json", "Coffe/datasets/function/testcases.json", "function_contracts.json", "function_sts.json", verbose = True)
    gen_file_sts("Coffe/datasets/file/data.json", "Coffe/datasets/file/testcases.json", "Coffe/datasets/file/best_solutions.json", "file_contracts.json", "file_sts.json", verbose = True)

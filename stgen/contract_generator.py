import ast
import random
from typing import List, Dict, Any, Tuple, Union
import time
import os
import json
import gc
from termcolor import colored
from tqdm import tqdm


from stgen.utils import make_request
from coffe.code_execution import untrusted_check
from coffe.sanitize import sanitize


class FuncContractGenerator:
    def __init__(self, prompt: str, testcases: list, code: str, entry_point: str, verbose = False):
        self.prompt = prompt
        self.testcases = testcases
        self.code = code
        self.entry_point = entry_point
        self.iteration = 20
        self.target_num = 4
        self.current_num = 0
        self.verbose = verbose
        self.history = [self.code]

        self.instructions = [
            "Generate a new, different contract (assertion) for the function. The assertion should check input validity or enforce a function invariant. Only return the new assert statement, nothing else.",
            "Create a unique contract (assertion) for the function that wasn't used before. Focus on input validation or function preconditions. Only provide the new assert statement.",
        ]

    def get_last_version(self):
        return self.history[-1]

    def update(self, new_code: str):
        self.current_num += 1
        self.history.append(new_code)
        if self.verbose:
            print(colored(f"Update {self.current_num} version", "blue"))

    def insert_contract_into_code(self, contract: str) -> str:
        lines = self.get_last_version().split("\n")
        func_start = next(i for i, line in enumerate(lines) if line.startswith(f"def {self.entry_point}"))
        indent = len(lines[func_start]) - len(lines[func_start].lstrip())
        lines.insert(func_start + 1, " " * (indent + 4) + contract)
        return "\n".join(lines)

    def generate_and_insert_contract(self) -> bool:
        instruction = random.choice(self.instructions)
        prompt = f"{instruction}\n\n"
        prompt += f"Task description:\n```\n{self.prompt}\n```\n"
        prompt += f"Current function code:\n```python\n{self.get_last_version()}\n```\n"
        
        demo_testcases = random.choices(self.testcases, k=min(3, len(self.testcases)))
        prompt += f"Example testcases:\n```python\n{demo_testcases}\n```\n"

        if len(self.history) > 1:
            existing_contracts = [line.strip() for line in self.history[-1].split("\n") if line.strip().startswith("assert")]
            if existing_contracts:
                prompt += f"Existing contracts (do not repeat these):\n```python\n{existing_contracts}\n```\n"

        prompt += "Generate a new, unique contract (assert statement) for this function. Return only the assert statement, nothing else."
        
        if self.verbose:
            print(prompt)
        ret = make_request(prompt)
        if self.verbose:
            print(ret)

        try:
            new_contract = sanitize(ret[0], "", codegen=False, global_code=True).strip()
            if len(new_contract) == 0:
                if self.verbose:
                    print(colored("Contract insertion failed", "red"))
                return False
            if not new_contract.startswith("assert"):
                new_contract = f"assert {new_contract}"
        except Exception as e:
            print(f"Error when sanitizing the contract: {e}")
            return False

        if self.verbose:
            print(colored("Prompt: ", "green"))
            print(prompt)
            print(colored("Generated contract: ", "green"))
            print(new_contract)

        new_code = self.insert_contract_into_code(new_contract)
        execute_code = new_code.replace(f"def {self.entry_point}", "def solution")
        
        time_limits = [100 for _ in self.testcases]
        stat, results = untrusted_check(
            io=False,
            code=execute_code,
            testcases=self.testcases,
            atol=None,
            ref_time=time_limits,
            fast_check=True,
            check=True,
            generator=False,
            gt_time_limit_factor=1.0
        )


        if len(results) > 0 and all(result["status"] == 1 for result in results):
            self.update(new_code)
            if self.verbose:
                print(colored("Contract insertion successful", "green"))
            return True
        else:
            if self.verbose:
                print(colored("Contract insertion failed", "red"))
            return False

    def gen(self):
        while self.iteration > 0 and self.current_num < self.target_num:
            self.iteration -= 1
            self.generate_and_insert_contract()
        
        return self.history[-1]
    
class FileContractGenerator:
    def __init__(self, prompt:str, testcases: list, code: str, io: bool, verbose = False):
        self.prompt = prompt
        self.testcases = testcases
        self.code = code
        self.io = io
        self.iteration = 20
        self.tartget_num = 4
        self.current_num = 0
        self.history = [self.code]
        self.verbose = verbose

        self.instructions = [
                "Please insert the constracts of inputs (using assertion) to the given code based on task description. Please consider the internal logical constraints and the data constrains of the inputs while generating this assertion. Just insert one more assert based on current version. Just reply edited code without any other text.\n",
                "Please insert the constracts of inputs (using assertion) to fullfill the data type and constrains to the given code based on task description. Insert on contracts per time. Just insert one more assert based on current version. Just reply edited code without any other text.\n",
            ]
        
    def get_last_version(self):
        return self.history[-1]

    def update(self, new_code: str):
        self.current_num += 1
        self.history.append(new_code)
        if self.verbose:
            print(colored(f"Update {self.current_num} version", "blue"))

    def get_update_pairs(self):
        if len(self.history) < 2:
            return None
        if len(self.history) >= 2:
            return [(self.history[i], self.history[i+1]) for i in range(len(self.history)-1)]
    
    def check_correctness(self, results: List) -> bool:
        if len(results) == 0:
            return False, "Empty results"
        for result in results:
            if result["status"] != 1:
                return False, result["input"]
        return True, ' '
    
    def contract_update(self):
        instruction = random.choice(self.instructions)
        prompt = f"\n{random.choice(instruction)}\n"

        prompt += f"Here is the task description of the code that we want to insert contracts:\n```\n{self.prompt}\n```"
        prompt += f"Here is the code of that we want to insert contracts:\n```\n{self.get_last_version()}\n```"
        demo_testcases = random.choices(self.testcases, k=random.randint(1, min(len(self.testcases), 10)))
        prompt += f"Here is the example inputs and outputs of the code:\n```\n{demo_testcases}\n```"

        prompt += instruction

        ret = make_request(
            prompt,
        )

        try:
            ret = sanitize(ret[0], "", codegen=False, global_code=True)
        except Exception as e:
            return False

        if self.verbose:
            print(colored("Prompt: ", "green"))
            print(prompt)
            print(colored("Response: ", "green"))
            print(ret)

        results = []

        time_limits = [100 for t in self.testcases]
        stat, results = untrusted_check(
            io=self.io,
            code=ret,
            testcases=self.testcases,
            atol=None,
            ref_time=time_limits,
            fast_check=True,
            check = True,
            generator=False,
            gt_time_limit_factor=1.0
        )
        correctness, input = self.check_correctness(results)
        if correctness:
            self.update(ret)
            if self.verbose:
                print(colored("Execution success", "green"))
            return True
        else:
            if self.verbose:
                print(colored("Execution failed", "red"))
                print(colored(f"Input: {input}", "red"))
            return False
        
    def gen(self):
        while self.iteration > 0 and self.current_num < self.tartget_num:
            self.iteration -= 1
            self.contract_update()
        

        return self.history[-1]


def gen_func_contracts(data_file, testcase_file, solution_file, output_file, verbose = False):
    data = json.load(open(data_file, "r"))
    testcases = json.load(open(testcase_file, "r"))
    solutions = json.load(open(solution_file, "r"))

    contracts = {}

    for d in tqdm(data):
        if d["final_prompt"] not in testcases or d["final_prompt"] not in solutions:
            continue
        entry_point = d["entry_point"]
        solution = solutions[d["final_prompt"]][0].replace(f"def solution", f"def {entry_point}")
        generator = FuncContractGenerator(d["prompt"], testcases[d["final_prompt"]], solution, entry_point, verbose = verbose)
        contracts[d["final_prompt"]] = generator.gen()

    
    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(contracts, sort_keys=True, indent=4, separators=(',', ': ')))


def gen_file_contracts(data_file, testcase_file, solution_file, output_file, verbose = False):
    data = json.load(open(data_file, "r"))
    testcases = json.load(open(testcase_file, "r"))
    solutions = json.load(open(solution_file, "r"))

    contracts = {}

    for d in tqdm(data):
        if d["final_prompt"] not in testcases or d["final_prompt"] not in solutions:
            continue
        solution, io = solutions[d["final_prompt"]]
        prompt = d["final_prompt"].replace("You are an expert Python programmer, and here is your task:\n", "").replace("\n```python", "")
        generator = FileContractGenerator(prompt, testcases[d["final_prompt"]], solution, io, verbose = verbose)
        contracts[d["final_prompt"]] = generator.gen()

    
    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(contracts, sort_keys=True, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    gen_func_contracts("Coffe/datasets/function/data.json", "Coffe/datasets/function/testcases.json", "Coffe/datasets/function/best_solutions.json", "function_contracts.json")
    #gen_file_contracts("Coffe/datasets/file/data.json", "Coffe/datasets/file/testcases.json", "Coffe/datasets/file/best_solutions.json", "file_contracts.json")
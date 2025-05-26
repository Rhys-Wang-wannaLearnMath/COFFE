
import json, os
import ast
import sys
import numpy as np
import itertools
from multiprocessing import Pool, Value
import time
from datetime import datetime
from tqdm import tqdm
import math
from scipy import stats
from termcolor import colored

from coffe.sanitize import sanitize, CodeProcessor
from coffe.dataset import Dataset
from coffe.code_execution import untrusted_check, untrusted_runtime_measure, untrusted_instruction_measure, untrusted_coverage_measure, check_success, is_all_equal, FAILED, untrusted_testcase_check



FAILED = -1
SUCCEED = 1

INF = 9999999999999999


class Extractor(object):
    def __init__(self, dataset, output_file, dataset_repo = "datasets") -> None:
        self.dataset = Dataset(dataset, data_path = os.path.join(dataset_repo, dataset.replace("/", "_")))
        self.output_file = output_file
        self.outputs = json.load(open(self.output_file, "r"))

        self.solutions = {}


    def get_entrypoint(self, instance):
        entry_point = ""
        if self.dataset.name == "openai_humaneval" or (self.dataset.name == "function" and instance["dataset"] == "openai_humaneval"):
            entry_point += instance["entry_point"]

        return entry_point

    def process_solution(self, solution, instance):
        if (self.dataset.name == "openai_humaneval" or (self.dataset.name == "function" and instance["dataset"] == "openai_humaneval")) and not solution.startswith("def"):
            lines = solution.splitlines()
            if len(lines) > 0:
                lines[0] = "    " + lines[0]
            solution = instance["prompt"] + "\n".join(lines)
        elif self.dataset.name == "openai_humaneval" or (self.dataset.name == "function" and instance["dataset"] == "openai_humaneval"):
            try:
                ast.parse(solution)
            except:
                lines = solution.splitlines()
                if len(lines) > 0:
                    lines[0] = "    " + lines[0]
                solution = instance["prompt"] + "\n".join(lines)

        return solution


    def get_solutions(self, codegen = False, chat = False):
        for index, prompt in enumerate(self.outputs):
            if prompt not in self.dataset.prompt2instance:
                #print('Cannot find the prompt of instance #{} in dataset, skipped.'.format(index))
                continue
            instance = self.dataset.prompt2instance[prompt]
            solutions = []
            if not self.outputs[prompt][1]:
                continue
            if isinstance(self.outputs[prompt][0], list):
                for code in self.outputs[prompt][0]:
                    solution = sanitize(code, self.get_entrypoint(instance), codegen = codegen, global_code = True if self.dataset.name in ["codeparrot/apps", "deepmind/code_contests", "file"] else False, chat = chat)
                    solution = self.process_solution(solution, instance)
                    solutions.append(solution)
            elif isinstance(self.outputs[prompt][0], str):
                solution = sanitize(self.outputs[prompt][0], self.get_entrypoint(instance), codegen = codegen, global_code = True if self.dataset.name in ["codeparrot/apps", "deepmind/code_contests", "file"] else False, chat = chat)
                solution = self.process_solution(solution, instance)
                solutions.append(solution)
            
            self.solutions[prompt] = solutions


    def save_solutions(self):
        filename = self.output_file.replace(".json", "_SOLUTIONS.json")
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(json.dumps(self.solutions, sort_keys=True, indent=4, separators=(',', ': ')))



    def process_solutions(self):
        count = 0
        total = 0
        for index, prompt in enumerate(self.solutions):
            print("Processing solutions for instance #{}".format(index), end = "\r", file = sys.stderr)
            new_solutions = []
            for i, solution in enumerate(self.solutions[prompt]):
                total += 1
                processor = CodeProcessor(solution, entry_point = self.dataset.prompt2instance[prompt]["entry_point"] if "entry_point" in self.dataset.prompt2instance[prompt] else None, force_rename = True if self.dataset.name in ["mbpp", "openai_humaneval", "function"] else False)
                res = processor.run()
                if FAILED == res[0]:
                    count += 1
                new_solutions.append(res)
                
            self.solutions[prompt] = new_solutions
        
        print(f"Processing completed for dataset {self.dataset.name}. {count}/{total} ({count/total}) invalid solutions found.")
        return 1 - count/total
    

class TestCaseProcessor(object):
    def __init__(self, output_file):
        self.testcases = json.load(open(output_file, "r"))
    
    def process_testcases(self, generator = False):
        processed_testcases = {}
        count = 0
        total = 0
        for prompt in tqdm(self.testcases, desc = "Checking Test Cases"):
            processed_testcases[prompt] = []
            if not generator:
                if len(self.testcases[prompt]) == 0:
                    continue
                tc = self.testcases[prompt][0]
                tc = tc.replace("\n", "").replace("```json", "").replace("```python", "").replace("```", "").strip()
                items = tc[1:-1].strip().split("},")
                for item in items:
                    total += 1
                    processed_tc = item + "}"
                    processed_tc = processed_tc.strip()
                    try:
                        stat = untrusted_testcase_check(processed_tc, generator=False)
                    except Exception as e:
                        count += 1
                        continue
                    if stat != "pass":
                        count += 1
                        continue
                    processed_testcases[prompt].append(processed_tc)
            else:
                for tc in self.testcases[prompt]:
                    total += 1
                    processed_tc = tc.replace("```json", "").replace("```python", "").replace("```", "").strip()
                    if "import random" in processed_tc:
                        processed_tc = processed_tc.replace("def generate_input():", "random.seed(1024)\ndef generate_input():")
                    try:
                        stat = untrusted_testcase_check(processed_tc, generator=True)
                    except Exception as e:
                        count += 1
                        continue
                    if stat != "pass":
                        count += 1
                        continue
                    processed_testcases[prompt].append(processed_tc)

        print(f"Processing completed. {count}/{total} invalid test cases found.")
        
        return processed_testcases

        


class Evaluator(object):
    def __init__(self, dataset, dataset_repo = "datasets", stressful = False):
        self.dataset = Dataset(dataset, data_path = os.path.join(dataset_repo, dataset.replace("/", "_")))
        self.dataset.load_testcases()
        self.dataset.load_groundtruths()
        self.dataset.load_best_groundtruths()
        if stressful:
            self.dataset.load_stressful_testcases()


    def load_solutions(self, solution_file):
        self.solution_file = solution_file
        self.solutions = json.load(open(self.solution_file, "r"))

    def check_element_type(lst, t):
        for l in lst:
            if not isinstance(l, t):
                if t == float and isinstance(l, int):
                    continue
                return False

        return True

    def transform_element_type(lst, t):
        new_lst = []
        for l in lst:
            new_lst.append(t(l))
        return new_lst

    def get_expected_outputs(self):
        prompts, overlong_prompts = self.dataset.get_all_prompts()

        max_count = 1
        count = 0

        for index, prompt in enumerate(prompts + overlong_prompts):
            #if "Write a function to find the nth newman" not in prompt:
            #    continue
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            gt, io = self.dataset.prompt2groundtruth[prompt][0]
            stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = False)
            
            if stat == "fail":
                self.dataset.prompt2testcase[prompt] = []
                count += 1
                if count > max_count:
                    raise ValueError("Groundtruth solution verification failed!")
                continue
            if len(results) != len(self.dataset.prompt2testcase[prompt]):
                raise ValueError("Num of returned results is inconsistent with original testcase inputs.")
            for i, res in enumerate(results):
                self.dataset.prompt2testcase[prompt][i]["output"] = res["model_output"]
            try:
                json.dumps(self.dataset.prompt2testcase[prompt])
            except:
                self.dataset.prompt2testcase[prompt] = []
        
        self.dataset.save_testcases()
        self.dataset.load_testcases()
        self.verify_groundtruth(remove_instance=True)
        self.dataset.save_testcases()

    def get_expected_outputs_for_stressful_testcases(self):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2stressful[prompt]) == 0:
                continue
            gt, io = self.dataset.prompt2groundtruth[prompt][0]
            stat, results = self.execute_code(gt, io, self.dataset.prompt2stressful[prompt], check = False)
            if stat == "fail":
                self.dataset.prompt2stressful[prompt] = []
                count += 1
                if count > 3:
                    raise ValueError("Groundtruth solution verification failed!")
                continue
            for i, res in enumerate(results):
                self.dataset.prompt2stressful[prompt][i]["output"] = res["model_output"]

    def fix_testcases(self, old_solution_file):
        prompts, overlong_prompts = self.dataset.get_all_prompts()

        data = json.load(open(old_solution_file, "r"))

        old_solutions = data["prompt2groundtruth"]
        prompt2ios = data["prompt2io"]

        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2groundtruth[prompt]) > 0:
                continue
            elif prompt not in self.dataset.prompt2groundtruth:
                continue
            elif prompt not in old_solutions:
                continue

            print(f"Instance # {index}")

            candidates = []
            candidate_gts = []
            for i, gts in enumerate(old_solutions[prompt]):
                print("Verifying instance #{} solution #{}     ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                try:
                    stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = False)
                except Exception as e:
                    print(e)
                    continue
                if check_success(results) and len(results) == len(self.dataset.prompt2testcase[prompt]):
                    candidates.append(results)
                    candidate_gts.append(gts)


            equal_groups = []
            for i, c1 in enumerate(candidates):
                print(i, end = '\r', file = sys.stderr)
                if len(equal_groups) == 0:
                    equal_groups.append([i])
                    continue
                equal = False
                for group in equal_groups:
                    if is_all_equal(candidates[group[0]], c1):
                        equal = True
                        group.append(i)
                        break
                if not equal:
                    equal_groups.append([i])
            scores = [len(group) for group in equal_groups]

            if len(scores) == 0:
                continue
            
            max_score = max(scores)
            if max_score > len(candidates) * 0.8:
                candidate = candidates[equal_groups[scores.index(max_score)][0]]
            else:
                candidate = None
            
            if candidate != None:
                testcases = []
                for i, testcase in enumerate(self.dataset.prompt2testcase[prompt]):
                    testcase["output"] = candidate[i]["model_output"]
                    testcases.append(testcase)
                print(equal_groups)
                print("Fix the incorrect expected outputs in testcase with confidence rate: {}".format(max_score / len(candidates)))
                self.dataset.prompt2testcase[prompt] = testcases
                self.dataset.prompt2groundtruth[prompt] = [candidate_gts[i] for i in equal_groups[scores.index(max_score)]]
                self.dataset.prompt2io[prompt] = prompt2ios[prompt]

        
        self.dataset.save_testcases()
        self.dataset.save_groundtruths()
                

    def verify_groundtruth(self, remove_instance = False, debug = False, start_index = None, verify_num = None, failed_case = False, stressful = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        success = 0
        visited_prompts = []
        if stressful:
            self.get_expected_outputs_for_stressful_testcases()
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            elif stressful and len(self.dataset.prompt2stressful[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if index in [40, 264, 2156, 2200, 3799, 4812] and self.dataset.name == "codeparrot/apps":
                self.dataset.prompt2groundtruth[prompt] = []
                continue
            visited_prompts.append(prompt)
            succeed = False
            new_solutions = []
            for i, gts in enumerate(self.dataset.prompt2groundtruth[prompt]):
                print("Verifying instance #{} solution #{}     ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                stat, results = self.execute_code(gt, io, self.dataset.prompt2testcase[prompt], check = True, fast_check = True)
                
                if stat == "fail":
                    failed_cases = [res for res in results if res["status"] == -1]
                    if debug:
                        pass
                    if remove_instance:
                        self.dataset.prompt2testcase[prompt] = []
                        print("Instance Removed.")
                else:
                    if stressful:
                        stat, results = self.execute_code(gt, io, self.dataset.prompt2stressful[prompt], check = True, fast_check = True)
                        if stat == "fail":
                            failed_cases = [res for res in results if res["status"] == -1]
                            if debug:
                                pass
                            if remove_instance:
                                self.dataset.prompt2stressful[prompt] = []
                                print("Instance Removed.")
                        else:
                            new_solutions.append(gts)
                            succeed = True
                    else:
                        new_solutions.append(gts)
                        succeed = True
            if succeed:
                success += 1
            else:
                if debug:
                    print("-"*20 + f"Instance#{index}" + "-"*20)
            print("Updated solutions for instance #{}, originally {} solutions, now {} solutions.".format(index, len(self.dataset.prompt2groundtruth[prompt]), len(new_solutions)), end = "\r")
            self.dataset.prompt2groundtruth[prompt] = new_solutions
        
        print("Verification completed, Pass: {}, Fail: {}.".format(success, len(self.dataset.prompt2instance) - success))

        if start_index != None and verify_num != None:
            if failed_case:
                return {prompt:self.dataset.prompt2groundtruth[prompt] for prompt in visited_prompts}, failed_cases
            else:
                return {prompt:self.dataset.prompt2groundtruth[prompt] for prompt in visited_prompts}
        else:
            if failed_case:
                return self.dataset.prompt2groundtruth, failed_cases
            else:
                return self.dataset.prompt2groundtruth
            
    def verify_testcases_on_groundtruths(self, testcases, start_index = None, verify_num = None, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        passed_testcases = {}
        total = 0
        fail = 0
        for index, prompt in enumerate(prompts + overlong_prompts):
            if prompt not in self.dataset.prompt2bestgroundtruth or len(self.dataset.prompt2bestgroundtruth[prompt]) == 0:
                continue
            if prompt not in testcases:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            passed_testcases[prompt] = []
            for i, testcase in enumerate(testcases[prompt]):
                print("Verifying instance #{} test case #{}     ".format(index, i), end = '\r', file = sys.stderr)
                succeed = True
                total += 1
                gt, io = self.dataset.prompt2bestgroundtruth[prompt]
                stat, results = self.execute_code(gt, io, [testcase], check = False, fast_check = False, generator = generator)
                if stat == "fail":
                    succeed = False
                if succeed:
                    passed_testcases[prompt].append(testcase)
                else:
                    fail += 1
        
        print("Verification completed, Pass: {}, Fail: {}.".format(total - fail, fail))
        
        return passed_testcases
            


    def execute_code(self, code, io, testcases, check = True, fast_check = True, total_timeout = 10, generator = False):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        stat, testcases = untrusted_check(io, code, testcases, 0, [30 for t in testcases], check = check, fast_check = fast_check, generator=generator)
        return stat, testcases


    def load_temp_data(self):
        passed_solutions = json.load(open(self.solution_file.replace("SOLUTIONS.json", "PASSED_SOLUTIONS.json"), "r"))

        return passed_solutions

    def verify_predictions(self, debug = False, start_index = None, verify_num = None, failed_case = False, stressful = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        passed_solutions = {}
        failed_cases = {}
        if stressful:
            self.get_expected_outputs_for_stressful_testcases()
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            if stressful and len(self.dataset.prompt2stressful[prompt]) == 0:
                continue
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in passed_solutions:
                continue
            if prompt not in self.solutions:
                continue
            if len(self.solutions[prompt]) == 0:
                continue
            
            correct_count = 0
            passed_solutions[prompt] = []
            failed_cases[prompt] = []
            total_timeout = 10
            for i, solution in enumerate(self.solutions[prompt]):
                print("Verifying prediction #{} for instance #{}         ".format(i, index), end = "\r", file = sys.stderr)
                cur_failed_cases = []
                s, io = solution
                if s == -1:
                    continue
                stat, r = self.execute_code(s, io, self.dataset.prompt2testcase[prompt], check = True, fast_check = True, total_timeout = total_timeout)
                if stat == "pass":
                    if stressful:
                        stat, r = self.execute_code(s, io, self.dataset.prompt2stressful[prompt], check = True, fast_check = True, total_timeout = total_timeout)
                        if stat == "pass":
                            correct_count += 1
                            passed_solutions[prompt].append(solution)
                        else:
                            if failed_case:
                                for result in r :
                                    if result["status"] == FAILED:
                                        try:
                                            json.dumps(result)
                                        except Exception as e:
                                            del result["model_output"]
                                            del result["output"]
                                        cur_failed_cases.append(result)
                    else:
                        correct_count += 1
                        passed_solutions[prompt].append(solution)
                else:
                    if failed_case:
                        for result in r:
                            if result["status"] == FAILED:
                                try:
                                    json.dumps(result)
                                except Exception as e:
                                    del result["model_output"]
                                cur_failed_cases.append(result)
                failed_cases[prompt].append(cur_failed_cases)
                            
        if failed_case:
            return passed_solutions, failed_cases
        else:
            return passed_solutions

    def measure_runtime_for_groundtruths(self, subset = None, start_index = None, verify_num = None, std = False, instr = False, stressful = False, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        time_costs = {}
        stds = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                continue
            elif not stressful and len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            elif stressful and len(self.dataset.prompt2stressful[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in time_costs:
                continue
            if subset != None and prompt not in subset:
                continue
            time_costs[prompt] = []
            stds[prompt] = []
            for i, gts in enumerate(self.dataset.prompt2groundtruth[prompt]):
                print("Measuring time/instr count for instance #{} solution #{}          ".format(index, i), end = '\r', file = sys.stderr)
                gt, io = gts
                if stressful:
                    testcases = self.dataset.prompt2stressful[prompt]
                else:
                    testcases = self.dataset.prompt2testcase[prompt]
                try:
                    if not std:
                        if not instr:
                            execution_time_mean = self.execute_code_for_runtime(gt, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean = self.execute_code_for_instr_count(gt, io, testcases, std = std, generator=generator)
                    else:
                        if not instr:
                            execution_time_mean, execution_time_std = self.execute_code_for_runtime(gt, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean, execution_time_std = self.execute_code_for_instr_count(gt, io, testcases, std = std, generator=generator)
                except Exception as e:
                    print(e)
                    continue
                time_cost = sum(execution_time_mean)
                if std:
                    mean_std = np.mean(execution_time_std)
                if time_cost == 0:
                    continue
                time_costs[prompt].append(execution_time_mean)
                if std:
                    stds[prompt].append(execution_time_std)


        if not std:   
            return time_costs
        else:
            return time_costs, stds
        
    def measure_runtime_for_best_groundtruths(self, subset = None, start_index = None, verify_num = None, std = False, instr = False, stressful = False, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        time_costs = {}
        stds = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if len(self.dataset.prompt2bestgroundtruth[prompt]) == 0:
                continue
            elif not stressful and (prompt not in self.dataset.prompt2testcase or len(self.dataset.prompt2testcase[prompt]) == 0):
                continue
            elif stressful and (prompt not in self.dataset.prompt2stressful or len(self.dataset.prompt2stressful[prompt]) == 0):
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in time_costs:
                continue
            if subset != None and prompt not in subset:
                continue
            time_costs[prompt] = []
            stds[prompt] = []
            gt, io = self.dataset.prompt2bestgroundtruth[prompt]
            print("Measuring time/instr count for instance #{}           ".format(index), end = '\r', file = sys.stderr)
            if stressful:
                testcases = self.dataset.prompt2stressful[prompt]
            else:
                testcases = self.dataset.prompt2testcase[prompt]
            try:
                if not std:
                    if not instr:
                        execution_time_mean = self.execute_code_for_runtime(gt, io, testcases, std = std, generator=generator)
                    else:
                        execution_time_mean = self.execute_code_for_instr_count(gt, io, testcases, std = std, generator=generator)
                else:
                    if not instr:
                        execution_time_mean, execution_time_std = self.execute_code_for_runtime(gt, io, testcases, std = std, generator=generator)
                    else:
                        execution_time_mean, execution_time_std = self.execute_code_for_instr_count(gt, io, testcases, std = std, generator=generator)
            except Exception as e:
                print(e)
                continue
            time_cost = sum(execution_time_mean)
            if std:
                mean_std = np.mean(execution_time_std)
            if time_cost == 0:
                continue
            time_costs[prompt].append(execution_time_mean)
            if std:
                stds[prompt].append(execution_time_std)


        if not std:   
            return time_costs
        else:
            return time_costs, stds
        
    def measure_testcase_runtime_on_groundtruths(self, testcases, subset = None, start_index = None, verify_num = None, std = False, instr = False, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        time_costs = {}
        stds = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if prompt not in self.dataset.prompt2bestgroundtruth or len(self.dataset.prompt2bestgroundtruth[prompt]) == 0:
                continue
            if prompt not in testcases or len(testcases[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in time_costs:
                continue
            if subset != None and prompt not in subset:
                continue
            time_costs[prompt] = []
            stds[prompt] = []
            print("Measuring time/instr count for instance #{}           ".format(index), end = '\r', file = sys.stderr)
            gt, io = self.dataset.prompt2bestgroundtruth[prompt]
            tcs = testcases[prompt]
            try:
                if not std:
                    if not instr:
                        execution_time_mean = self.execute_code_for_runtime(gt, io, tcs, std = std, generator=generator)
                    else:
                        execution_time_mean = self.execute_code_for_instr_count(gt, io, tcs, std = std, generator= generator)
                else:
                    if not instr:
                        execution_time_mean, execution_time_std = self.execute_code_for_runtime(gt, io, tcs, std = std, generator=generator)
                    else:
                        execution_time_mean, execution_time_std = self.execute_code_for_instr_count(gt, io, tcs, std = std, generator=generator)
            except Exception as e:
                print(e)
                continue
            time_cost = sum(execution_time_mean)
            if std:
                mean_std = np.mean(execution_time_std)
            if time_cost == 0:
                continue
            time_costs[prompt].append(execution_time_mean)
            if std:
                stds[prompt].append(execution_time_std)


        if not std:   
            return time_costs
        else:
            return time_costs, stds
        
    def measure_testcase_runtime_on_predictions(self, solutions, testcases, subset = None, start_index = None, verify_num = None, std = False, instr = False, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        time_costs = {}
        stds = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if prompt not in solutions or len(solutions[prompt]) == 0:
                continue
            if prompt not in testcases or len(testcases[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in time_costs:
                continue
            if subset != None and prompt not in subset:
                continue
            time_costs[prompt] = []
            stds[prompt] = []
            for i, solution in enumerate(solutions[prompt]):
                print("Measuring time/instr count for instance #{} solution #{}          ".format(index, i), end = '\r', file = sys.stderr)
                s, io = solution
                tcs = testcases[prompt]
                try:
                    if not std:
                        if not instr:
                            execution_time_mean = self.execute_code_for_runtime(s, io, tcs, std = std, generator=generator)
                        else:
                            execution_time_mean = self.execute_code_for_instr_count(s, io, tcs, std = std, generator= generator)
                    else:
                        if not instr:
                            execution_time_mean, execution_time_std = self.execute_code_for_runtime(s, io, tcs, std = std, generator=generator)
                        else:
                            execution_time_mean, execution_time_std = self.execute_code_for_instr_count(s, io, tcs, std = std, generator=generator)
                except Exception as e:
                    print(e)
                    continue
                time_cost = sum(execution_time_mean)
                if std:
                    mean_std = np.mean(execution_time_std)
                if time_cost == 0:
                    continue
                time_costs[prompt].append(execution_time_mean)
                if std:
                    stds[prompt].append(execution_time_std)


        if not std:   
            return time_costs
        else:
            return time_costs, stds

    def measure_coverage_for_testcases(self, testcases = None, subset = None, start_index = None, verify_num = None, stressful = False, generator = False):
        prompts, overlong_prompts = self.dataset.get_all_prompts()
        coverage = {}
        for index, prompt in enumerate(prompts + overlong_prompts):
            if prompt not in self.dataset.prompt2bestgroundtruth or len(self.dataset.prompt2bestgroundtruth[prompt]) == 0:
                continue
            elif testcases == None and not stressful and len(self.dataset.prompt2testcase[prompt]) == 0:
                continue
            elif testcases == None and stressful and len(self.dataset.prompt2stressful[prompt]) == 0:
                continue
            if start_index != None and index < start_index:
                continue
            if start_index != None and verify_num != None and index >= start_index + verify_num:
                break
            if prompt in coverage:
                continue
            if subset != None and prompt not in subset:
                continue
            coverage[prompt] = []
            gt, io = self.dataset.prompt2bestgroundtruth[prompt]
            print("Measuring coverage of test cases for instance #{}          ".format(index), end = '\r', file = sys.stderr)
            if testcases == None:
                if stressful:
                    tcs = self.dataset.prompt2stressful[prompt]
                else:
                    tcs = self.dataset.prompt2testcase[prompt]
            else:
                tcs = testcases[prompt]
            try:
                cov = self.execute_code_for_coverage(gt, io, tcs, generator = generator)
            except Exception as e:
                print(e)
                continue
            coverage[prompt].append(cov)
        
        return coverage

        



    def measure_runtime_for_predictions(self, subset = None, start_index = None, verify_num = None, large_testcase = False, std = False, instr = False, stressful = False, generator = False):
        if large_testcase:
            large_testcases = {}
        if subset == None:
            prompts, overlong_prompts = self.dataset.get_all_prompts()
            time_costs = {}
            stds = {}
            for index, prompt in enumerate(prompts + overlong_prompts):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if prompt not in self.solutions:
                    continue
                if len(self.solutions[prompt]) == 0:
                    continue
                if not stressful and (prompt not in self.dataset.prompt2testcase or len(self.dataset.prompt2testcase[prompt]) == 0):
                    continue
                if stressful and (prompt not in self.dataset.prompt2stressful or len(self.dataset.prompt2stressful[prompt]) == 0):
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                    continue
                if subset != None and prompt not in subset:
                    continue
                time_costs[prompt] = []
                stds[prompt] = []
                if large_testcase:
                    large_testcases[prompt] = []
                for i, solution in enumerate(self.solutions[prompt]):
                    if start_index != None and verify_num != None:
                        print("[Time: {}] Measuring time/instr count for prediction #{} for instance #{}/{}         ".format(datetime.now(), i, index, start_index + verify_num), end = "\r", file = sys.stderr)
                    else:
                        print("[Time: {}] Measuring time/instr count for prediction #{} for instance #{}/{}         ".format(datetime.now(), i, index, len(prompts + overlong_prompts)), end = "\r", file = sys.stderr)
                    s, io = solution
                    if stressful:
                        testcases = self.dataset.prompt2stressful[prompt]
                    else:
                        testcases = self.dataset.prompt2testcase[prompt]
                    if not std:
                        if not instr:
                            execution_time_mean = self.execute_code_for_runtime(s, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean = self.execute_code_for_instr_count(s, io, testcases, std = std, generator=generator)
                    else:
                        if not instr:
                            execution_time_mean, execution_time_std = self.execute_code_for_runtime(s, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean, execution_time_std = self.execute_code_for_instr_count(s, io, testcases, std = std, generator=generator)
                    time_cost = sum(execution_time_mean)
                    if std:
                        mean_std = np.mean(execution_time_std)
                    if time_cost == 0:
                        continue
                    if large_testcase:
                        large_testcases[prompt].append(testcases[execution_time_mean.index(max(execution_time_mean))])
                    time_costs[prompt].append(execution_time_mean)
                    if std:
                        stds[prompt].append(execution_time_std)    
        else:
            time_costs = {}
            stds = {}
            for index, prompt in enumerate(subset):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if prompt not in self.solutions:
                    continue
                if len(self.solutions[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                    continue
                time_costs[prompt] = []
                stds[prompt] = []
                if large_testcase:
                    large_testcases[prompt] = []
                for i, solution in enumerate(self.solutions[prompt]):
                    print("[Time: {}] Measuring time/instr count for prediction #{} for instance #{}/{}         ".format(datetime.now(), i, index, start_index + verify_num), end = "\r", file = sys.stderr)
                    s, io = solution
                    if stressful:
                        testcases = self.dataset.prompt2stressful[prompt]
                    else:
                        testcases = self.dataset.prompt2testcase[prompt]
                    if not std:
                        if not instr:
                            execution_time_mean = self.execute_code_for_runtime(s, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean = self.execute_code_for_instr_count(s, io, testcases, std = std, generator=generator)
                    else:
                        if not instr:
                            execution_time_mean, execution_time_std = self.execute_code_for_runtime(s, io, testcases, std = std, generator=generator)
                        else:
                            execution_time_mean, execution_time_std = self.execute_code_for_instr_count(s, io, testcases, std = std, generator=generator)
                    time_cost = sum(execution_time_mean)
                    if std:
                        mean_std = np.mean(execution_time_std)
                    if time_cost == 0:
                        continue
                    if large_testcase:
                        large_case = testcases[execution_time_mean.index(max(execution_time_mean))]
                        large_case["global"] = io
                        large_testcases[prompt].append(large_case)
                    time_costs[prompt].append(execution_time_mean)
                    if std:
                        stds[prompt].append(execution_time_std)

        if large_testcase:
            if std:
                return time_costs, large_testcases, stds
            else:
                return time_costs, large_testcases
        else:
            if std:
                return time_costs, stds
            else:
                return time_costs


    def measure_all_runtime_for_predictions(self, models, solutions, subset = None, start_index = None, verify_num = None):
        if subset == None:
            prompts, overlong_prompts = self.dataset.get_all_prompts()
            time_costs = {}
            for index, prompt in enumerate(prompts + overlong_prompts):
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                    continue
                if subset != None and prompt not in subset:
                    continue
                time_costs[prompt] = {}
                for model in models:
                    time_costs[prompt][model] = []
                    if prompt not in solutions[model] or len(solutions[model][prompt]) == 0:
                        continue
                    for i, solution in enumerate(solutions[model][prompt]):
                        print("[Time: {}] Verifying prediction #{} for instance #{}         ".format(datetime.now(), i, index), end = "\r", file = sys.stderr)
                        s, io = solution
                        results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                        time_cost = sum(results)
                        if time_cost == 0:
                            continue
                        time_costs[prompt][model].append(time_cost)
        else:
            time_costs = {}
            for index, prompt in enumerate(subset):
                if start_index != None and index < start_index:
                    continue
                if start_index != None and verify_num != None and index >= start_index + verify_num:
                    break
                if len(self.dataset.prompt2testcase[prompt]) == 0:
                    continue
                if len(self.dataset.prompt2groundtruth[prompt]) == 0:
                    continue
                time_costs[prompt] = {}
                for model in models:
                    time_costs[prompt][model] = []
                    if prompt not in solutions[model] or len(solutions[model][prompt]) == 0:
                        continue
                    for i, solution in enumerate(solutions[model][prompt]):
                        print("[Time: {}] Verifying prediction #{} for instance #{}/{} for model {}".format(datetime.now(), i, index, start_index + verify_num, model), end = "\r", file = sys.stderr)
                        s, io = solution
                        results = self.execute_code_for_runtime(s, io, self.dataset.prompt2testcase[prompt])
                        time_cost = sum(results)
                        if time_cost == 0:
                            continue
                        time_costs[prompt][model].append(time_cost)
        return time_costs

    def execute_code_for_runtime(self, code, io, testcases, std = False, generator = False):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        results = untrusted_runtime_measure(io, code, testcases, [5 for t in testcases], std = std, generator = generator)
        return results


    def execute_code_for_instr_count(self, code, io, testcases, std = False, generator = False):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        results = untrusted_instruction_measure(io, code, testcases, [5 for t in testcases], std = std, generator = generator)
        return results

    def execute_code_for_coverage(self, code, io, testcases, generator = False):
        if len(testcases) == 0:
            raise ValueError("No testcase to be executed.")
        results = untrusted_coverage_measure(io, code, testcases, [5 for t in testcases], generator=generator)
        return results


class Metrics(object):
    def __init__(self):
        pass

    # unbiased estimator from https://github.com/openai/human-eval
    def pass_at_k(
        self,
        num_samples,
        num_correct,
        k: int,
    ):
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n: int, c: int, k: int) -> float:
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            assert len(num_samples) == len(num_correct)
            num_samples_it = iter(num_samples)

        return np.array(
            [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
        )

    def correlation(self, x, y):
        if len(x) != len(y):
            raise ValueError("The length of x and y must be the same.")
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        x_var = np.var(x) * len(x)
        y_var = np.var(y) * len(y)
        temp = 0
        for i in range(0, len(x)):
            temp += (x[i] - x_avg) * (y[i] - y_avg)
        
        r = temp / (np.sqrt(x_var) * np.sqrt(y_var))

        return r

    def rsd(self, mean, std):
        if len(mean) != len(std):
            raise ValueError("The length of mean and std must be the same.")

        rsd = 0
        for i in range(0, len(mean)):
            if mean[i] < 0 or std[i] < 0:
                raise ValueError("Negative data found, unable to calculate RSD.")
            rsd += std[i] / mean[i]
        
        rsd = rsd / len(mean)

        return rsd

    def cal(self, metric, prediction_file, data_type = None):
        if metric == "correlation":
            if prediction_file.endswith("TIME.json"):
                time_file = prediction_file
                instr_file = prediction_file.replace("TIME.json", "INSTRUCTION.json")
            elif prediction_file.endswith("time.json") :
                time_file = prediction_file
                instr_file = prediction_file.replace("time.json", "instruction_count.json")
            else:
                raise ValueError("prediction file must end with TIME.json or time.json for correlation calculation!")
            time_data = json.load(open(time_file, "r"))
            instr_data = json.load(open(instr_file, "r"))
            x = []
            y = []
            for prompt in time_data["time"]:
                if prompt not in instr_data["instr_count"]:
                    raise ValueError("Prompt entry does not exist.")
                if len(time_data["time"][prompt]) != len(instr_data["instr_count"][prompt]):
                    raise ValueError("The length of time data and instruction count data is not consistent.")
                for i in range(0, len(time_data["time"][prompt])):
                    if len(time_data["time"][prompt][i]) != len(instr_data["instr_count"][prompt][i]):
                        raise ValueError("The length of time data and instruction count data is not consistent.")
                    for j in range(0, len(time_data["time"][prompt][i])):
                        x.append(time_data["time"][prompt][i][j])
                        y.append(instr_data["instr_count"][prompt][i][j])
            
            return self.correlation(x, y)
        
        elif metric == "pass1":
            if "," not in prediction_file:
                data = json.load(open(prediction_file, "r"))
            else:
                data = {}
                prediction_files = prediction_file.split(",")
                for f in prediction_files:
                    temp = json.load(open(f, "r"))
                    for prompt in temp:
                        data[prompt] = temp[prompt]
            
            num_correct = []
            for prompt in data:
                num_correct.append(len(data[prompt]))

            pass_at_k = np.mean(self.pass_at_k(1, num_correct, 1))

            return pass_at_k
        
        elif metric == "rsd":
            if data_type not in ["time", "instr_count"]:
                raise ValueError("Please indicate the data type: time or instruction count for RSD calculation.")
            data = json.load(open(prediction_file, "r"))

            mean = []
            std = []

            for prompt in data[data_type]:
                if prompt not in data["std"]:
                    raise ValueError("Prompt entry does not exist.")
                if len(data[data_type][prompt]) != len(data["std"][prompt]):
                    raise ValueError("The length of time data and instruction count data is not consistent.")
                for i in range(0, len(data[data_type][prompt])):
                    if len(data[data_type][prompt][i]) != len(data["std"][prompt][i]):
                        raise ValueError("The length of time data and instruction count data is not consistent.")
                    for j in range(0, len(data[data_type][prompt][i])):
                        std.append(data["std"][prompt][i][j])
                        mean.append(data[data_type][prompt][i][j])

            return self.rsd(mean, std)

        elif metric == "line_coverage":
            total_line_coverage = 0
            total_num = 0
            if "," not in prediction_file:
                data = json.load(open(prediction_file, "r"))
            else:
                data = {}
                prediction_files = prediction_file.split(",")
                for f in prediction_files:
                    temp = json.load(open(f, "r"))
                    for prompt in temp:
                        data[prompt] = temp[prompt]
            for prompt in data:
                if len(data[prompt]) == 0:
                    total_line_coverage += 0
                    total_num += 1
                for cov in data[prompt]:
                    total_line_coverage += cov["summary"]["covered_lines"] / (cov["summary"]["covered_lines"] + cov["summary"]["missing_lines"])
                    total_num += 1
            
            return total_line_coverage / total_num
                

        elif metric == "branch_coverage":
            total_branch_coverage = 0
            total_num = 0
            if "," not in prediction_file:
                data = json.load(open(prediction_file, "r"))
            else:
                data = {}
                prediction_files = prediction_file.split(",")
                for f in prediction_files:
                    temp = json.load(open(f, "r"))
                    for prompt in temp:
                        data[prompt] = temp[prompt]
            for prompt in data:
                for cov in data[prompt]:
                    if cov["summary"]["num_branches"] > 0:
                        total_branch_coverage += cov["summary"]["covered_branches"] / cov["summary"]["num_branches"]
                        total_num += 1
                    else:
                        total_branch_coverage += 1.0
                        total_num += 1
            
            return total_branch_coverage / total_num
        

        elif metric == "max":
            if data_type not in ["time", "instr_count"]:
                raise ValueError("Please indicate the data type: time or instruction count.")
            data = json.load(open(prediction_file, "r"))
            max_values = 0
            count = 0
            for prompt in data[data_type]:
                if len(data[data_type][prompt]) > 0:
                    max_values += max(data[data_type][prompt][0])
                    count += 1
            
            return max_values / count
        
        elif metric == "avg":
            if data_type not in ["time", "instr_count"]:
                raise ValueError("Please indicate the data type: time or instruction count.")
            data = json.load(open(prediction_file, "r"))
            mean_values = 0
            count = 0
            for prompt in data[data_type]:
                if len(data[data_type][prompt]) > 0:
                    mean_values += np.mean(data[data_type][prompt][0])
                    count += 1
            
            return mean_values / count
        
        elif metric == "testcase_compilable_rate":
            if "," not in prediction_file:
                data = json.load(open(prediction_file, "r"))
                ori_data = json.load(open(prediction_file.replace("_COMPILABLE.json", ".json"), "r"))
            else:
                data = {}
                ori_data = {}
                prediction_files = prediction_file.split(",")
                for f in prediction_files:
                    temp = json.load(open(f, "r"))
                    ori_temp = json.load(open(f.replace("_COMPILABLE.json", ".json"), "r"))
                    for prompt in temp:
                        data[prompt] = temp[prompt]
                        ori_data[prompt] = ori_temp[prompt]
            
            success = 0
            total = 0

            for prompt in ori_data:
                if len(ori_data[prompt]) == 1:
                    total += 20
                else:
                    total += len(ori_data[prompt])
                if prompt not in data:
                    continue
                success += len(data[prompt])
            
            
            return success / total
        
        elif metric == "accuracy":
            if "," not in prediction_file:
                data = json.load(open(prediction_file, "r"))
                ori_data = json.load(open(prediction_file.replace("_PASSED.json", ".json"), "r"))
            else:
                data = {}
                ori_data = {}
                prediction_files = prediction_file.split(",")
                for f in prediction_files:
                    temp = json.load(open(f, "r"))
                    ori_temp = json.load(open(f.replace("_PASSED.json", ".json"), "r"))
                    for prompt in temp:
                        data[prompt] = temp[prompt]
                        ori_data[prompt] = ori_temp[prompt]
            success = 0
            total = 0

            for prompt in ori_data:
                if len(ori_data[prompt]) == 1:
                    total += 20
                else:
                    total += len(ori_data[prompt])
                if prompt not in data:
                    continue
                success += len(data[prompt])
            
            return success / total
        
        
        elif metric == "rsd_plus":
            prediction_files = prediction_file.split(",")
            if len(prediction_files) not in [2,4]:
                raise ValueError("Please provide two or four files for calculation.")
            data = {}
            passed_solutions = json.load(open(prediction_files[0], "r"))
            data = json.load(open(prediction_files[1], "r"))["instr_count"]
            if len(prediction_files) > 2:
                temp_solutions = json.load(open(prediction_files[2], "r"))
                for prompt in temp_solutions:
                    passed_solutions[prompt] = temp_solutions[prompt]
                temp_cic = json.load(open(prediction_files[3], "r"))["instr_count"]
                for prompt in temp_cic:
                    data[prompt] = temp_cic[prompt]

            means = []
            stds = []
            for prompt in passed_solutions:
                if len(passed_solutions[prompt]) <= 1:
                    continue
                if prompt not in data or len(data[prompt]) <= 1:
                    means.append(1)
                    stds.append(0)
                    continue
                values = [sum(cic) for cic in data[prompt]]
                means.append(np.mean(values))
                stds.append(np.std(values, ddof = 1))
                       
            return self.rsd(means, stds)
            
        elif metric == "speedup":
            if data_type not in ["time", "instr_count"]:
                raise ValueError("Please indicate the data type: time or instruction count.")
            prediction_files = prediction_file.split(",")
            if len(prediction_files) not in [2,4]:
                raise ValueError("Please provide the index file and the cic file.")
            print(colored("Note that speedup only consider correct samples, use efficient_at_1 to include the impacts of correctness.", "yellow"))
            speedups = {}
            indexes = json.load(open(prediction_files[0], "r"))
            cic = json.load(open(prediction_files[1], "r"))[data_type]
            if len(prediction_files) > 2:
                temp_indexes = json.load(open(prediction_files[2], "r"))
                temp_cic = json.load(open(prediction_files[3], "r"))[data_type]
                for prompt in temp_indexes:
                    indexes[prompt] = temp_indexes[prompt]
                    cic[prompt] = temp_cic[prompt]

            for prompt in cic:
                if prompt not in indexes:
                    continue
                if len(cic[prompt]) != len(indexes[prompt]):
                    #print(f"Inconsistent number of instances: {len(cic[prompt])} and {len(indexes[prompt])}")
                    continue
                    #raise ValueError(f"Inconsistent number of instances: {len(cic[prompt])} and {len(indexes[prompt])}")
                gt = sum(cic[prompt][0])
                if gt >= INF:
                    continue
                if len(indexes[prompt]) > 1:
                    for i, baseline in enumerate(indexes[prompt]):
                        if i == 0:
                            continue
                        if sum(cic[prompt][i]) >= INF:
                            continue
                        if baseline not in speedups:
                            speedups[baseline] = [0, 0]
                        speedups[baseline][0] += gt
                        speedups[baseline][1] += sum(cic[prompt][i])
            
            for baseline in speedups:
                speedups[baseline] = speedups[baseline][0] / speedups[baseline][1]
            
            return speedups
        
        elif metric == "efficient_at_1":
            if data_type not in ["time", "instr_count"]:
                raise ValueError("Please indicate the data type: time or instruction count.")
            prediction_files = prediction_file.split(",")
            if len(prediction_files) not in [2,4]:
                raise ValueError("Please provide the index file and the cic file.")
            fast = {}
            indexes = json.load(open(prediction_files[0], "r"))
            cic = json.load(open(prediction_files[1], "r"))[data_type]
            if len(prediction_files) > 2:
                temp_indexes = json.load(open(prediction_files[2], "r"))
                temp_cic = json.load(open(prediction_files[3], "r"))[data_type]
                for prompt in temp_indexes:
                    indexes[prompt] = temp_indexes[prompt]
                    cic[prompt] = temp_cic[prompt]
            
            invalid = 0
            for prompt in cic:
                if prompt not in indexes:
                    continue
                if len(cic[prompt]) != len(indexes[prompt]):
                    #print(f"Inconsistent number of instances: {len(cic[prompt])} and {len(indexes[prompt])}")
                    invalid += 1
                    continue
                    #raise ValueError(f"Inconsistent number of instances: {len(cic[prompt])} and {len(indexes[prompt])}")
                gt = sum(cic[prompt][0])
                if len(indexes[prompt]) > 1:
                    for i, baseline in enumerate(indexes[prompt]):
                        if i == 0:
                            continue
                        if baseline not in fast:
                            fast[baseline] = []
                        if gt > sum(cic[prompt][i]):
                            fast[baseline].append(1)
                        else:
                            fast[baseline].append(0)
            
            eff_k = {}
            
            for baseline in fast:
                if len(fast[baseline]) < len(indexes) - invalid:
                    fast[baseline] += [0 for i in range(0, len(indexes) - len(fast[baseline]) - invalid)]
                eff_k[baseline] = float(np.mean(self.pass_at_k(1, fast[baseline], 1)))
            
            return eff_k



import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.dummy import Pool


from coffe.evaluator import Evaluator, Extractor, Metrics, TestCaseProcessor
from coffe.sandbox import SandBox


def merge_results(args, clean = True):
    # Merge the result files output by different dockers into one single file
    if args.prediction and args.metric not in ["accuracy", "testcase_time", "testcase_instr_count", "coverage"]:
        time_results = {}
        testcase_results = {}
        if args.metric in ["testcase_solution_time", "testcase_solution_instr_count"]:
            testcase_file, solution_file = args.prediction.split(",")
            testcase_name = testcase_file.split("/")[-1].split("_testcases")[0]
        for i in range(args.parallel_num):
            if args.metric == "time":
                if args.stressful:
                    time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME_{i}.json")
                    testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME_LARGE_TESTCASES_{i}.json")
                else:
                    time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME_{i}.json")
                    testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME_LARGE_TESTCASES_{i}.json")
            elif args.metric == "testcase_solution_time":
                time_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_TIME_{i}.json")
            elif args.metric == "instr_count":
                if args.stressful:
                    time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_{i}.json")
                    testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_LARGE_TESTCASES_{i}.json")
                else:
                    time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_{i}.json")
                    testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_LARGE_TESTCASES_{i}.json")
            elif args.metric == "testcase_solution_instr_count":
                time_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_INSTRUCTION_{i}.json")
            time_res = json.load(open(time_file, "r"))
            if args.output_testcase:
                testcase_res = json.load(open(testcase_file, "r"))
            for key in time_res:
                if key not in time_results:
                    time_results[key] = time_res[key]
                else:
                    for prompt in time_res[key]:
                        time_results[key][prompt] = time_res[key][prompt]
            if args.output_testcase:
                for prompt in testcase_res:
                    testcase_results[prompt] = testcase_res[prompt]
            if clean:
                os.remove(time_file)
                if args.output_testcase:
                    os.remove(testcase_file)
        if args.metric == "time":
            if args.stressful:
                time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME.json")
                testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME_LARGE_TESTCASES.json")
            else:
                time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME.json")
                testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME_LARGE_TESTCASES.json")
        elif args.metric == "testcase_solution_time":
            time_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_TIME.json")
        elif args.metric == "instr_count":
            if args.stressful:
                time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION.json")
                testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_LARGE_TESTCASES.json")
            else:
                time_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION.json")
                testcase_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_LARGE_TESTCASES.json")
        elif args.metric == "testcase_solution_instr_count":
            time_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_INSTRUCTION.json")
        with open(time_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(time_results, sort_keys=True, indent=4, separators=(',', ': ')))
        if args.output_testcase:
            with open(testcase_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(testcase_results, sort_keys=True, indent=4, separators=(',', ': ')))
    elif args.prediction and args.metric == "accuracy":
        testcase_results = {}
        for i in range(args.parallel_num):
            testcase_file = args.prediction.replace("_COMPILABLE.json", f"_PASSED_{i}.json")
            testcase_res = json.load(open(testcase_file, "r"))
            for prompt in testcase_res:
                if prompt not in testcase_results:
                    testcase_results[prompt] = testcase_res[prompt]
            if clean:
                os.remove(testcase_file)
        
        testcase_file = args.prediction.replace("_COMPILABLE.json", f"_PASSED.json")
        with open(testcase_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(testcase_results, sort_keys=True, indent=4, separators=(',', ': ')))
    elif args.prediction and args.metric in ["testcase_time", "testcase_instr_count"]:
        testcase_results = {}
        for i in range(args.parallel_num):
            if args.metric == "testcase_time":
                testcase_file = args.prediction.replace("_PASSED.json", f"_TIME_{i}.json")
                output_file = args.prediction.replace("_PASSED.json", f"_TIME.json")
            elif args.metric == "testcase_instr_count":
                testcase_file = args.prediction.replace("_PASSED.json", f"_INSTRUCTION_{i}.json")
                output_file = args.prediction.replace("_PASSED.json", f"_INSTRUCTION.json")
            testcase_res = json.load(open(testcase_file, "r"))
            for key in testcase_res:
                if key not in testcase_results:
                    testcase_results[key] = testcase_res[key]
                else:
                    for prompt in testcase_res[key]:
                        testcase_results[key][prompt] = testcase_res[key][prompt]
            if clean:
                os.remove(testcase_file)
        
        
        with open(output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(testcase_results, sort_keys=True, indent=4, separators=(',', ': ')))
    elif args.prediction and args.metric == "coverage":
        testcase_results = {}
        for i in range(args.parallel_num):
            testcase_file = args.prediction.replace("_SELECTED.json", f"_COVERAGE_{i}.json")
            output_file = args.prediction.replace("_SELECTED.json", f"_COVERAGE.json")
            testcase_res = json.load(open(testcase_file, "r"))
            for prompt in testcase_res:
                if prompt not in testcase_results:
                    testcase_results[prompt] = testcase_res[prompt]
            if clean:
                os.remove(testcase_file)
        with open(output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(testcase_results, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        results = {}
        for i in range(args.parallel_num):
            if args.metric == "time":
                if args.stressful:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_time_{i}.json")
                else:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_time_{i}.json")
            elif args.metric == "instr_count":
                if args.stressful:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_instruction_count_{i}.json")
                else:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_instruction_count_{i}.json")
            elif args.metric == "coverage":
                if args.stressful:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_coverage_{i}.json")
                else:
                    filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_coverage_{i}.json")
            res = json.load(open(filename, "r"))
            for key in res:
                if key not in results:
                    results[key] = res[key]
                else:
                    for prompt in res[key]:
                        results[key][prompt] = res[key][prompt]
            if clean:
                os.remove(filename)
        if args.metric == "time":
            if args.stressful:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_time.json")
            else:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_time.json")
        elif args.metric == "instr_count":
            if args.stressful:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_instruction_count.json")
            else:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_instruction_count.json")
        elif args.metric == "coverage":
            if args.stressful:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_coverage.json")
            else:
                filename = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_coverage.json")
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
            

def evaluate(args, command):
    # Correctness should be evaluated by correctness test cases
    if args.metric == "correctness":
        args.stressful = False
    
    if args.single_worker:
        _evaluate(args)
        return None
    
    if args.final_metric:
        metrics = Metrics()
        res = metrics.cal(args.final_metric, args.prediction, args.metric)
        print(args.final_metric)
        print(res)
        filename = os.path.join(args.output_path, f"{args.final_metric}_results.json")
        with open(filename, "w", encoding = "utf-8") as f:
            f.write(json.dumps(res, sort_keys=True, indent=4, separators=(',', ': ')))
        return None

    if not args.prediction and not os.path.exists(os.path.join(args.output_path, args.dataset)):
        os.mkdir(os.path.join(args.output_path, args.dataset))

    # Set up docker environment
    sandbox = SandBox(args.work_dir, args.perf_path)

    command += " -w"
    # Enable Parallelism
    if args.parallel_num > 1 and args.metric not in ["compilable_rate", "correctness"]:
        evaluator = Evaluator(args.dataset, dataset_repo = args.dataset_path)
        pool = Pool(args.parallel_num)
        data = []
        if args.prediction:
            if "," not in args.prediction:
                command = command.replace(args.prediction, os.path.join("/data", args.prediction), 1)
            else:
                files = args.prediction.split(",")
                command = command.replace(args.prediction, ",".join([os.path.join("/data", f) for f in files]), 1)
        for i in range(args.parallel_num):
            cur_command = command + f" -i {i}"
            cur_command = cur_command.replace(args.dataset_path, os.path.join("/data", args.dataset_path), 1).replace(args.output_path, os.path.join("/data", args.output_path), 1)
            data.append([cur_command, i, 3600*24*4])
        pool.map(sandbox._run, data)
        merge_results(args)
    # Disable Parallelism
    elif args.metric == "compilable_rate":
        _evaluate(args)
    else:
        if args.metric == "correctness":
            command += f" -i 0"
        if args.prediction:
            if "," not in args.prediction:
                command = command.replace(args.prediction, os.path.join("/data", args.prediction), 1)
            else:
                files = args.prediction.split(",")
                command = command.replace(args.prediction, ",".join([os.path.join("/data", f) for f in files]), 1)
        sandbox.run(command.replace(args.dataset_path, os.path.join("/data", args.dataset_path), 1).replace(args.output_path, os.path.join("/data", args.output_path), 1), 0, 3600*24)

    return None



def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    print("NON_JSON_SERIALIZABLE object {} encountered.".format(type(obj)))
    return "<NON_JSON_SERIALIZABLE>" 
        

def _evaluate(args):
    evaluator = Evaluator(args.dataset, dataset_repo = args.dataset_path, stressful = args.stressful)
    ##################################################################
    ################   LLM Prediction Evaluation    ##################
    ##################################################################
    if args.prediction and args.metric not in ["accuracy", "testcase_compilable_rate", "testcase_time", "testcase_instr_count"]:
        print("Path to the prediction file is set, switch to evaluate the predictions.")
        if args.metric not in ["testcase_solution_time", "testcase_solution_instr_count"]:
            evaluator.load_solutions(args.prediction)
        # Enable Parallelism
        # Compilable Rate evaluation does not support parallelism because it is already quite fast
        # Correctness evaluation will be parallized via multi-processing
        # Time and CPU instruction count evaliuation will be parallized via multiple dockers to avoid one influence the other
        if args.parallel_num > 1 and args.index >= 0:
            prompts, overlong_prompts = evaluator.dataset.get_all_prompts()
            total_num = len(prompts) + len(overlong_prompts)
            verify_num = (total_num // args.parallel_num) + 1
            start_index = verify_num * args.index
            if args.metric == "compilable_rate":
                print("Evaluation of compilable rate cannot be parallized, switch to single processing...")
                extractor = Extractor(args.dataset, args.prediction, dataset_repo = args.dataset_path)
                extractor.get_solutions(codegen = True if "codegen" in args.extra_options else False, chat = True if "chat" in args.extra_options else False)
                compilable_rate = extractor.process_solutions()
                extractor.save_solutions()
                print(f"Compilable Rate: {compilable_rate}")
            elif args.metric == "correctness":
                if not args.prediction.endswith("SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with SOLUTIONS.json to evaluate correctness.")
                passed_solutions = {}
                failed_testcases = {}
                if args.output_testcase:
                    futures = []
                    with ProcessPoolExecutor(max_workers=args.parallel_num) as executor:
                        for task_index in range(args.parallel_num):
                            start_index = verify_num * task_index
                            futures.append(
                                executor.submit(
                                    evaluator.verify_predictions,
                                    start_index = start_index,
                                    verify_num = verify_num,
                                    failed_case = True,
                                    stressful = args.stressful
                                )
                            )
                        for future in as_completed(futures):
                            temp_passed_solutions, temp_failed_testcases = future.result()
                            for prompt in temp_passed_solutions:
                                passed_solutions[prompt] = temp_passed_solutions[prompt]
                            for prompt in temp_failed_testcases:
                                failed_testcases[prompt] = temp_failed_testcases[prompt]
                    #passed_solutions, failed_testcases = evaluator.verify_predictions(failed_case = True, start_index = start_index, verify_num = verify_num)
                    testcase_file = args.prediction.replace("SOLUTIONS.json", f"FAILED_TESTCASES.json")
                    with open(testcase_file, "w", encoding = "utf-8") as f:
                        f.write(json.dumps(failed_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    futures = []
                    with ProcessPoolExecutor(max_workers=args.parallel_num) as executor:
                        for task_index in range(args.parallel_num):
                            start_index = verify_num * task_index
                            futures.append(
                                executor.submit(
                                    evaluator.verify_predictions,
                                    start_index = start_index,
                                    verify_num = verify_num,
                                    stressful = args.stressful
                                )
                            )
                        for future in as_completed(futures):
                            temp_passed_solutions= future.result()
                            for prompt in temp_passed_solutions:
                                passed_solutions[prompt] = temp_passed_solutions[prompt]
                    #passed_solutions = evaluator.verify_predictions()
                output_file = args.prediction.replace("SOLUTIONS.json", f"PASSED_SOLUTIONS.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(passed_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "time":
                if not args.prediction.endswith("PASSED_SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with PASSED_SOLUTIONS.json to evaluate execution time.")
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if args.output_testcase:
                    if "std" in args.extra_options:
                        time_costs, large_testcases, time_stds = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = True, std = True, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs, "std": time_stds}
                    else:
                        time_costs, large_testcases = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = True, std = False, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs}
                    if args.stressful:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME_LARGE_TESTCASES_{args.index}.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                    else:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME_LARGE_TESTCASES_{args.index}.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    if "std" in args.extra_options:
                        time_costs, time_stds = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = False, std = True, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs, "std": time_stds}
                    else:
                        time_costs = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = False, std = False, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs}
                if args.stressful:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME_{args.index}.json")
                else:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "instr_count":
                if not args.prediction.endswith("PASSED_SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with PASSED_SOLUTIONS.json to evaluate CPU instruction count.")
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if args.output_testcase:
                    if "std" in args.extra_options:
                        instr_counts, large_testcases, instr_counts_stds = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = True, std = True, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts, "std": instr_counts_stds}
                    else:
                        instr_counts, large_testcases = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = True, std = False, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts}
                    if args.stressful:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_LARGE_TESTCASES_{args.index}.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                    else:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_LARGE_TESTCASES_{args.index}.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    if "std" in args.extra_options:
                        instr_counts, instr_counts_stds = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = False, std = True, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts, "std": instr_counts_stds}
                    else:
                        instr_counts = evaluator.measure_runtime_for_predictions(start_index = start_index, verify_num = verify_num, large_testcase = False, std = False, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts}
                if args.stressful:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_{args.index}.json")
                else:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "coverage":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                data = evaluator.measure_coverage_for_testcases(testcases=testcases, start_index = start_index, verify_num = verify_num, generator=generator)
                output_file = args.prediction.replace("SELECTED.json", f"COVERAGE_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_solution_time":
                if args.prediction.count(",") == 1:
                    testcase_file, solution_file = args.prediction.split(",")
                else:
                    raise ValueError("The value of -p option should be the format of <testcase_file>,<solution_file>")
                testcases = json.load(open(testcase_file, "r"))
                solutions = json.load(open(solution_file, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, start_index=start_index, verify_num=verify_num, std = True, generator=generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, start_index=start_index, verify_num=verify_num, std = False, generator=generator)
                    data = {"time": time_costs}
                testcase_name = testcase_file.split("/")[-1].split("_testcases")[0]
                output_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_TIME_{args.index}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_solution_instr_count":
                if args.prediction.count(",") == 1:
                    testcase_file, solution_file = args.prediction.split(",")
                else:
                    raise ValueError("The value of -p option should be the format of <testcase_file>,<solution_file>")
                testcases = json.load(open(testcase_file, "r"))
                solutions = json.load(open(solution_file, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_counts_stds = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, start_index=start_index, verify_num=verify_num, std = True, instr = True, generator=generator)
                    data = {"instr_count": instr_counts, "std": instr_counts_stds}
                else:
                    instr_counts = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, start_index=start_index, verify_num=verify_num, std = False, instr = True, generator=generator)
                    data = {"instr_count": instr_counts}
                testcase_name = testcase_file.split("/")[-1].split("_testcases")[0]
                output_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_INSTRUCTION_{args.index}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            else:
                raise ValueError(f"Unknown metric: {args.metric}")
        # Disable Parallelism
        else:
            if args.metric == "compilable_rate":
                extractor = Extractor(args.dataset, args.prediction, dataset_repo = args.dataset_path)
                extractor.get_solutions(codegen = True if "codegen" in args.extra_options else False, chat = True if "chat" in args.extra_options else False)
                compilable_rate = extractor.process_solutions()
                extractor.save_solutions()
                print(f"Compilable Rate: {compilable_rate}")
            elif args.metric == "correctness":
                if not args.prediction.endswith("SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with SOLUTIONS.json to evaluate correctness.")
                if args.output_testcase:
                    passed_solutions, failed_testcases = evaluator.verify_predictions(failed_case = True, stressful = args.stressful)
                    testcase_file = args.prediction.replace("SOLUTIONS.json", f"FAILED_TESTCASES.json")
                    with open(testcase_file, "w", encoding = "utf-8") as f:
                        f.write(json.dumps(failed_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    passed_solutions = evaluator.verify_predictions(stressful = args.stressful)
                output_file = args.prediction.replace("SOLUTIONS.json", "PASSED_SOLUTIONS.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(passed_solutions, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "time":
                if not args.prediction.endswith("PASSED_SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with PASSED_SOLUTIONS.json to evaluate execution time.")
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if args.output_testcase:
                    if "std" in args.extra_options:
                        time_costs, large_testcases, time_stds = evaluator.measure_runtime_for_predictions(large_testcase = True, std = True, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs, "std": time_stds}
                    else:
                        time_costs, large_testcases = evaluator.measure_runtime_for_predictions(large_testcase = True, std = False, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs}
                    if args.stressful:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_LARGE_TESTCASES.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                    else:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_LARGE_TESTCASES.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    if "std" in args.extra_options:
                        time_costs, time_stds = evaluator.measure_runtime_for_predictions(large_testcase = False, std = True, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs, "std": time_stds}
                    else:
                        time_costs = evaluator.measure_runtime_for_predictions(large_testcase = False, std = False, stressful = args.stressful, generator=generator)
                        data = {"time": time_costs}
                if args.stressful:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_TIME.json")
                else:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_TIME.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "instr_count":
                if not args.prediction.endswith("PASSED_SOLUTIONS.json"):
                    raise ValueError("The filename of the prediction file must ends with PASSED_SOLUTIONS.json to evaluate CPU instruction count.")
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if args.output_testcase:
                    if "std" in args.extra_options:
                        instr_counts, large_testcases, instr_counts_stds = evaluator.measure_runtime_for_predictions(arge_testcase = True, std = True, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts, "std": instr_counts_stds}
                    else:
                        instr_counts, large_testcases = evaluator.measure_runtime_for_predictions(large_testcase = True, std = False, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts}
                    if args.stressful:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION_LARGE_TESTCASES.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                    else:
                        with open(args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION_LARGE_TESTCASES.json"), "w", encoding = "utf-8") as f:
                            f.write(json.dumps(large_testcases, default=set_default, indent=4, separators=(',', ': ')))
                else:
                    if "std" in args.extra_options:
                        instr_counts, instr_counts_stds = evaluator.measure_runtime_for_predictions(large_testcase = False, std = True, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts, "std": instr_counts_stds}
                    else:
                        instr_counts = evaluator.measure_runtime_for_predictions(large_testcase = False, std = False, instr = True, stressful = args.stressful, generator=generator)
                        data = {"instr_count": instr_counts}
                if args.stressful:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"STRESSFUL_INSTRUCTION.json")
                else:
                    output_file = args.prediction.replace("PASSED_SOLUTIONS.json", f"CORRECTNESS_INSTRUCTION.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "coverage":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                data = evaluator.measure_coverage_for_testcases(testcases=testcases, generator=generator)
                output_file = args.prediction.replace("SELECTED.json", "COVERAGE.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_solution_time":
                if args.prediction.count(",") == 1:
                    testcase_file, solution_file = args.prediction.split(",")
                else:
                    raise ValueError("The value of -p option should be the format of <testcase_file>,<solution_file>")
                testcases = json.load(open(testcase_file, "r"))
                solutions = json.load(open(solution_file, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, std = True, generator=generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, std = False, generator=generator)
                    data = {"time": time_costs}
                testcase_name = testcase_file.split("/")[-1].split("_testcases")[0]
                output_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_TIME.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_solution_instr_count":
                if args.prediction.count(",") == 1:
                    testcase_file, solution_file = args.prediction.split(",")
                else:
                    raise ValueError("The value of -p option should be the format of <testcase_file>,<solution_file>")
                testcases = json.load(open(testcase_file, "r"))
                solutions = json.load(open(solution_file, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_counts_stds = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, std = True, instr = True, generator=generator)
                    data = {"instr_count": instr_counts, "std": instr_counts_stds}
                else:
                    instr_counts = evaluator.measure_testcase_runtime_on_predictions(solutions, testcases, std = False, instr = True, generator=generator)
                    data = {"instr_count": instr_counts}
                testcase_name = testcase_file.split("/")[-1].split("_testcases")[0]
                output_file = solution_file.replace("PASSED_SOLUTIONS.json", f"{testcase_name}_INSTRUCTION.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            else:
                raise ValueError(f"Unknown metric: {args.metric}")
    ##################################################################
    ############   Ground Truth Solution Evaluation    ###############
    ##################################################################
    else:
        print("Path to the prediction file is not set, evaluating the ground truths.")
        # Enable Parallelism
        # Compilable Rate evaluation does not support parallelism because it is already quite fast
        # Correctness evaluation will be parallized via multi-processing
        # Time and CPU instruction count evaliuation will be parallized via multiple dockers to avoid one influence the other
        if args.parallel_num > 0 and args.index >= 0:
            passed_groundtruths = {}
            prompts, overlong_prompts = evaluator.dataset.get_all_prompts()
            total_num = len(prompts) + len(overlong_prompts)
            verify_num = (total_num // args.parallel_num) + 1
            start_index = verify_num * args.index
            if args.metric == "correctness":
                futures = []
                with ProcessPoolExecutor(max_workers=args.parallel_num) as executor:
                    print(f"Using {args.parallel_num} workers to process.")
                    for task_index in range(args.parallel_num):
                        start_index = verify_num * task_index
                        futures.append(
                            executor.submit(
                                evaluator.verify_groundtruth,
                                start_index = start_index,
                                verify_num = verify_num
                            )
                        )
                    for future in as_completed(futures):
                        temp_passed_groundtruths = future.result()
                        for prompt in temp_passed_groundtruths:
                            passed_groundtruths[prompt] = temp_passed_groundtruths[prompt]
                #passed_groundtruths = evaluator.verify_groundtruth(start_index = start_index, verify_num = verify_num)
                output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(passed_groundtruths, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "time":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_runtime_for_best_groundtruths(start_index = start_index, verify_num = verify_num, std = True, stressful = args.stressful, generator=generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_runtime_for_best_groundtruths(start_index = start_index, verify_num = verify_num, std = False, stressful = args.stressful, generator=generator)
                    data = {"time": time_costs}
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_time_{args.index}.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_time_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "instr_count":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_count_stds = evaluator.measure_runtime_for_best_groundtruths(start_index = start_index, verify_num = verify_num, std = True, instr = True, stressful = args.stressful, generator=generator)
                    data = {"instr_count": instr_counts, "std": instr_count_stds}
                else:
                    instr_counts = evaluator.measure_runtime_for_best_groundtruths(start_index = start_index, verify_num = verify_num, std = False, instr = True, stressful = args.stressful, generator=generator)
                    data = {"instr_count": instr_counts}
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_instruction_count_{args.index}.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_instruction_count_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "coverage":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                data = evaluator.measure_coverage_for_testcases(start_index = start_index, verify_num = verify_num, stressful = args.stressful, generator=generator)
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_coverage_{args.index}.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_coverage_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_compilable_rate":
                print("Evaluation of compilable rate cannot be parallized, switch to single processing...")
                processor = TestCaseProcessor(args.prediction)
                if "generator" in args.extra_options:
                    data = processor.process_testcases(generator=True)
                else:
                    data = processor.process_testcases()
                output_file = args.prediction.replace(".json", f"_COMPILABLE.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "accuracy":
                testcases = json.load(open(args.prediction, "r"))
                if "generator" in args.extra_options:
                    data = evaluator.verify_testcases_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num, generator=True) 
                else:
                    data = evaluator.verify_testcases_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num) 
                output_file = args.prediction.replace("_COMPILABLE.json", f"_PASSED_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_time":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_testcase_runtime_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num, std = True, generator = generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_testcase_runtime_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num, std = False, generator = generator)
                    data = {"time": time_costs}
                output_file = args.prediction.replace("_PASSED.json", f"_TIME_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_instr_count":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_count_stds = evaluator.measure_testcase_runtime_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num, instr = True, std = True, generator = generator)
                    data = {"instr_count": instr_counts, "std": instr_count_stds}
                else:
                    instr_counts = evaluator.measure_testcase_runtime_on_groundtruths(testcases, start_index=start_index, verify_num=verify_num, instr = True, std = False, generator = generator)
                    data = {"instr_count": instr_counts}
                output_file = args.prediction.replace("_PASSED.json", f"_INSTRUCTION_{args.index}.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            else:
                raise ValueError(f"Unknown metric: {args.metric}")
        # Disable Parallelism
        else:
            if args.metric == "correctness":
                passed_groundtruths = evaluator.verify_groundtruth()
                output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(passed_groundtruths, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "time":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_runtime_for_best_groundtruths(std = True, stressful = args.stressful, generator=generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_runtime_for_best_groundtruths(std = False, stressful = args.stressful, generator=generator)
                    data = {"time": time_costs}
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_time.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_corrrectness_time.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "instr_count":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_count_stds = evaluator.measure_runtime_for_best_groundtruths(std = True, instr = True, stressful = args.stressful, generator=generator)
                    data = {"instr_count": instr_counts, "std": instr_count_stds}
                else:
                    instr_counts = evaluator.measure_runtime_for_best_groundtruths(std = False, instr = True, stressful = args.stressful, generator=generator)
                    data = {"instr_count": instr_counts}
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_instruction_count.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_instruction_count.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "coverage":
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                data = evaluator.measure_coverage_for_testcases(stressful = args.stressful, generator=generator)
                if args.stressful:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_stressful_coverage.json")
                else:
                    output_file = os.path.join(args.output_path, args.dataset, f"verified_groundtruths_correctness_coverage.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))    
            elif args.metric == "testcase_compilable_rate":
                processor = TestCaseProcessor(args.prediction)
                if "generator" in args.extra_options:
                    data = processor.process_testcases(generator=True)
                else:
                    data = processor.process_testcases()
                output_file = args.prediction.replace(".json", f"_COMPILABLE.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "accuracy":
                testcases = json.load(open(args.prediction, "r"))
                if "generator" in args.extra_options:
                    data = evaluator.verify_testcases_on_groundtruths(testcases, generator = True) 
                else:
                    data = evaluator.verify_testcases_on_groundtruths(testcases) 
                output_file = args.prediction.replace("_COMPILABLE.json", f"_PASSED.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_time":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    time_costs, time_stds = evaluator.measure_testcase_runtime_on_groundtruths(testcases, std = True, generator = generator)
                    data = {"time": time_costs, "std": time_stds}
                else:
                    time_costs = evaluator.measure_testcase_runtime_on_groundtruths(testcases, std = False, generator = generator)
                    data = {"time": time_costs}
                output_file = args.prediction.replace("_PASSED.json", "_TIME.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            elif args.metric == "testcase_instr_count":
                testcases = json.load(open(args.prediction, "r"))
                generator = False
                if "generator" in args.extra_options:
                    generator = True
                if "std" in args.extra_options:
                    instr_counts, instr_count_stds = evaluator.measure_testcase_runtime_on_groundtruths(testcases, instr = True, std = True, generator = generator)
                    data = {"instr_count": instr_counts, "std": instr_count_stds}
                else:
                    instr_counts = evaluator.measure_testcase_runtime_on_groundtruths(testcases, instr = True, std = False, generator = generator)
                    data = {"instr_count": instr_counts}
                output_file = args.prediction.replace("_PASSED.json", "_INSTRUCTION.json")
                with open(output_file, "w", encoding = "utf-8") as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
            else:
                raise ValueError(f"Unknown metric: {args.metric}")
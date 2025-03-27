import argparse
from multiprocessing import Value
import os
from termcolor import colored
import sys
import json

from coffe.evaluate import evaluate
from coffe.config import benchmarks
from coffe.dataset import Dataset


def init(args):
    if os.path.exists(args.dataset):
        dataset_path = args.dataset
    else:
        raise FileNotFoundError(f"Cannot find the path for dataset: {args.dataset}!")
    if os.path.exists(args.workdir):
        workdir = args.workdir
    else:
        raise FileNotFoundError(f"Working directory path does not exist: {args.workdir}!")
    if os.path.exists(args.perf):
        perf_path = args.perf
    else:
        raise FileNotFoundError(f"Working directory path does not exist: {args.workdir}!")
    
    data = {"dataset": dataset_path, "workdir": workdir, "perf_path": args.perf}
    
    with open("coffe_init.json", "w", encoding = "utf-8") as f:
        f.write(json.dumps(data))

    for benchmark in benchmarks:
        dataset = Dataset(benchmark, data_path = os.path.join(dataset_path, benchmarks[benchmark]["path"]))
        dataset.print_info()
    
    print(colored("Coffe initialized!", "green"))

def check_init():
    if os.path.exists("coffe_init.json"):
        data = json.load(open("coffe_init.json", "r"))
        if "dataset" in data and "workdir" in data:
            if data["workdir"] != os.getcwd():
                print("Your current dir is {}, but your working dir of coffe is set to {}.".format(data["workdir"], os.getcwd()))
                print("This may cause potential errors, consider initialize coffe again.")
                exit()
            return data["dataset"], data["workdir"], data["perf_path"]
        else:
            raise ValueError("Coffe configuration corrupted, please initialize coffe again.")
    else:
        raise ValueError("You must initialize Coffe first!")
    
def info(args):
    print("Welcome to use Coffe!")
    print("Coffe is a time efficiency evaluation framework for Python code generation.")
    print("For more details, please see https://github.com/JohnnyPeng18/Coffe.")
    print("Use `coffe init [dataset_path] [workdir_path]` to initialize Coffe first!")
    print("To see the options of Coffe, please use -h option.")

def check_input_file(filename, metric):
    if metric == "correctness" and not filename.endswith("SOLUTIONS.json"):
        raise ValueError("The filename of the prediction file must ends with SOLUTIONS.json to evaluate correctness.")
    if metric in ["time", "instr_count"] and not filename.endswith("PASSED_SOLUTIONS.json"):
        raise ValueError("The filename of the prediction file must ends with PASSED_SOLUTIONS.json to evaluate time efficiency.")

def eval(args):
    if not (hasattr(args, "checked_init") and args.checked_init):
        args.dataset_path, args.work_dir, args.perf_path = check_init()
    if hasattr(args, "command") and args.command:
        command = args.command
    else:
        command = 'coffe eval ' + ' '.join(sys.argv[2:])
    if "-x" in command:
        print(colored("You are running the code on your host machine using -x option, this may cause security issues.", "red"))
    
    if args.parallel_num > 0  and args.host_machine:
        print(colored("You cannot use multiple workers because you are on your host machine with option -w."))
        args.parallel_num = 0

    if args.host_machine:
        args.single_worker = True

    if args.final_metric:
        evaluate(args, command)
        return None
    
    if args.dataset in ["codeparrot/apps", "deepmind/code_contests", "file"] and "generator" not in args.extra_options:
        args.extra_options += "generator"
        command += " -e generator"

    if args.metric == "correctness" and (not args.single_worker or args.host_machine):
        dataset = Dataset(args.dataset, data_path = os.path.join(args.dataset_path, benchmarks[args.dataset]["path"]))
        dataset.load_best_groundtruths()
        results = {}
        indexes = {}
        for prompt in dataset.prompt2bestgroundtruth:
            results[prompt] = [dataset.prompt2bestgroundtruth[prompt]]
            indexes[prompt] = ["gt"]

    if "," in args.prediction and args.metric in ["compilable_rate", "correctness"]:
        print("Handling multiple prediction files...")
        prediction_files = args.prediction.split(",")
        ori_pred = args.prediction
        
        for pred in prediction_files:
            args.prediction = pred
            check_input_file(pred, args.metric)
            sub_command = command.replace(ori_pred, pred)
            evaluate(args, sub_command)
            if args.metric == "correctness" and (not args.single_worker or args.host_machine):
                res = json.load(open(pred.replace("_SOLUTIONS.json", "_PASSED_SOLUTIONS.json"), "r"))
                for prompt in results:
                    results[prompt] += res[prompt]
                    indexes[prompt] += [pred.replace("_SOLUTIONS.json", "")] * len(res[prompt])
        
        if args.metric == "correctness"  and (not args.single_worker or args.host_machine):
            dataset_name = args.dataset.replace("/", "_")
            solution_file = os.path.join(os.path.dirname(pred), f"{dataset_name}_all_PASSED_SOLUTIONS.json")
            index_file = os.path.join(os.path.dirname(pred), f"{dataset_name}_all_indexes.json")
            with open(solution_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
            with open(index_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
            print(f"Correctness evaluation complete, all predictions and groundtruths have been saved to {solution_file} and {index_file}.")
            print("Please use the above files to do performance measurement.")
        
    elif "," in args.prediction:
        raise ValueError("You could only give multiple prediction files when evaluating compilable rate and correctness!")
    else:
        check_input_file(args.prediction, args.metric)
        evaluate(args, command)
        if args.metric == "correctness" and (not args.single_worker or args.host_machine):
            res = json.load(open(args.prediction.replace("_SOLUTIONS.json", "_PASSED_SOLUTIONS.json"), "r"))
            for prompt in results:
                results[prompt] += res[prompt]
                indexes[prompt] += [args.prediction.replace("_SOLUTIONS.json", "")] * len(res[prompt])
            dataset_name = args.dataset.replace("/", "_")
            solution_file = os.path.join(os.path.dirname(args.prediction), f"{dataset_name}_all_PASSED_SOLUTIONS.json")
            index_file = os.path.join(os.path.dirname(args.prediction), f"{dataset_name}_all_indexes.json")
            with open(solution_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
            with open(index_file, "w", encoding = "utf-8") as f:
                f.write(json.dumps(indexes, sort_keys=True, indent=4, separators=(',', ': ')))
            print(f"Correctness evaluation complete, all predictions and groundtruths have been saved to {solution_file} and {index_file}.")
            print("Please use the above files to do performance measurement.")

def pipe(args):
    args.dataset_path, args.work_dir, args.perf_path = check_init()
    args.stressful = True
    args.single_worker = False
    args.index = -1
    args.subset = ""
    args.output_testcase = False

    args.checked_init = True

    if args.final_metric not in ["speedup", "efficient_at_1"]:
        raise ValueError("The final metric could only be speedup or efficient_at_1 in pipeline mode.")

    final_metric = args.final_metric if args.final_metric else ""

    args.final_metric = None

    ori_prediction = args.prediction

    
    print(colored("+++++++++++Step 1: Checking Syntax Errors...", "green"))
    command = 'coffe eval ' + ' '.join(sys.argv[2:]).replace("-f ", "").replace(final_metric, "")
    command += " -m compilable_rate"
    args.metric = "compilable_rate"
    print(f"Executing Command: {command}...")
    args.command = command
    eval(args)
    print(colored("Done!", "green"))

    print(colored("+++++++++++Step 2: Checking Correctness...", "green"))
    command = 'coffe eval ' + ' '.join(sys.argv[2:]).replace(args.metric, "correctness").replace("-f ", "").replace(final_metric, "")
    command = command.replace(ori_prediction, ori_prediction.replace(".json", "_SOLUTIONS.json"))
    command += " -m correctness"
    args.prediction = ori_prediction.replace(".json", "_SOLUTIONS.json")
    args.metric = "correctness"
    print(f"Executing Command: {command}...")
    args.command = command
    eval(args)
    print(colored("Done!", "green"))
    args.stressful = True

    print(colored("+++++++++++Step 3: Measuring GPU Instruction Count...", "green"))
    if "," in ori_prediction:
        dirname = os.path.dirname(ori_prediction.split(",")[-1])
    else:
        dirname = os.path.dirname(ori_prediction)
    dataset_name = args.dataset.replace("/", "_")
    args.prediction = os.path.join(dirname, f"{dataset_name}_all_PASSED_SOLUTIONS.json")
    command = 'coffe eval ' + ' '.join(sys.argv[2:]).replace(args.metric, "instr_count").replace("-f ", "").replace(final_metric, "")
    command = command.replace(ori_prediction, args.prediction)
    command += " -m instr_count"

    if args.dataset in ["codeparrot/apps", "deepmind/code_contests", "file"] and "generator" not in args.extra_options:
        args.extra_options = "generator"
        command += " -e generator"
    
    args.metric = "instr_count"
    print(f"Executing Command: {command}...")
    args.command = command
    eval(args)
    print(colored("Done!", "green"))

    print(colored("Measurement Finished. CPU instruction count results stored into {}".format(args.prediction.replace("_PASSED_SOLUTIONS.json", "_STRESSFUL_INSTRUCTION.json")), "green"))
    
    print(colored("+++++++++++Step 4: Calculating Metrics...", "green"))
    args.final_metric = final_metric
    args.single_worker = False
    command = 'coffe eval ' + ' '.join(sys.argv[2:])
    command = command.replace(ori_prediction, args.prediction.replace("_PASSED_SOLUTIONS.json", "_indexes.json") + "," + args.prediction.replace("_PASSED_SOLUTIONS.json", "_STRESSFUL_INSTRUCTION.json"))
    command += " -m instr_count"
    args.prediction =  args.prediction.replace("_PASSED_SOLUTIONS.json", "_indexes.json") + "," + args.prediction.replace("_PASSED_SOLUTIONS.json", "_STRESSFUL_INSTRUCTION.json")
    args.metric = "instr_count"
    print(f"Executing Command: {command}...")
    args.command = command
    eval(args)
    print(colored("Metrics result written into file: {}".format(os.path.join(args.output_path, f"{args.final_metric}_results.json")), "green"))

    print(colored("Pipeline Finished!", "green"))

def main():
    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='cmd')
    arg_parser.set_defaults(func = info)

    init_parser = sub_parsers.add_parser('init')
    init_parser.add_argument('-d', '--dataset', required = False, default= os.path.join("Coffe", "datasets"), type=str, help = "Path to the COFFE benchmark location")
    init_parser.add_argument('-w', '--workdir', required = False, default=os.getcwd(), type=str, help = "The working directory for dockers and results")
    init_parser.add_argument('-p', '--perf', required = False, default=os.path.join("Coffe", "perf.json"), type=str, help = "Path to the COFFE perf.json config file")
    init_parser.set_defaults(func = init)


    evaluate_parser = sub_parsers.add_parser('eval')
    evaluate_parser.add_argument('dataset', help = "Benchmark name")
    evaluate_parser.add_argument('output_path', help = "Path to the output directory")
    evaluate_parser.add_argument('-p', '--prediction', required = False, type = str, help = "Path to the prediction file")
    evaluate_parser.add_argument('-i', '--index', required = False, default = -1, type = int, help = "The index of workers in parallel processing, should NOT be manually set")
    evaluate_parser.add_argument('-n', '--parallel_num', required = False, default = 0, type = int, help = "The number of workers in parallel processing")
    evaluate_parser.add_argument('-s', '--subset', required = False, default = "", type = str, help = "Path to the file of a subset of indexes")
    evaluate_parser.add_argument('-r', '--stressful', required = False, default = True, action = "store_true", help = "Enable stressful test cases")
    evaluate_parser.add_argument('-t', '--output_testcase', required = False, default = False, action = "store_true", help = "Output the most expensive test case for time/instr_count measurement or the first failed case for correctness check")
    evaluate_parser.add_argument('-m', '--metric', required = False, default = "correctness", type = str, help = "The metric to be evaluated, can be compilable_rate, correctness, time, or instr_count for code solutions and testcase_compilable_rate, accuracy, testcase_time, testcase_instr_count, coverage for test cases, and testcase_solution_time, testcase_solution_instr_count for test cases on predictions")
    evaluate_parser.add_argument('-e', '--extra_options', required = False, default = "", type = str, help = "Extra options for the evaluation")
    evaluate_parser.add_argument('-w', '--single_worker', required = False, default = False, action = "store_true", help = "Running as a single internal worker instead of calling docker containers. DANGEROUS!")
    evaluate_parser.add_argument('-x', '--host_machine', required = False, default = False, action = "store_true", help = "Running code on host machine instead of calling docker containers. DANGEROUS!")
    evaluate_parser.add_argument('-f', '--final_metric', required = False, type = str, help = "The final metric calculated based on the measurement results, can be correlation, rsd, pass_k, line_coverage, branch_coverage, max, avg, testcase_compilable_rate, accuracy, efficient_at_1")
    evaluate_parser.set_defaults(func = eval)

    pipeline_parser = sub_parsers.add_parser('pipe')
    pipeline_parser.add_argument('dataset', help = "Benchmark name")
    pipeline_parser.add_argument('output_path', help = "Path to the output directory")
    pipeline_parser.add_argument('-p', '--prediction', required = False, type = str, help = "Path to the prediction file")
    pipeline_parser.add_argument('-n', '--parallel_num', required = False, default = 0, type = int, help = "The number of workers in parallel processing")
    pipeline_parser.add_argument('-e', '--extra_options', required = False, default = "", type = str, help = "Extra options for the evaluation")
    pipeline_parser.add_argument('-x', '--host_machine', required = False, default = False, action = "store_true", help = "Running code on host machine instead of calling docker containers. DANGEROUS!")
    pipeline_parser.add_argument('-f', '--final_metric', required = False, type = str, help = "The final metric calculated based on the measurement results, can be speedup or efficient_at_1.")
    pipeline_parser.set_defaults(func = pipe)


    args = arg_parser.parse_args()
    args.func(args)
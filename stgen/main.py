from genericpath import isdir
import json
import os
import argparse
from termcolor import colored


from stgen.st_generator import gen_file_sts, gen_func_sts
from stgen.contract_generator import gen_file_contracts, gen_func_contracts



def info(args):
    print("Welcome to use STGen!")
    print("STGen is part of the Coffe benchmark and enables stressful test case generation.")
    print("For more details, please see https://github.com/JohnnyPeng18/Coffe/stgen.")
    print("To see the options of STGen, please use -h option.")

def check_environ():
    key = os.environ.get('API_KEY')
    base_url = os.environ.get('BASE_URL')
    if key == None or base_url == None:
        raise ValueError("Please set environment variable $API_KEY and $BASE_URL before making requests to remote LLM services.")


def contract(args):
    check_environ()
    if args.level == "func":
        if os.path.isdir(args.output_path):
            output_file = os.path.join(args.output_path, "func_contracts.json")
        else:
            output_file = args.output_path
        gen_func_contracts(args.data_file, args.test_file, args.solution_file, output_file, verbose = args.verbose)
    elif args.level == "file":
        if os.path.isdir(args.output_path):
            output_file = os.path.join(args.output_path, "file_contracts.json")
        else:
            output_file = args.output_path
        gen_file_contracts(args.data_file, args.test_file, args.solution_file, output_file, verbose = args.verbose)
    else:
        raise ValueError(f"Unrecognized option: {args.level}!")
    print(f"Generated contracts stored into {output_file}")

def st(args):
    check_environ()
    if args.level == "func":
        if os.path.isdir(args.output_path):
            output_file = os.path.join(args.output_path, "func_stressful_testcases.json")
        else:
            output_file = args.output_path
        gen_func_sts(args.data_file, args.test_file, args.contract_file, output_file, verbose = args.verbose, num = args.num)
    elif args.level == "file":
        if not args.solution_file:
            raise ValueError("Path to solution file must be indicated with option -s when generating file-level stressful test cases!")
        if os.path.isdir(args.output_path):
            output_file = os.path.join(args.output_path, "file_stressful_testcases.json")
        else:
            output_file = args.output_path
        gen_file_sts(args.data_file, args.test_file, args.solution_file, args.contract_file, output_file, verbose = args.verbose, num = args.num)
    else:
        raise ValueError(f"Unrecognized option: {args.level}!")
    print(f"Generated Stressful Test Cases stored into {output_file}")


def pipe(args):
    check_environ()
    if args.level == "func":
        if os.path.isdir(args.output_path):
            contract_file = os.path.join(args.output_path, "func_contracts.json")
            st_file = os.path.join(args.output_path, "func_stressful_testcases.json")
        else:
            contract_file = os.path.join(os.path.dirname(args.output_path), "func_contracts.json")
            st_file = args.output_path
        print(colored("+++++++++++Step 1: Generating Contracts...", "green"))
        gen_func_contracts(args.data_file, args.test_file, args.solution_file, contract_file)
        print(f"Generated contracts stored into {contract_file}")
        print(colored("Done!", "green"))
        print(colored("+++++++++++Step 2: Generating Stressful Test Cases...", "green"))
        gen_func_sts(args.data_file, args.test_file, contract_file, st_file, num = args.num)
        print(f"Generated Stressful Test Cases stored into {st_file}")
        print(colored("Done!", "green"))
        print(colored("Pipeline Finished!", "green"))
    elif args.level == "file":
        if not args.solution_file:
            raise ValueError("Path to solution file must be indicated with option -s when generating file-level stressful test cases!")
        if os.path.isdir(args.output_path):
            contract_file = os.path.join(args.output_path, "file_contracts.json")
            st_file = os.path.join(args.output_path, "file_stressful_testcases.json")
        else:
            contract_file = os.path.join(os.path.dirname(args.output_path), "file_contracts.json")
            st_file = args.output_path
        print(colored("+++++++++++Step 1: Generating Contracts..", "green"))
        gen_file_contracts(args.data_file, args.test_file, args.solution_file, contract_file)
        print(f"Generated contracts stored into {contract_file}")
        print(colored("Done!", "green"))
        print(colored("+++++++++++Step 2: Generating Stressful Test Cases...", "green"))
        gen_file_sts(args.data_file, args.test_file, args.solution_file, contract_file, st_file, num = args.num)
        print(f"Generated Stressful Test Cases stored into {st_file}")
        print(colored("Done!", "green"))
        print(colored("Pipeline Finished!", "green"))
    else:
        raise ValueError(f"Unrecognized option: {args.level}!")



def main():
    arg_parser = argparse.ArgumentParser()
    sub_parsers = arg_parser.add_subparsers(dest='cmd')
    arg_parser.set_defaults(func = info)

    contract_parser = sub_parsers.add_parser('contract')
    contract_parser.add_argument('-o', '--output_path', required=True, type=str, help='Path to the output contract file')
    contract_parser.add_argument('-l', '--level', required = False, default="func", type=str, help="Should be func for function-level or file for file-level stressful test cases.")
    contract_parser.add_argument('-d', '--data_file', required=True, type=str, help='Path to the data file')
    contract_parser.add_argument('-t', '--test_file', required=True, type=str, help='Path to the correctness test case file')
    contract_parser.add_argument('-s', '--solution_file', required=True, type=str, help='Path to the ground truth solution file')
    contract_parser.add_argument('-v', '--verbose', required=False, default=False, action = "store_true", help='Display debug information')
    contract_parser.set_defaults(func = contract)

    st_parser = sub_parsers.add_parser('st')
    st_parser.add_argument('-o', '--output_path', required=True, type=str, help='Path to the output contract file')
    st_parser.add_argument('-l', '--level', required = False, default="func", type=str, help="Should be func for function-level or file for file-level stressful test cases.")
    st_parser.add_argument('-d', '--data_file', required=True, type=str, help='Path to the data file')
    st_parser.add_argument('-t', '--test_file', required=True, type=str, help='Path to the correctness test case file')
    st_parser.add_argument('-s', '--solution_file', required=False, type=str, help='Path to the ground truth solution file')
    st_parser.add_argument('-c', '--contract_file', required=True, type=str, help='Path to the contract file')
    st_parser.add_argument('-n', '--num', required=False, type=int, default=5, help='Number of stressful test cases to generate for each instance')
    st_parser.add_argument('-v', '--verbose', required=False, default=False, action = "store_true", help='Display debug information')
    st_parser.set_defaults(func = st)

    pipe_parser = sub_parsers.add_parser('pipe')
    pipe_parser.add_argument('-o', '--output_path', required=True, type=str, help='Path to the output contract file')
    pipe_parser.add_argument('-l', '--level', required = False, default="func", type=str, help="Should be func for function-level or file for file-level stressful test cases.")
    pipe_parser.add_argument('-d', '--data_file', required=True, type=str, help='Path to the data file')
    pipe_parser.add_argument('-t', '--test_file', required=True, type=str, help='Path to the correctness test case file')
    pipe_parser.add_argument('-s', '--solution_file', required=False, type=str, help='Path to the ground truth solution file')
    pipe_parser.add_argument('-n', '--num', required=False, type=int, default=5, help='Number of stressful test cases to generate for each instance')
    pipe_parser.set_defaults(func = pipe)

    args = arg_parser.parse_args()
    args.func(args)

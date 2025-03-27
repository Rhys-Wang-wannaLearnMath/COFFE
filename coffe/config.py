
benchmarks = {
    "openai_humaneval": {
        "code_keyword": "canonical_solution",
        "testcase_keyword": "testcases",
        "add_list": False,
        "path": "openai_humaneval"
    },
    "mbpp": {
        "code_keyword": "code",
        "testcase_keyword": "testcases",
        "add_list": False,
        "path": "mbpp"
    },
    "codeparrot/apps": {
        "code_keyword": "solutions",
        "testcase_keyword": "input_output",
        "add_list": True,
        "path": "codeparrot_apps"
    },
    "deepmind/code_contests": {
        "code_keyword": "solutions",
        "testcase_keyword": ["private_tests", "generated_tests"],
        "add_list": True,
        "path": "deepmind_code_contests"
    },
    "function": {
        "path": "function"
    },
    "file": {
        "path": "file"
    }
}


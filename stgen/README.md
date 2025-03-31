# STGen
STGen is a framework that generates stressful test cases for code with correctness test cases. It is used in the COFFE benchmark to add stressful test cases.

## Installation
The installation of COFFE benchmark automatically installs STGen, you do not need to perform extra operations to install STGen again.


## Usage
### Pipeline
STGen provides a pipeline to directly generate stressful test cases. You can use the following command:
```bash
stgen pipe -l func -d Coffe/datasets/function/data.json -t Coffe/datasets/function/testcases.json -s Coffe/datasets/function/best_solutions.json -n 5 -o <output_path>
```

This command generates five stressful test cases for function-level code given three data files in the Coffe benchmark.
If you want to generate stressful test cases for other code, you should prepare the three data files follow the formats of the above three files. For the `data.json` file, each instance should contain the `entry_point`, `prompt` and `final_prompt` fields for function-level instances and `prompt` and `final_prompt` fields for file-level instances.

### Single Phase
The generation process in STGen contains two phases: 1) contract generation and 2) test case generation. We also provide commands to conduct the single phases.

**Phase I - Contract Generation:**

You can use the following command to only generate contracts for code:
```shell
stgen contract -l func -d Coffe/datasets/function/data.json -t Coffe/datasets/function/testcases.json -s Coffe/datasets/function/best_solutions.json -o <output_path>
```
This command generates contracts for function-level code given the data files in the COFFE benchmark. Each contract is the original code containing extra `assert` statements, which could be used to guide the generation of stressful test cases.

**Phase II - Test Case Generation:**

You can use the following command to only generate stressful test cases given the previously generated contracts:
```shell
stgen st -l func -d Coffe/datasets/function/data.json -t Coffe/datasets/function/testcases.json -s Coffe/datasets/function/best_solutions.json -c <contract_file_path> -o <output_path> -n 5
```
This command generates five stressful test cases for function-level code given the data files and previously generated contract file `<contract_file_path>`. 
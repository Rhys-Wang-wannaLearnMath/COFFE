# COFFE

COFFE is a Python benchmark for evaluating the time efficiency of LLM-generated code. It is released by the FSE'25 paper "[COFFE: A Code Efficiency Benchmark for Code Generation](https://arxiv.org/abs/2502.02827)". You can also refer to the [project webpage](https://www.yunpeng.site/projects/coffe/) for more details.

## Data

COFFE is designed for evaluating both function-level code and file-level code. It contains selected instances from HumanEval, MBPP, APPS and Code Contests. COFFE keeps the original test cases in these benchmarks as *correctness test cases* and adds new test cases designed for time efficiency evaluation as *stressful test cases*.

**Statistics:**

|Category|#Instance|#Solution/Instance|#Correctness/Instance | #Stressful/Instance|
|----|----|----|----|----|
|Function-Level|398|1.00|5.72|4.99|
|File-Level|358|66.93|43.68|4.95|

**Data Files:**

All instances in COFFE are in `Coffe/datasets`, where `Coffe/datasets/function` contains all function-level instances and `Coffe/datasets/file` contains all file-level instances. In each repo:
- `best_solutions.json` contains the best ground truth solution COFFE uses to calculate efficient@1 and speedup.
- `stressful_testcases.json` contains all stressful test cases COFFE adds.
- `solutions.json` contains all ground truth solutions in original benchmarks.
- `testcases.json` contains all correctness test cases in original benchmarks.

## Installation

**Requirements:**
- Linux Machine
- Docker 
- Python>=3.10

We suggest you create a virtural environment before installing COFFE.

1. To use COFFE, please clone this repo in your current workspace `workspace/` and execute:
```bash
cd Coffe && pip install .
```
2. COFFE comes with a docker image, to install it:
```bash
docker build . -t coffe
```
Note that if your network requires proxy, please modify the Dockerfile in `Coffe/` to indicate it, otherwise the docker image building process could fail.

3. Go back to your workspace and initialize COFFE:
```bash
cd .. && coffe
```
If your installation succeeds, you could see the statistics of COFFE. 

If your installation fails with the reason `Your OS does not support measuring CPU instruction counts`, this may be because of a permission error. In this case, check the default permission level in your system:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

If the output is greater than `2`, then you do not have permission to measure CPU instruction counts by default.

To enable this measurement, try to set the `/proc/sys/kernel/perf_event_paranoid` to `2`, `1`, `0,` or `-1`. The smaller it is, the larger the permission you have.

```
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

This will temporarily allow you to access the CPU instruction count and will expire after you restart the system.

If you want to permanently allow the measurement (This may induce security issues!):

Edit the `/etc/sysctl.conf` by adding the following line:

```
kernel.perf_event_paranoid= -1
```

## Usage

### Pipeline

When you prepare the predictions from LLMs, COFFE provides a pipeline to calculate the efficient@1 and speedup defined in the paper:
```bash
coffe pipe <benchmark_name> <output_repo> 
-p <pred_file_1>,...,<pred_file_n> 
-f <efficient_at_1/speedup> 
-n <number_of_workers>
```
This command has four phases:
1. santize the predictions.
2. select the correct predictions based on correctness test cases.
3. evaluate the GPU instruction count based on stressful test cases.
4. calculate the final metrics.

For example:
```bash
coffe pipe function Coffe/examples/function -p Coffe/examples/function/GPT-4o.json -f efficient_at_1 -n 8
```
This command evaluates the predictions from GPT-4o on the function-level instances of COFFE. If you want to evaluate other LLMs, please prepare a `JSON` file with the same format as `Coffe/examples/function/GPT-4o.json`. 

**Prediction File Format:**

In the `JSON` file, the key is the prompt used to query the LLM for the results, you could get the prompts in `datasets/function/prompts.json` and `datasets/file/prompts.json`. The value contains two objects, the first is a list contains the raw outputs from LLMs and the second is an indicator for the whether the raw output is valid.

**Note:**

In default, COFFE will run all predictions in docker. However, if you could not successfully install the docker or want to run the predictions on the host machine, you can add the `-x` option.


### Single Evaluation

The `pipe` command provides an entire pipeline for calculating the final metrics. This pipeline could also be completed by executing the following four single evaluation commands.

1. Sanitize the predictions
```bash
coffe eval <benchmark_name> <output_repo> 
-p <pred_file_1>,...,<pred_file_n> 
-m compilable_rate
```
This commands output a file ending with `SOLUTIONS.json` that contains the predictions without syntax errors.

2. Select correct predictions
```bash
coffe eval <benchmark_name> <output_repo> 
-p <pred_file_1>,...,<pred_file_n> 
-m correctness
-n <number_of_workers>
```
This commands accept prediction files ending with `SOLUTIONS.json` and output a file ending with `PASSED_SOLUTIONS.json` that contains the predictions pass all correctness solutions.

**Note:**
This command will combine all correct solutions and ground truth solutions together into files `<dataset_name>_all_indexes.json` (used in step 4) and `<dataset_name>_all_PASSED_SOLUTIONS.json` for the next step.

3. Evaluate the GPU instruction count
```bash
coffe eval <benchmark_name> <output_repo> 
-p <pred_file>
-m instr_count
-n <number_of_workers>
```
This command will evaluate the GPU instruction count each prediction consumes and output a file ending with `STRESSFUL_INSTRUCTION.json`. 

**Note:**
This command could only accept one single prediction file ending with `PASSED_SOLUTIONS.json`.

4. Calculating the efficient@1/speedup
```bash
coffe eval <benchmark_name> <output_repo> 
-p <index_file>,<pred_file>
-m instr_count
-f <efficient_at_1/speedup>
```
This command calculate the efficient@1 or speedup.

**Note:**
This command requires the index file and the instruction file as COFFE compares the performance of predictions with grouth truth solutions to calculate the metrics.

## STGen

For details about the stressful test case generation approach STGen, please see `stgen/`.


## Cite
If you use COFFE, please cite us:
```
@misc{peng2025coffe,
      title={COFFE: A Code Efficiency Benchmark for Code Generation}, 
      author={Yun Peng and Jun Wan and Yichen Li and Xiaoxue Ren},
      year={2025},
      eprint={2502.02827},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2502.02827}, 
}
```

import openai
import os
import re
import string
from tqdm import tqdm

import copy
from typing import Any, List
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


def make_request(
    query,
    model="gpt-4o",
    n=1,
    temperature=0.7,
    max_tokens=4096,
    initial_instruction="You are a helpful assistant about software development.",
):
    messages = [
        {"role": "system", "content": initial_instruction},
        {"role": "user", "content": query},
    ]
    # make it less than 4096 tokens
    retry_times = 0
    key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    if key == None or base_url == None:
        raise ValueError(
            "Please set environment variable $API_KEY and $BASE_URL before making requests to remote LLM services."
        )
    client = openai.OpenAI(api_key=key, base_url=base_url)

    while retry_times < 3:
        try:
            answers = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [x.message.content for x in answers.choices]
        except Exception as e:
            print(e)
            retry_times += 1

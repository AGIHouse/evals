"""
This file defines various helper functions for interacting with the OpenAI API.
"""
import logging

import backoff
import openai
import time
import random
import requests
from openai import util


def generate_dummy_chat_completion():
    return {
        "id": "dummy-id",
        "object": "chat.completion",
        "created": 12345,
        "model": "dummy-chat",
        "usage": {"prompt_tokens": 56, "completion_tokens": 6, "total_tokens": 62},
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a dummy response."},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def generate_dummy_completion():
    return {
        "id": "dummy-id",
        "object": "text_completion",
        "created": 12345,
        "model": "dummy-completion",
        "choices": [
            {
                "text": "This is a dummy response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    if kwargs["model"] == "dummy-completion":
        return generate_dummy_completion()

    result = openai.Completion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    return result


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.Timeout,
    ),
    max_value=60,
    factor=1.5,
)
def openai_chat_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a chat completion.
    `args` and `kwargs` match what is accepted by `openai.ChatCompletion.create`.
    """
    if kwargs["model"] == "dummy-chat":
        return generate_dummy_chat_completion()

    result = openai.ChatCompletion.create(*args, **kwargs)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])
    print(result)
    return result


def agi_answer(endpoint, instruction, question, temperature=0.1, top_p=0.75, top_k=40, beams=4, max_tokens=128, max_retries=10):
    for i in range(max_retries):
        response = requests.post(f'{endpoint}/run/predict', json={
        	'data': [
        		instruction,
        		question,
        		temperature,
        		top_p,
        		top_k,
        		beams,
        		max_tokens,
        	]
        })
        
        if response.status_code == 200:            
            res = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": response.json()['data'][0],
                            "role": "assistant"
                        }
                    }
                ],
                "created": int(time.time()*1000),
                "id": f"agi-{random.getrandbits(128)}",
                "model": "agi",
                "object": "chat.completion",
                "usage": {
                    "completion_tokens": 42,
                    "prompt_tokens": 42,
                    "total_tokens": 42
                }
            }

            return util.convert_to_openai_object(
                res,
                api_key='dummy-key',
                api_version=None,
                organization='davinci-3',
                engine=None,
                plain_old_data=False,
            )
        
        time.sleep(1)
    
    raise Exception(f"Max retries ({max_retries}) exceeded for {endpoint}")

def agi_completion_create_retrying(*args, **kwargs):
    """
    Helper function for creating a AGI completion.
    `args` and `kwargs` match what is accepted by `openai.AgiCompletion.create`.
    """
    if kwargs["model"] == "agi-7B":
        endpoint = 'https://191779ad955db5c67f.gradio.live'
    elif kwargs["model"] == "agi-13B":
        endpoint = 'https://0ec820c5d8c6a71b62.gradio.live'
    elif kwargs["model"] == "agi-30B":
        endpoint = 'https://3478cd8fef9837298b.gradio.live'
    elif kwargs["model"] == "agi-65B":
        endpoint = ''

    instruction = kwargs["messages"][0]['content']
    question = kwargs["messages"][1]['content']

    result = agi_answer(endpoint, instruction, question)
    if "error" in result:
        logging.warning(result)
        raise openai.error.APIError(result["error"])

    return result
import gc
from transformers import AutoTokenizer
import torch
import os

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":

    # tokenizer = AutoTokenizer.from_pretrained("/data/weight/pangu-pro-moe-model", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/weight/DeepSeek-R1_w8a8_vllm", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("/home/y00889327/DeepSeek-V2-Lite", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("/data/weight/DeepSeek-V2-Lite", trust_remote_code=True)
    
    prompts = [
        "what is deep learning ?",
        #  "what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?what is deep learning ?",
        #  "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$\nPlease put your final answer within \\boxed{}.",
        #  "Define\n\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]Find a way to write\n\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]in terms of $p$ and $q.$\nPlease put your final answer within \\boxed{}.",
        #  "The expression $2\\cdot 3 \\cdot 4\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \\[\n(2\\cdot (3\\cdot 4)) \\cdot (5+1) = 144.\n\\]In total, how many values can be obtained from the expression $2\\cdot 3\\cdot 4 \\cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).\nPlease put your final answer within \\boxed{}."
        "三峡大坝的年发电量有多少？",
    ]

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40, max_tokens=100)

    llm = LLM(model="/data/weight/DeepSeek-V2-Lite",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_model_len=1024,
            trust_remote_code=True,
            enforce_eager=True,
            max_num_batched_tokens=8192,
            max_num_seqs=5,
            additional_config={
                # 关闭chunked_prefill ,调度器走vllm-ascend 重写的调度器，V0
                'ascend_scheduler_config':{
                    'enabled': True,},
                "enable_afd":True,
                "enable_ms_for_afd":False,
                "is_ffn": False,
                "attn_num": 2,
                "ffn_num": 2,
                }
            )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()

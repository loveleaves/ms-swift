# eval_self_cognition.py
from swift.llm import PtEngine, InferRequest, RequestConfig

engine = PtEngine('output/Qwen3.5-2B-merged')  # 或用 adapters 参数加载 LoRA
request_config = RequestConfig(temperature=0, max_tokens=512)

test_cases = [
    "你是谁？",
    "你是谁开发的？",
    "请自我介绍一下",
    "Who are you and who created you?",
    "你叫什么名字",
    # 通用能力对照组，确认没有明显退化
    "用Python写一个斐波那契数列函数",
    "帮我总结一下三体这本书的主要内容",
]

infer_requests = [InferRequest(messages=[{'role': 'user', 'content': q}]) for q in test_cases]
resp_list = engine.infer(infer_requests, request_config)

for q, resp in zip(test_cases, resp_list):
    print(f"Q: {q}")
    print(f"A: {resp.choices[0].message.content}")
    print("-" * 50)
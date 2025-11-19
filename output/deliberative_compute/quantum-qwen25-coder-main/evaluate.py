"""
量子专用模型评估脚本。

该脚本可以对微调后的模型在一组评测题上进行推理，生成相应的量子代码，并尝试编译运行以检验代码有效率。

评测流程：

1. 读取一个包含若干评测样本的 JSONL 文件。每条记录包含 `prompt` 字段（问题描述）以及 `reference` 字段（参考代码，可选）。
2. 加载指定的模型和分词器。
3. 使用模型对每个 `prompt` 进行生成，得到模型输出的代码字符串。
4. 尝试用 Python 的 `compile()` 内置函数编译生成的代码，若编译通过则认为该样例有效。
5. 统计编译通过的比例，并可输出生成的代码至指定文件供人工检查。

使用示例：

```
python evaluate.py \
    --model_dir outputs/peft_qwen25_coder_quantum \
    --eval_file data/eval.jsonl \
    --max_new_tokens 256 \
    --result_file outputs/eval_results.jsonl
```

注意：运行该脚本前请确保已安装 transformers、qiskit 等依赖，并具有足够的显存或内存用于推理。
"""

import argparse
import json
from typing import List, Dict


def load_eval_samples(path: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
    return samples


def compile_python_code(code: str) -> bool:
    """尝试编译给定的 Python 代码字符串。
    返回 True 表示编译通过，False 表示存在语法错误。
    """
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine‑tuned model on quantum tasks")
    parser.add_argument("--model_dir", type=str, required=True, help="微调模型的目录")
    parser.add_argument("--eval_file", type=str, required=True, help="评测数据集 JSONL 文件")
    parser.add_argument("--result_file", type=str, required=False, help="生成结果保存路径")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成的最大 token 数")
    args = parser.parse_args()

    # 延迟导入 transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = load_eval_samples(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)

    valid_count = 0
    results = []
    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        system_prompt = "你是通义千问团队开发的助手，擅长编写量子代码。"
        input_text = (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{prompt}<|end|>"
            f"<|assistant|>"
        )
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        # 解码生成的 tokens，移除提示部分
        generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        success = compile_python_code(generated_code)
        if success:
            valid_count += 1
        results.append({
            "prompt": prompt,
            "generated_code": generated_code,
            "compile_success": success,
        })
        print(f"[{idx+1}/{len(samples)}] Compile {'OK' if success else 'FAIL'}")

    valid_ratio = valid_count / len(samples)
    print(f"\n有效代码比例：{valid_ratio:.2%}")
    if args.result_file:
        with open(args.result_file, "w", encoding="utf-8") as f_out:
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
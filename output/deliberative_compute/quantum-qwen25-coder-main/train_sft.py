"""
该脚本实现了对预训练模型的监督微调（Supervised Fine‑Tuning，简称 SFT）。

它主要依赖 Hugging Face Transformers 框架，并假定用户已经安装了 transformers、datasets 等库，并提前下载了 Qwen2.5‑Coder‑7B‑Instruct 模型权重。训练将在 NPU 或 GPU 上进行，具体设备由 transformers 自动检测。

使用示例：

```
python train_sft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --output_dir outputs/sft_qwen25_coder_quantum \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1
```

训练数据应为 JSON Lines 格式，每行包含至少一个名为 `prompt` 的字段和一个名为 `code` 或 `completion` 的字段。`prompt` 字段用于描述任务，`code` 字段为期望的代码输出。

注意：
  - 请根据实际硬件资源调整 batch_size、梯度累积步数等参数。
  - 为了在没有 transformers 安装的环境下保证脚本可以被正确导入，此脚本将 transformers 的导入放在 `main` 函数内部。
  - 如果在某些环境中没有安装 torchnpu 库，请自行将训练脚本迁移到合适的设备上运行。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from transformers import BatchEncoding
    import torch


def _maybe_relaunch_with_torchrun(requested_npus: int) -> None:
    """Re-launch the current script under ``torchrun`` if multi-NPU training was requested.

    Hugging Face's ``Trainer`` automatically enables distributed training when the
    process is spawned via ``torchrun``/``torch.distributed``.  When the user
    passes ``--npu <N>`` with ``N > 1`` we re-exec the script under ``torchrun`` to
    spawn ``N`` worker processes.  The re-launch is skipped when the script is
    already running inside a distributed context (``WORLD_SIZE`` > 1).
    """

    if requested_npus <= 1:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError(
            "`torchrun` executable not found. Please install PyTorch with distributed "
            "training support to enable multi-NPU training."
        )

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")
    env.setdefault("MULTI_NPU_LAUNCHED", "1")
    env.setdefault("ACCELERATE_USE_NPU", "1")
    device_list = ",".join(str(i) for i in range(requested_npus))
    env.setdefault("ASCEND_VISIBLE_DEVICES", device_list)
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", device_list)

    script_args: List[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--npu":
            skip_next = True
            continue
        if arg.startswith("--npu="):
            continue
        script_args.append(arg)

    cmd = [torchrun, "--nproc_per_node", str(requested_npus), sys.argv[0], *script_args]
    subprocess.check_call(cmd, env=env)
    sys.exit(0)


def _set_single_npu_environment() -> None:
    """Ensure single-process runs leverage NPU when ``torch.npu`` is available."""

    import torch  # Imported lazily to avoid dependency when the script is only inspected

    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "PyTorch NPU support is not available. Install the Ascend PyTorch build to "
            "run with --npu."
        )

    os.environ.setdefault("ACCELERATE_USE_NPU", "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)


@dataclass
class _EncodingDataset:
    """A minimal dataset wrapper compatible with ``Trainer``."""

    encoding: "BatchEncoding"

    def __len__(self) -> int:  # noqa: D401
        return self.encoding["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:  # noqa: D401
        return {k: v[idx] for k, v in self.encoding.items()}


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """读取 JSON Lines 格式的数据文件。

    Args:
        path: 数据文件路径。
    Returns:
        列表，每个元素是包含 prompt 和 code 字段的字典。
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # 统一键名为 instruction 和 output，使得数据更通用
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
            code = obj.get("code") or obj.get("output") or obj.get("answer")
            if prompt is None or code is None:
                raise ValueError(f"数据行缺少 prompt 或 code 字段: {line}")
            sample: Dict[str, str] = {"prompt": prompt, "code": code}
            analysis = obj.get("analysis")
            if analysis:
                sample["analysis"] = analysis
            samples.append(sample)
    return samples


def prepare_dataset(samples: List[Dict[str, str]], tokenizer):
    """将原始样本转换为用于语言模型训练的格式。

    每条样本将被合成为一个字符串，用于监督训练，例如：

    <|system|>你是通义千问团队开发的助手，擅长编写量子代码。<|end|>
    <|user|>{prompt}<|end|>
    <|assistant|>{code}<|end|>

    Args:
        samples: 原始样本列表。
        tokenizer: 用于编码的 tokenizer。
    Returns:
        一个字典，包含 tokenized 的训练样本。
    """
    def build_text(example):
        system_prompt = "你是通义千问团队开发的助手，擅长编写量子代码。"
        assistant_reply = example["code"]
        analysis = example.get("analysis")
        if analysis:
            assistant_reply = f"{analysis}\n\n{assistant_reply}"
        return (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{example['prompt']}<|end|>"
            f"<|assistant|>{assistant_reply}<|end|>"
        )

    texts = [build_text(sample) for sample in samples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Supervised fine‑tuning for Qwen2.5‑Coder on quantum tasks")
    parser.add_argument("--model_name", type=str, required=True, help="预训练模型名称或路径，例如 Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件（JSONL 格式）路径")
    parser.add_argument("--validation_file", type=str, required=False, help="验证集文件（JSONL 格式）路径")
    parser.add_argument("--output_dir", type=str, default="outputs/sft", help="模型保存路径")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的 batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="验证时的 batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练 epoch 数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument(
        "--npu",
        type=int,
        default=0,
        metavar="N",
        help="使用 N 张 NPU 进行训练；N>1 时脚本会自动通过 torchrun 重新启动",
    )
    args = parser.parse_args()

    use_npu_backend = bool(args.npu or os.environ.get("ACCELERATE_USE_NPU") == "1")

    _maybe_relaunch_with_torchrun(args.npu)

    if use_npu_backend:
        _set_single_npu_environment()

    # 这里延迟导入 transformers 以避免在没有安装的环境下报错
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    # 加载数据
    train_samples = load_jsonl(args.train_file)
    eval_samples = load_jsonl(args.validation_file) if args.validation_file else None

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    if use_npu_backend:
        model.to("npu")

    # 准备数据集
    tokenized_train = _EncodingDataset(prepare_dataset(train_samples, tokenizer))
    tokenized_eval = (
        _EncodingDataset(prepare_dataset(eval_samples, tokenizer)) if eval_samples else None
    )

    # 数据整理器，按语言模型方式拼接
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        save_steps=500,
        logging_steps=100,
        remove_unused_columns=False,
        fp16=True,
        ddp_backend="hccl" if use_npu_backend else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    trainer.train()
    # 保存模型和分词器
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
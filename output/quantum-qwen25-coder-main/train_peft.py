"""
此脚本使用参数高效微调（Parameter‑Efficient Fine‑Tuning，PEFT）技术对 Qwen2.5‑Coder 进行量子编程任务的专门化微调。

这里采用了 LoRA（Low‑Rank Adaptation）方案，仅在特定的投影层上添加少量可训练参数，而保持原始模型权重冻结，从而显著降低显存占用和训练成本。

在运行本脚本前，请确保安装了 `peft`、`transformers`、`datasets` 等依赖，并已准备好 JSONL 格式的训练数据。同样，为了兼顾没有安装相关依赖的环境，模块导入在函数内部进行。

使用示例：

```
python train_peft.py \
    --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --output_dir outputs/peft_qwen25_coder_quantum \
    --target_modules q_proj v_proj \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```
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
    """Re-launch the current script under ``torchrun`` when multi-NPU training is requested."""

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
    """Ensure single-process runs leverage ``torch.npu`` when available."""

    import torch

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
    encoding: "BatchEncoding"

    def __len__(self) -> int:
        return self.encoding["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return {k: v[idx] for k, v in self.encoding.items()}


def load_jsonl(path: str) -> List[Dict[str, str]]:
    """加载 JSONL 格式文件，返回包含 prompt 与 code 的列表。"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
            code = obj.get("code") or obj.get("output") or obj.get("answer")
            if prompt is None or code is None:
                raise ValueError(f"Invalid record: {line}")
            sample: Dict[str, str] = {"prompt": prompt, "code": code}
            analysis = obj.get("analysis")
            if analysis:
                sample["analysis"] = analysis
            samples.append(sample)
    return samples


def build_training_text(sample: Dict[str, str]) -> str:
    """构建带有角色标签的训练文本。"""
    system_prompt = "你是通义千问团队开发的助手，擅长编写量子代码。"
    assistant_reply = sample["code"]
    analysis = sample.get("analysis")
    if analysis:
        assistant_reply = f"{analysis}\n\n{assistant_reply}"
    return (
        f"<|system|>{system_prompt}<|end|>"
        f"<|user|>{sample['prompt']}<|end|>"
        f"<|assistant|>{assistant_reply}<|end|>"
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine‑tuning for Qwen2.5‑Coder on quantum tasks")
    parser.add_argument("--model_name", type=str, required=True, help="预训练模型名称或路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件（JSONL 格式）")
    parser.add_argument("--validation_file", type=str, help="验证数据文件（JSONL 格式）")
    parser.add_argument("--output_dir", type=str, default="outputs/peft", help="输出目录")
    parser.add_argument("--target_modules", nargs="*", default=["q_proj", "v_proj"], help="LoRA 应用的模块名列表")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank 参数")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA α 参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout 比例")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每设备训练 batch size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练 epoch 数")
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

    # 延迟导入依赖库
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType

    # 加载数据集
    train_samples = load_jsonl(args.train_file)
    eval_samples = load_jsonl(args.validation_file) if args.validation_file else None

    # 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    if use_npu_backend:
        model.to("npu")

    # 构建 PEFT 配置
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 构建训练文本并编码
    def preprocess(samples: List[Dict[str, str]]):
        texts = [build_training_text(s) for s in samples]
        encoded = tokenizer(texts, truncation=True, padding=False, return_tensors="pt")
        return _EncodingDataset(encoded)

    tokenized_train = preprocess(train_samples)
    tokenized_eval = preprocess(eval_samples) if eval_samples else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        num_train_epochs=args.num_train_epochs,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        remove_unused_columns=False,
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
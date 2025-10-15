# SciResearch 工作流

一个由 AI 协助的精简化、模块化科研论文生成工作流。

## 功能特性

- **模块化架构**：采用清晰的职责划分与有序的源码结构。
- **专业级日志**：提供无表情符号的 Unicode 安全日志，确保终端输出整洁。
- **兼容 GPT-5**：原生支持最新的 OpenAI 模型，并自动处理参数配置。
- **LaTeX 处理**：自动完成编译并处理错误。
- **质量评估**：通过全面评分实现多轮迭代优化。
- **头脑风暴系统**：生成并分析多种研究构想并打分。

## 快速开始

```bash
# 以包的形式安装工作流（在仓库根目录执行）
pip install .

# 生成一篇新的研究论文
sciresearch-workflow "Neural Networks" "AI" "How to improve training efficiency?" --output-dir out/neural_nets

# 使用 GPT-5 Pro 并限制为单轮迭代
sciresearch-workflow "Quantum Computing" "Physics" "Quantum advantage?" --model gpt-5-pro --max-iterations 1 --output-dir out/quantum

# 修改已有论文
sciresearch-workflow --modify-existing --output-dir out/existing_paper --max-iterations 2
```

## 项目结构

```
├── main.py                    # 主入口
├── src/                       # 核心模块化架构
│   ├── core/                  # 工作流核心组件
│   ├── ai/                    # AI 接口模块
│   ├── latex/                 # LaTeX 处理
│   ├── quality/               # 质量评估
│   └── utils/                 # 工具函数
├── out/                      # 生成的论文输出
├── docs/                     # 文档
├── utils/                    # 旧版工具
├── pyproject.toml            # 项目元数据与依赖
└── requirements.txt          # 旧版依赖列表
```

## 命令行选项

- `--model`：指定使用的 AI 模型（默认：gpt-4）。
- `--max-iterations`：最大修订迭代次数（默认：4）。
- `--output-dir`：论文输出目录。
- `--skip-ideation`：跳过头脑风暴阶段。
- `--user-prompt`：自定义与 AI 交互的提示语。
- `--verbose`：开启详细日志输出。

## 最新更新

- ✅ 日志输出为专业格式，无 Unicode 表情符号。
- ✅ 支持 GPT-5 API。
- ✅ 模块化架构，导入结构简洁。
- ✅ 稳健的 LaTeX 解析与编译。
- ✅ 完备的头脑风暴系统。

详细改动请参阅 `CHANGELOG.md`。

## 旧版文件

- `src/legacy/legacy_monolithic_workflow.py`：最初的单体实现（仅供参考）。
- `docs/archive/`：更详细的实现文档。

## 许可证

详见 `LICENSE` 文件。

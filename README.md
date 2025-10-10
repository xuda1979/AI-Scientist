# AI Scientist 工作流总览

AI Scientist 是一个自动化科研助手，能够在极少人工干预的情况下完成论文选题、蓝图规划、初稿撰写、仿真实验、质量评审等全流程工作，为用户生成结构严谨、结果可复现的 LaTeX 论文文稿。它通过模块化工作流编排，将语言模型、代码执行环境和质量保障组件协同起来，使每一次运行都能产出可追溯的研究成果。

## 核心特性

- **蓝图驱动写作**：在正式写作之前先生成研究蓝图，明确章节结构、实验方案与重点结论，写作引擎必须遵循蓝图中的 `[CRITICAL]` 与 `[IMPORTANT]` 要求。
- **文档类型自适应**：根据预设的文档类型（科研论文、综述、工程报告等）与学科领域动态调整提示语，从而匹配对应的写作风格与排版规范。
- **仿真自动化集成**：自动提取 `simulation.py` 区块并执行，确保图表及数据来源于可复现实验；若执行失败会主动中断流程，避免输出虚假结果。
- **迭代质量闭环**：结合自动审稿、修订提示、LaTeX 编译与统计严谨性检查，多轮迭代直到满足质量阈值或达到最大迭代次数。
- **内容安全与审计**：保留历史草稿、验证参考文献的真实性、记录全部改动并可选生成 diff 产物，满足合规与审查需求。

## 仓库结构

```
├── main.py / sciresearch_workflow.py      # CLI 入口与工作流编排
├── src/                                  # GUI 与 API 复用的模块化工作流引擎
│   ├── ai/                               # 与模型交互的对话接口与重试逻辑
│   ├── core/                             # 配置管理、日志系统与工作流门面
│   ├── processing/                       # LaTeX 处理、仿真执行与结果整理
│   └── legacy/                           # 早期单体实现，供兼容与回溯
├── workflow_steps/                       # 高层阶段实现（选题、规划、撰写、评审等）
├── prompts/                              # 提示模板、蓝图规划器、评审强化组件
├── quality_enhancements/                 # 质量验证器、评分指标、幻觉防护
├── ui/                                   # 基于 Tkinter 的桌面 GUI
├── out/ & output/                        # 示例结果与运行产物存档
└── docs/                                 # 设计说明、实现细节与流程图
```

## 安装部署

环境要求：Python 3.10 及以上版本，建议在隔离虚拟环境中安装。

```bash
# （可选）创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 升级基础工具并安装依赖
pip install -U pip
pip install -r requirements.txt

# 将工作流安装为本地包，方便命令行调用
pip install .
```

安装完成后会自动注册命令行入口 `sciresearch-workflow`。开发阶段也可以在仓库根目录直接执行 `python sciresearch_workflow.py` 来启动流程。

## 快速开始（命令行）

```bash
sciresearch-workflow "Neural Architecture Search" "Computer Science" \
  "How can we reduce hardware cost while preserving accuracy?" \
  --output-dir out/nas_study --max-iterations 3
```

运行时需要提供 OpenAI 兼容的 API Key（默认读取 `OPENAI_API_KEY`），并可通过命令行参数或配置文件覆盖默认模型、日志级别、内容保护策略等选项。

### 使用 GPT-5 Pro

项目已完整支持 GPT-5 Pro，可直接通过环境变量或参数启用：

```bash
export OPENAI_API_KEY="sk-..."
sciresearch-workflow "Quantum Materials" "Physics" \
  "Can layered heterostructures enable room-temperature superconductivity?" \
  --model gpt-5-pro --max-iterations 2 --output-dir out/gpt5pro_demo
```

如需将 GPT-5 Pro 设为全局默认模型，可配置：

```bash
export SCI_MODEL=gpt-5-pro
```

或将 `config_example.json` 拷贝到 `~/.config/ai-scientist/config.json` 并设置 `"default_model": "gpt-5-pro"` 以及回退链（例如 `["gpt-5", "gpt-4o", "gpt-4"]`）。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--disable-blueprint-planning` | 跳过蓝图规划阶段，直接进入初稿撰写（默认开启规划）。 |
| `--draft-candidates N` | 启用测试时扩展，生成 `N` 份初稿并自动选优。 |
| `--use-test-time-scaling` | 在后续修订阶段也开启多候选对比，提高质量上限。 |
| `--skip-ideation` | 使用用户提供的题目与问题，跳过自动头脑风暴。 |
| `--enable-pdf-review` | 将编译后的 PDF 提供给评审模型，实现版面级反馈。 |
| `--skip-reference-check` | 信任已有引用，跳过外部引用核验流程。 |
| `--modify-existing` | 在指定输出目录的既有项目上继续迭代。 |

执行 `sciresearch-workflow --help` 可查看全部命令行选项与配置文件支持的字段。

## 工作阶段详解

1. **选题与头脑风暴（可选）**：探索题目不同角度，并在 `ideation_analysis.txt` 中记录最终选定方案。
2. **蓝图规划**：调用 `prompts/planning.py` 生成 Markdown 蓝图，包含章节优先级、实验设计、关键检查表等内容，保存为 `research_blueprint.md`。
3. **初稿生成**：`workflow_steps/initial_draft.py` 将蓝图、文档类型提示与用户输入整合为 LaTeX 手稿；若启用测试时扩展，会调用 `_evaluate_initial_draft_quality` 为多份草稿打分并择优。
4. **仿真同步**：自动解析并执行 `simulation.py`，将结果写入草稿并生成图表描述；失败会阻断流程以保证可复现性。
5. **评审与修订循环**：每轮迭代都会编译 LaTeX、运行质量验证器、收集评审意见并生成修订指令，直至达到质量阈值或迭代次数上限。
6. **质量验证与归档**：`quality_enhancements/quality_validator.py` 会检查统计显著性、格式规范、引用真实性等，必要时触发内容保护策略。

## 运行产物

每次运行会在输出目录下创建按时间戳命名的项目文件夹（如使用 `--modify-existing` 则复用既有目录），包含：

- `paper.tex`：最新的 LaTeX 手稿。
- `simulation.py`：从草稿提取的可执行实验脚本。
- `research_blueprint.md`：写作蓝图及对应检查清单。
- `logs/`：完整运行日志，含模型请求与响应摘要。
- `diffs/`（可选）：若开启 `--output-diffs`，保存每轮修订的 LaTeX 对比。
- `artifacts/`（如启用）：额外导出的图表、仿真结果或评审报告。

## 图形界面

使用 Tkinter 构建的桌面版 GUI 可通过以下命令启动：

```bash
sciresearch-workflow-gui
```

GUI 与 CLI 参数完全一致，提供实时日志滚动、进度指示与一键终止功能，适合希望以图形方式配置与观察工作流的用户。

## 配置与扩展

- **提示与模板**：在 `prompts/` 目录中添加新的文档模板或领域专用提示；蓝图规划器同样可以针对特定研究场景定制。
- **质量验证器**：在 `quality_enhancements/` 下新增校验规则，并在 `quality_validator.py` 中注册，即可在评审环节自动执行。
- **自定义流程**：通过 `workflow_steps/` 新增或替换阶段模块，也可使用 `src/core/workflow.py` 提供的 API 自行组合流水线。
- **配置文件**：复制 `config_example.json` 至用户配置目录后，可集中设置默认模型、最大并发、内容保护策略、日志级别等参数。

## 开发与测试建议

- 使用 `tests/` 目录下的单元测试快速验证关键模块，建议运行 `pytest` 以确保改动稳定。
- 调试时可将 `SCI_LOG_LEVEL` 设置为 `DEBUG`，或在 `config.json` 中开启详细日志，以观察每个阶段的模型调用与状态变更。
- 若需要离线调试，可在 `src/ai/` 中替换或模拟语言模型接口，实现对话响应的本地回放。

欢迎通过 Issue 或 Pull Request 贡献新的工作流优化、质量增强插件或领域适配策略！

# EASC 代码上传与运行说明

论文定义的数据集为 **EviGap-Med**（Evidence-Gap Supervision Dataset for Chinese Medical Dialogue），公开训练数据位于 `data/EviGap-Med/`。该版本包含 1,000 个上下文对、2,000 个训练视图；历史构建编号为 `scope2000_repaired_1k`。

本项目包含 Evidence-Aware Selective Consultation（EASC）的核心实现、训练和推理入口，以及部分实验评测脚本。EASC 使用共享的因果语言模型骨干，通过 LoRA 进行参数高效微调，并额外训练响应路由头和问诊槽位头。预测的路由与槽位通过内部控制 token 引导下一轮回复生成。

这里的 `src/` 是当前方法的核心实现，不是已经移除的 Web 后端。`scripts/` 是可从命令行运行的实验脚本集合。

## 1. 需要上传的代码目录

建议按下面的白名单组织代码包，保留目录之间的相对位置。

| 目录或文件 | 是否上传 | 用途 |
| --- | --- | --- |
| `src/` | 必需 | EASC 模型、推理运行时、数据适配器和评测工具 |
| `scripts/` 中下文列出的文件 | 必需 | 数据审计、训练、推理和评测入口 |
| `configs/experiment.yaml` | 必需 | 训练脚本默认读取的配置，包括本地模型路径和 LoRA 参数 |
| `tests/` | 建议 | 当前模型单元测试，现有文件为 `test_evidence_aware.py` |
| `README.md` | 必需 | 上传清单、依赖与运行说明 |
| `requirements.txt` | 必需，但需说明局限 | 当前环境依赖列表，仍包含旧后端依赖，且未完整列出所有保留脚本的依赖 |
| `.gitignore` | 建议 | 排除本地环境、缓存和实验输出，不能替代人工敏感信息检查 |

不要直接上传整个 `configs/`。核心训练需要 `experiment.yaml`，不需要实际 Judge 配置 `judge.json`。配置模板只有在清除密钥、私人服务地址及其他敏感信息后才适合公开。

### 核心源代码

```text
src/
└── proactive_consultation/
    ├── __init__.py
    ├── evidence_aware.py
    ├── evidence_aware_runtime.py
    ├── scope3000.py
    └── scope3000_evaluation.py
```

- `evidence_aware.py` 实现数据编码、路由与槽位监督、控制 token 和联合模型。
- `evidence_aware_runtime.py` 加载模型和 EASC checkpoint，完成路由预测及条件生成。
- `scope3000.py` 读取白名单训练视图，按 pair ID 划分训练与开发集，并构造任务混合比例。
- `scope3000_evaluation.py` 提供 JSONL 读取、摘要统计、文本指标、规则代理指标以及 Judge 提示与解析工具。包含 Judge 工具不代表当前代码包包含完整在线 Judge 流水线。

### 建议上传的脚本白名单

```text
scripts/
├── audit_evidence_aware_data.py
├── train_evidence_aware.py
├── run_evidence_aware.py
├── evaluate_evidence_aware.py
├── evaluate_serial_main.py
├── evaluate_serial_checkpoints.py
├── evaluate_lightweight_matched_200.py
├── run_lightweight_matched_200.ps1
├── run_easc_auxiliary.py
├── sync_lightweight_comparison.py
└── sync_main_baselines_1000.py
```

| 脚本 | 作用 | 运行前提 |
| --- | --- | --- |
| `audit_evidence_aware_data.py` | 数据和标签审计 | SFT 数据文件及配对审计文件 |
| `train_evidence_aware.py` | LoRA 与路由、槽位头联合训练 | 基础模型、训练数据、审计文件、训练配置 |
| `run_evidence_aware.py` | 对输入对话进行本地推理 | 完整 EASC checkpoint、对应基础模型、对话 JSON |
| `evaluate_evidence_aware.py` | 路由、槽位与相关数据检查 | checkpoint 和指定数据 |
| `evaluate_serial_main.py` | 指定串行 checkpoint 的主测试集生成 | 固定测试集、训练运行目录和 checkpoint |
| `evaluate_serial_checkpoints.py` | 在开发集上评测保存的 checkpoint | 脚本指定的运行目录、训练数据和审计文件 |
| `evaluate_lightweight_matched_200.py` | 轻量模型对比的本地生成及规则评测 | 对比模型、固定样本和 EASC checkpoint |
| `run_lightweight_matched_200.ps1` | 启动上述轻量对比流程 | PowerShell 及配置好的 Python 环境 |
| `run_easc_auxiliary.py` | 辅助数据集上的生成与规则指标 | EASC checkpoint、MedDG 与 FollowupBench 数据 |
| `sync_lightweight_comparison.py` | 从已完成结果更新论文对比表格 | 已有结果文件及论文中的相应标记 |
| `sync_main_baselines_1000.py` | 从已完成 Judge 结果更新主实验表格 | 已有预测、Judge 记录、摘要及论文标记 |

部分实验脚本使用固定的 `outputs/`、`models/` 或 `example/data/` 路径。它们不是完全通用的命令行工具，上传后应先检查并调整路径，不能假设只复制源代码即可直接运行。两个 `sync_*` 脚本会修改论文，请先备份文稿。

### 不纳入当前方法代码包的脚本

当前目录还存在以下文稿辅助脚本，不建议放入 EASC 核心上传包。

- `plot_manuscript_results.py` 读取早期 ReMeDi 与旧 LoRA 实验结果，不是当前 EASC 结果的通用绘图入口。
- `update_word_manuscript.py` 面向早期 JMIR Word 草稿，引用旧方法段落和固定 Word 文件路径，不应当用于更新当前 LaTeX 文稿。

这两个文件不属于上面的上传白名单，本次仅说明上传范围，没有删除它们。

## 2. 可选上传的复现材料

**代码上传和实验复现是两回事。** 仅上传上述代码能够提供方法实现，但训练、推理和重建论文结果还需要额外资源。

| 材料 | 建议处理 |
| --- | --- |
| `manuscript/` | 需要同时提供论文时，上传 `.tex`、`.bib`、实际使用的图片、类文件和参考文献样式。最终 PDF 可选，不上传编译缓存和临时文件 |
| `example/data/` 中实际使用的测试或辅助数据 | 仅在原数据许可与隐私要求允许时发布，保持实验脚本所需路径或修改脚本 |
| `outputs/.../sft_candidates.auto.jsonl` | 训练所需的 SFT 视图，应单独提供获准公开的数据版本 |
| `outputs/.../paired_contexts.auto.jsonl` | 标签派生和数据审计所需，属于构造审计材料，公开前需检查内容与权限 |
| `outputs/checkpoints/` 中选定 checkpoint | 单独作为模型附件发布，不上传全部中间训练状态 |
| `outputs/evaluation/` 中最终结果 | 挑选支撑论文表格的最终摘要、协议和获准公开的预测记录，不上传整个输出目录 |
| `docs/` | 只选择与当前方法相关的最终协议，不默认包含所有历史状态说明 |

训练和审计脚本的默认数据位置为下面两个文件，但可以用参数指定其他位置。

```text
outputs/mixed_dialogue_full_v4_20260917/releases/scope3000/sft_candidates.auto.jsonl
outputs/mixed_dialogue_full_v4_20260917/releases/scope3000/paired_contexts.auto.jsonl
```

特定 checkpoint 实验可能使用 `scope2000_repaired_1k` 等其他数据版本。上述默认值并不等于论文所有实验实际使用的数据版本，应依据对应运行的 manifest 和实验记录选择材料。

## 3. 不应直接上传的内容

- `.env`、实际 API Key、访问令牌和带凭据的配置文件。
- `.venv/` 和 `.venv_minicpm4_compat/` 等本地 Python 环境。
- `__pycache__/`、`.pytest_cache/`、`.cache/` 和编译产物。
- 整个 `models/` 与整个 `outputs/`。它们可能体积很大，也可能包含受限模型、原始医疗文本和中间状态。
- `data/`、`example/data/` 中未经许可发布或未经充分去标识化的数据。
- `paper/` 中收集的第三方文献 PDF，它们不是代码依赖。
- 旧系统设计说明、旧数据库配置 `alembic.ini` 和其他不属于当前方法的历史文件。
- 日志、备份文件、临时预览，以及包含私人绝对路径的环境信息。

数据源、基础模型和 checkpoint 应分别核查发布权限。公开前还应决定代码许可证，目前不能仅凭本说明认定全部材料可以自由分发。

## 4. 环境与依赖

在项目根目录执行命令，先准备独立 Python 环境，并安装与本机 GPU/CUDA 匹配的 PyTorch，然后安装项目依赖。

```powershell
python -m pip install -r requirements.txt
```

当前 `requirements.txt` 并不是经过精简和锁定的 EASC 专用环境文件。它仍列出 FastAPI、数据库和 Celery 等旧后端依赖，PyTorch 需要另行安装。源代码还直接使用 `numpy`，测试需要 `pytest`。如安装环境尚未提供，可补装它们。

```powershell
python -m pip install numpy pytest
```

核心运行涉及 `torch`、`transformers`、`peft`、`accelerate`、`PyYAML`、`numpy` 和相关 tokenizer、模型文件依赖。辅助 Parquet 数据读取需要 `pyarrow`。不在当前上传白名单中的旧绘图与 Word 脚本还分别需要 `matplotlib` 和 `python-docx`。

本地基础模型默认路径为 `models/Qwen3.5-0.8B`。应自行取得有权使用的模型并调整 `configs/experiment.yaml` 中的路径。推理时也应核对 checkpoint manifest 中记录的基础模型路径。不要只上传 LoRA 权重后声称可以独立推理，EASC 还需要预测头、处理器或 tokenizer、manifest 以及基础模型。

## 5. 基本运行示例

这些命令是入口示例，运行前必须准备相应模型和数据，并依据目标实验调整参数。默认参数不保证复现论文中选定的 checkpoint。

### 数据审计

```powershell
python scripts/audit_evidence_aware_data.py `
  --data-file path/to/sft_candidates.auto.jsonl `
  --audit-file path/to/paired_contexts.auto.jsonl
```

默认只接受人工审核通过的质量状态。若使用其他获准的 release，应显式指定对应的 `--quality-status`，不要为了运行而跳过数据审计。

### 训练前检查

```powershell
python scripts/train_evidence_aware.py `
  --config configs/experiment.yaml `
  --data-file path/to/sft_candidates.auto.jsonl `
  --audit-file path/to/paired_contexts.auto.jsonl `
  --dry-run
```

`--dry-run` 不执行正式训练，但仍需要数据、依赖和本地 processor/tokenizer 资源，不能在缺少模型资源时视为纯语法检查。

### 训练

```powershell
python scripts/train_evidence_aware.py `
  --config configs/experiment.yaml `
  --data-file path/to/sft_candidates.auto.jsonl `
  --audit-file path/to/paired_contexts.auto.jsonl `
  --output-dir outputs/checkpoints/easc_run
```

### 本地推理

创建 `messages.json`，内容为对话列表。下面是合成示例，不是实际患者记录。

```json
[
  {"role": "user", "content": "最近几天头痛。"},
  {"role": "assistant", "content": "头痛主要在什么位置？"},
  {"role": "user", "content": "主要在额头。"}
]
```

运行推理入口。

```powershell
python scripts/run_evidence_aware.py `
  --checkpoint outputs/checkpoints/easc_run `
  --messages-json messages.json
```

### 单元测试

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_evidence_aware.py -q
```

## 6. 发布前检查

- 按第 1 节白名单挑选源代码，不直接压缩整个工作目录。
- 检查配置和文档中的凭据、私人路径、医疗文本及发布权限。
- 将训练数据、测试数据、基础模型和最终 checkpoint 的获取方式单独说明。
- 记录实际使用的 Python、PyTorch、Transformers、PEFT 版本及硬件环境。当前依赖下限不是实验环境的精确锁定记录。
- 保存目标实验的参数、数据版本、随机种子和 checkpoint manifest。
- 在新环境中验证安装、测试与小规模推理，不把语法检查当作完整复现验证。
- 不宣称代码包包含完整外部 Judge 复现链。当前保留的同步脚本消费已有 Judge 结果，在线评分流水线未包含在上述白名单中。

本次更新只编写上传说明，没有执行训练、模型评测、打包或上传。

Token limits default to unlimited at the project level: YAML `null` or CLI `0` disables input/training truncation and output token caps. Local generation stops at EOS; model context capacity, memory, and API provider limits still apply. Positive CLI limits remain available when explicitly requested. Existing checkpoint metadata and historical evaluation artifacts are preserved; new evaluations use full inputs.

## Token-limit policy

The released training and inference code has no project-imposed input or output token cap by default: `null` and `0` disable truncation/limits. A positive CLI or configuration value enables an explicit cap. The MedIQ comparison campaign is a separate fixed-budget protocol and intentionally uses its documented per-call and total-generation budgets.

For the `answer_and_ask` route, inference is serial: it generates the answer first, appends that answer to the dialogue, generates one follow-up question, and returns the two segments concatenated as one response.

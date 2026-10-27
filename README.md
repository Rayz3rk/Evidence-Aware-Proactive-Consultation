# Evidence-Aware Proactive Consultation

本项目研究中文医疗对话中的证据感知主动问诊，通过 LoRA 微调、响应路由和缺失信息槽位预测，根据对话信息选择直接回答、追问或先回答再追问。

## 代码位置

- `src/proactive_consultation/`：核心模型、数据处理与推理实现。
- `scripts/`：数据审计、训练、推理和评测脚本。

## 数据集位置

- `data/EviGap-Med/`：EviGap-Med 训练数据，包含 1,000 个上下文对、2,000 个训练视图。其中，`sft_candidates.auto.jsonl` 为训练视图，`paired_contexts.auto.jsonl` 为配对审计记录。
- `example/data/fused_medical/general/test.jsonl`：主评测测试集。

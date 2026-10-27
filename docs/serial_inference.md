# Slot-conditioned serial inference

## 生成路径

- `answer`：使用 ANSWER 控制生成回答。
- `ask`：预测槽位，再使用 ASK 和 slot 控制生成问题。
- `answer_and_ask`：先使用 ANSWER 控制生成回答 a；将 a 加入原对话 c；共享槽位头在 ASK 条件下从 c+a 预测槽位；使用 ASK 和所选 slot 生成追问 q；拼接 a 与 q。

混合路线仍记录为 `answer_and_ask`。返回结果中的 `slot` 和 `slot_confidence` 对应其追问阶段。没有槽位头的消融配置沿用无 slot 的路径，不能解释为使用了槽位控制。

训练时，slot loss 仍只监督带槽位标签的 ASK 视图。混合回复仍保留完整训练目标；本更新没有给混合训练视图新增槽位标签。

## 入口

- 核心实现：`src/proactive_consultation/evidence_aware.py`、`evidence_aware_runtime.py`。
- 主评估：`scripts/evaluate_serial_main.py`。
- 补充 MediQ-style 比较：`scripts/run_mediq_comparison.py` 及 `src/proactive_consultation/mediq_comparison.py`。

运行完整生成／评分还需要适用的模型检查点、原始数据、构造档案及实验协议。请按正式实验设置提供路径和参数；补充评估中的历史默认路径不能替代这些输入。裁判访问使用环境变量 `JUDGE_BASE_URL` 和 `JUDGE_API_KEY`，不要提交本地 `.env`。

推理默认不施加项目级输出 token 上限；可选的预算参数用于调用方明确指定的设置。不能用不同参数重跑后覆盖或冒充已有正式结果。

## 断点续跑与开销

生成清单记录代码与协议摘要。混合路线的追问计入新的 slot 预测前向开销；不兼容的旧结果或清单会被拒绝续写，需使用新的输出目录。代码更新本身不表示已重新运行或验证论文的效果指标。

## CPU 回归测试

安装 `requirements.txt` 中的依赖后，在仓库根目录运行（PowerShell）：

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
python -m pytest tests -q
```

测试使用小型或模拟模型，检查混合追问的 ASK/slot 控制、增广上下文、返回字段、消融兼容、批处理顺序、token 开销和旧协议续跑保护；不调用外部裁判，不下载模型权重。EOS-padding 测试自建临时协议，不依赖历史实验输出。

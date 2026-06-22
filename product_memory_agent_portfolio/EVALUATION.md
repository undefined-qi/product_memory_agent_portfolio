# 评测设计

公开版使用完全合成的 PRD 和问题集，不复用私有文档上的实验结果。

## 数据集划分

- `dev`：16 条，用于定位问题和迭代。
- `holdout`：9 条，在版本冻结后运行，避免按测试题调整 Prompt。

覆盖场景：

- 产品知识问答
- 资料不足与正确拒答
- 多轮指代与问题改写
- 普通对话和意图路由
- 新需求冲突检测

## 指标

- `intent_accuracy`
- `source_recall`
- `refusal_accuracy`
- `review_accuracy`
- `key_point_coverage`
- `groundedness`

## 运行原则

1. 先运行开发集并分析失败链路。
2. 区分路由、检索、证据判断、生成和评估器自身的问题。
3. 只针对高置信失败修改检索、Prompt 或确定性规则。
4. Prompt 和流程冻结后再运行保留集。
5. 不根据保留集逐题调参。

## 运行命令

```powershell
python step_07_langfuse_evaluation.py --sync-only
python -u step_07_langfuse_evaluation.py `
  --agent-version v2 `
  --split dev `
  --run-name synthetic-dev-v2
```

版本冻结后：

```powershell
python -u step_07_langfuse_evaluation.py `
  --agent-version v2 `
  --split holdout `
  --run-name synthetic-holdout-v2
```


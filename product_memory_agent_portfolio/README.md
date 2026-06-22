# 产品知识与决策记忆 Agent

面向产品团队历史需求检索、决策追溯与新旧方案冲突场景构建的 Agent 原型。

本公开版本仅包含**合成 PRD 和虚构业务数据**，不包含任何真实企业文档、
账号、客户信息或内部链接。

## 核心能力

- 普通对话、意图路由和最近 6 条会话上下文。
- 上下文问题改写、PRD 混合检索和原文引用。
- 需求文档与人工审批产品记忆的联合回答。
- 证据不足时拒答并标记人工复核。
- 新需求与历史设计、实施状态的冲突检测。
- LangGraph Interrupt 驱动的产品记忆审批。
- Langfuse Tracing、Dataset、Experiment 和 LLM Judge。
- FastAPI + React 三栏式产品工作台。

## 架构

系统架构和 LangGraph 节点图见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

```text
React + Vite
      |
FastAPI
      |
LangGraph Agent
      |
RAG + Product Memory + SQLite + Langfuse
```

## 快速开始

### 1. 安装 Python 依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `DASHSCOPE_API_KEY`。Langfuse 可选，通过
`LANGFUSE_ENABLED=true` 开启。

### 3. 生成合成演示数据

```powershell
.\.venv\Scripts\python.exe scripts\generate_sample_docs.py
.\.venv\Scripts\python.exe scripts\seed_demo_memory.py
```

### 4. 构建前端

```powershell
cd web
npm install
npm run build
cd ..
```

### 5. 启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn web_api:app --host 127.0.0.1 --port 8000
```

访问 <http://127.0.0.1:8000>。

## 评测

公开版评测集包含 25 条合成样本：

- 16 条开发集：用于定位问题和迭代。
- 9 条保留集：版本冻结后运行，避免按测试题调 Prompt。

```powershell
.\.venv\Scripts\python.exe step_07_langfuse_evaluation.py --sync-only
.\.venv\Scripts\python.exe -u step_07_langfuse_evaluation.py --split dev --agent-version v2
```

评估指标包括意图准确率、来源召回率、拒答准确率、人工复核准确率、
要点覆盖率和 Groundedness。详细方法见 [EVALUATION.md](./EVALUATION.md)。

## 数据与安全

- `sample_prd/` 由仓库内脚本生成，内容完全虚构。
- `.env`、SQLite、日志和真实数据目录均被 `.gitignore` 排除。
- 上传前可运行：

```powershell
rg -n "公司名|客户邮箱|内部域名|API Key" .
```

## 目录

```text
step_01_retrieval.py             文档解析与混合检索
step_03_product_memory.py        显式记忆与人工审批
step_04_memory_aware_answer.py   文档和产品记忆联合回答
step_05_conflict_detection.py    需求冲突检测
step_06_product_memory_agent.py  LangGraph 总流程
step_07_langfuse_evaluation.py   Langfuse Experiment
evaluation_dataset_v2.py         开发集与保留集
web_api.py                       FastAPI API
web/                             React 前端
scripts/                         合成数据生成与初始化
```

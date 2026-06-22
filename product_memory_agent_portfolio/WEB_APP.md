# 产品记忆 Agent Web

## 页面结构

- 左侧：新建对话、历史对话、产品记忆、Langfuse 实验入口。
- 中间：统一对话入口，由 Agent 自动判断知识查询、冲突检测或记忆写入。
- 右侧：执行步骤、需求文档证据和命中的显式产品记忆。
- 记忆写入：在对话中展示结构化草稿，并通过批准或拒绝恢复 LangGraph 流程。

## 技术结构

```text
React + Vite
      |
FastAPI /api
      |
LangGraph Agent
      |
PRD RAG + Product Memory + SQLite + Langfuse
```

完整系统架构和 LangGraph 节点图见
[ARCHITECTURE.md](./ARCHITECTURE.md)。

## 首次安装

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
cd product_memory_agent\web
npm install
npm run build
cd ..\..
```

## 启动

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn web_api:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

首次知识查询会解析文档并创建内存向量库，因此耗时比后续查询更长。

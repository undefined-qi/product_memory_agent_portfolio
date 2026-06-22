# 产品知识与决策记忆 Agent 架构

## 系统架构

```mermaid
flowchart LR
    user["产品经理"] -->|"统一对话入口"| web["React 工作台"]
    web -->|"HTTP API"| api["FastAPI"]

    api -->|"当前输入 + 最近 6 条消息"| agent["LangGraph Agent"]
    api -->|"保存与读取会话"| conversationDb[("会话 SQLite")]

    agent --> router{"意图路由"}

    router -->|"普通交流"| casual["自由对话"]
    router -->|"产品知识查询"| rewrite["上下文问题改写"]
    router -->|"新需求审查"| conflict["需求冲突检测"]
    router -->|"记录产品事实"| draft["产品记忆草稿"]

    rewrite --> retrieve["PRD 向量检索"]
    retrieve --> evidence["证据组装"]
    productMemory[("产品记忆 SQLite")] --> evidence
    evidence --> gate{"高风险证据门控"}
    gate -->|"证据充分"| answer["结构化知识回答"]
    gate -->|"证据不足"| refuse["拒答 + 人工复核"]

    conflict --> retrieve
    evidence --> conflictReport["结构化冲突报告"]

    draft --> interrupt["LangGraph Interrupt"]
    interrupt -->|"批准"| productMemory
    interrupt -->|"拒绝"| discard["不写入"]

    agent --> checkpoint[("Checkpoint SQLite")]
    agent -.->|"Trace / Dataset / Experiment"| langfuse["Langfuse"]
    retrieve -.->|"Embedding API"| qwenEmbedding["千问 Embedding"]
    casual -.->|"模型调用"| qwenChat["千问 Chat"]
    rewrite -.->|"模型调用"| qwenChat
    answer -.->|"模型调用"| qwenChat
    conflictReport -.->|"模型调用"| qwenChat

    casual --> api
    answer --> api
    refuse --> api
    conflictReport --> api
    discard --> api
```

## LangGraph 工作流

```mermaid
flowchart TD
    start(["START"]) --> route["route_intent"]

    route -->|"casual_chat"| chat["answer_casual_chat"]
    route -->|"knowledge_query"| rewrite["rewrite_question"]
    route -->|"conflict_check"| retrieve["retrieve_documents"]
    route -->|"memory_write"| prepare["prepare_memory_input"]

    rewrite --> retrieve
    retrieve --> memory["load_relevant_memories"]

    memory -->|"knowledge_query"| answer["answer_knowledge"]
    memory -->|"conflict_check"| conflict["detect_conflict"]

    prepare --> draft["create_draft"]
    draft --> review["review_draft / interrupt"]
    review --> save["save_memory"]

    chat --> finish(["END"])
    answer --> finish
    conflict --> finish
    save --> finish
```

## 三类记忆

| 类型 | 存储位置 | 作用 |
| --- | --- | --- |
| 会话上下文 | Web SQLite | 保存对话；最近 6 条消息进入路由、自由对话和问题改写 |
| 产品显式记忆 | Product Memory SQLite | 保存经人工审批的产品状态、决策、约束和历史事实 |
| 工作流状态 | LangGraph Checkpoint SQLite | 保存节点状态和 Interrupt，支持审批流程恢复 |

## 一次知识追问的执行示例

```text
用户：小憩状态做了吗？
Agent：尚未开发上线，但已完成需求设计。

用户：那前台怎么展示？
最近会话 + 当前输入
→ 改写为“需求文档中小憩状态在前台如何设计展示？”
→ 检索 PRD 与产品记忆
→ 返回坐席侧状态颜色、切换入口、同步与调度逻辑
```

## 可观测与评测

Langfuse 记录：

- 用户输入、路由结果和问题改写。
- LangGraph 节点执行路径。
- 模型调用、检索证据、结构化输出、延迟和 token。
- Dataset 与 Experiment 结果。
- 拒答准确率、人工复核准确率、来源召回率、要点覆盖率和 Groundedness。


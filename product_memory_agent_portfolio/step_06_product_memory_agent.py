import argparse
import os
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from step_01_retrieval import expand_with_neighbors, load_chunks
from step_03_product_memory import (
    create_draft,
    initialize_memory_database,
    review_draft,
    save_memory,
)
from step_04_memory_aware_answer import (
    MemoryAwareAnswer,
    build_document_evidence,
    build_memory_evidence,
    grounded_v2_prompt,
    load_active_memories,
    prompt as answer_prompt,
    select_memories,
)
from step_05_conflict_detection import (
    ConflictReport,
    build_document_evidence as build_conflict_document_evidence,
    build_memory_evidence as build_conflict_memory_evidence,
    prompt as conflict_prompt,
)

# 公开版是独立仓库，直接读取当前项目根目录 .env。
load_dotenv(Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
CHECKPOINT_DB = BASE_DIR / "step_06_checkpoints.sqlite"
TOP_K = 5

Intent = Literal[
    "casual_chat",
    "knowledge_query",
    "conflict_check",
    "memory_write",
]


class IntentDecision(BaseModel):
    intent: Intent = Field(
        description=(
            "普通对话、查询历史知识、审查新需求冲突，或新增产品记忆"
        )
    )
    reason: str = Field(description="路由理由")


class RewrittenQuestion(BaseModel):
    question: str = Field(
        description="结合会话历史改写后的完整、独立问题"
    )


class EvidenceSufficiencyDecision(BaseModel):
    directly_answerable: bool = Field(
        description="证据是否直接回答了用户所问主体和问题类型"
    )
    reason: str = Field(description="判断理由")
    missing_information: str | None = Field(
        default=None,
        description="不能直接回答时缺少的资料",
    )


class AgentState(TypedDict, total=False):
    user_input: str
    conversation_history: list[dict[str, str]]
    agent_version: Literal["v1", "v2"]
    raw_input: str
    intent: Intent
    route_reason: str
    rewritten_query: str
    documents: list[Document]
    memories: list[dict]
    draft: dict
    approved: bool
    review_comment: str
    memory_id: int
    answer: str
    result: str
    structured_result: dict
    needs_review: bool


_chunks: list[Document] | None = None
_vector_store: InMemoryVectorStore | None = None


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def langfuse_enabled() -> bool:
    enabled = os.getenv(
        "LANGFUSE_ENABLED",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}
    return bool(
        enabled
        and
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def build_run_config(thread_id: str) -> dict:
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "project": "product-memory-agent",
            "environment": "development",
        },
        "run_name": "product-memory-agent",
    }

    if langfuse_enabled():
        config["callbacks"] = [CallbackHandler()]

    return config


def flush_langfuse() -> None:
    if langfuse_enabled():
        get_client().flush()


def get_retrieval_resources() -> tuple[
    list[Document],
    InMemoryVectorStore,
]:
    """首次需要检索时才解析文档并建立内存向量库。"""
    global _chunks, _vector_store

    if _chunks is None or _vector_store is None:
        _chunks = load_chunks()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-v4",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=(
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            check_embedding_ctx_length=False,
            chunk_size=10,
        )
        _vector_store = InMemoryVectorStore.from_documents(
            documents=_chunks,
            embedding=embeddings,
        )

    return _chunks, _vector_store


def format_conversation_history(
    history: list[dict[str, str]],
) -> str:
    if not history:
        return "无"

    role_labels = {
        "user": "用户",
        "assistant": "助手",
    }
    return "\n".join(
        f"{role_labels.get(item['role'], item['role'])}："
        f"{item['content']}"
        for item in history
    )


def history_messages(
    history: list[dict[str, str]],
) -> list[tuple[str, str]]:
    role_map = {
        "user": "user",
        "assistant": "assistant",
    }
    return [
        (role_map[item["role"]], item["content"])
        for item in history
        if item.get("role") in role_map and item.get("content")
    ]


def requires_strict_evidence_gate(question: str) -> bool:
    """对容易产生范围替代或否定性推断的问题启用额外门控。"""
    high_risk_markers = (
        "入驻",
        "是否存在",
        "有没有",
        "是否需要",
        "需不需要",
        "无需",
        "不需要",
    )
    return any(marker in question for marker in high_risk_markers)


def character_bigrams(text: str) -> set[str]:
    normalized = "".join(
        character.lower()
        for character in text
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )
    return {
        normalized[index:index + 2]
        for index in range(len(normalized) - 1)
    }


def lexical_relevance(query: str, document: Document) -> float:
    query_bigrams = character_bigrams(query)
    if not query_bigrams:
        return 0.0
    searchable_text = (
        f"{document.metadata.get('section', '')} "
        f"{document.page_content}"
    )
    return (
        len(query_bigrams & character_bigrams(searchable_text))
        / len(query_bigrams)
    )


def route_intent(state: AgentState) -> dict:
    user_input = state["user_input"].strip()
    memory_commands = (
        "记录一条产品记忆",
        "记录产品记忆",
        "保存一条产品记忆",
        "保存产品记忆",
        "写入产品记忆",
        "新增产品记忆",
    )
    if any(command in user_input for command in memory_commands):
        return {
            "intent": "memory_write",
            "route_reason": "用户明确要求记录或保存产品记忆",
            "documents": [],
            "memories": [],
            "answer": "",
            "result": "",
            "structured_result": {},
            "needs_review": False,
            "rewritten_query": "",
        }

    router = get_model().with_structured_output(IntentDecision)
    decision = router.invoke(
        [
            (
                "system",
                "判断用户意图："
                "casual_chat 表示问候、自我介绍、感谢、闲聊、"
                "或不需要查询产品资料的一般交流；"
                "knowledge_query 表示查询历史需求、设计、实施状态或原因；"
                "conflict_check 表示提交一条新需求，检查与历史方案是否冲突；"
                "memory_write 表示要求记录、更新或保存一条产品事实、状态、约束或决策。"
                "判断代词和省略表达时，必须结合最近会话。"
                "仅返回结构化结果。",
            ),
            (
                "user",
                "最近会话：\n"
                f"{format_conversation_history(state.get('conversation_history', []))}"
                "\n\n当前输入：\n"
                f"{user_input}",
            ),
        ]
    )
    return {
        "intent": decision.intent,
        "route_reason": decision.reason,
        "documents": [],
        "memories": [],
        "answer": "",
        "result": "",
        "structured_result": {},
        "needs_review": False,
        "rewritten_query": "",
    }


def choose_intent(state: AgentState) -> str:
    return state["intent"]


def answer_casual_chat(state: AgentState) -> dict:
    messages = [
        (
            "system",
            "你是产品知识与决策记忆 Agent。"
            "你可以自然地进行问候、自我介绍和一般交流。"
            "你的核心能力是查询需求文档、结合已审批产品记忆回答、"
            "检查新需求冲突，以及通过人工审批记录产品记忆。"
            "普通对话不需要引用需求文档，也不要声称自己查询了资料。"
            "不要把需求设计误称为已上线功能。"
            "回答自然、简洁，不使用 emoji。",
        ),
        *history_messages(state.get("conversation_history", [])),
        ("user", state["user_input"]),
    ]
    response = get_model().invoke(messages)
    return {
        "answer": str(response.content),
        "needs_review": False,
        "structured_result": {
            "sufficient": True,
            "document_citations": [],
            "memory_citations": [],
        },
    }


def rewrite_question(state: AgentState) -> dict:
    history = state.get("conversation_history", [])
    if not history:
        return {"rewritten_query": state["user_input"]}

    rewriter = get_model().with_structured_output(RewrittenQuestion)
    result = rewriter.invoke(
        [
            (
                "system",
                "结合最近会话，把当前产品知识问题改写成完整、独立、"
                "可直接用于检索的问题。"
                "只补全“它、这个、那项功能”等指代对象，"
                "保留当前问题真正询问的方面。"
                "不得把助手上一轮回答中的结论、状态、日期或推断"
                "附加成当前问题的前提，除非用户本轮明确重复了该前提。"
                "例如，历史讨论“小憩状态”，当前问“那前台怎么展示？”，"
                "应改写为“需求文档中小憩状态在前台如何设计展示？”，"
                "不能改写为“尚未上线的小憩状态前台如何展示？”。"
                "不得添加会话中没有的事实。"
                "如果当前问题已经完整，保持原意即可。"
                "只返回结构化结果。",
            ),
            (
                "user",
                "最近会话：\n"
                f"{format_conversation_history(history)}"
                "\n\n当前问题：\n"
                f"{state['user_input']}",
            ),
        ]
    )
    return {"rewritten_query": result.question.strip()}


def retrieve_documents(state: AgentState) -> dict:
    chunks, vector_store = get_retrieval_resources()
    query = (
        state.get("rewritten_query")
        if state["intent"] == "knowledge_query"
        else state["user_input"]
    )
    dense_results = vector_store.similarity_search(
        query or state["user_input"],
        k=TOP_K,
    )
    lexical_results = sorted(
        chunks,
        key=lambda document: lexical_relevance(
            query or state["user_input"],
            document,
        ),
        reverse=True,
    )

    retrieved = []
    seen = set()
    top_lexical = lexical_results[0]
    if lexical_relevance(
        query or state["user_input"],
        top_lexical,
    ) >= 0.35:
        retrieved.append(top_lexical)
        seen.add(
            (
                top_lexical.metadata.get("source"),
                top_lexical.metadata.get("paragraph_start"),
            )
        )

    for document in dense_results:
        key = (
            document.metadata.get("source"),
            document.metadata.get("paragraph_start"),
        )
        if key not in seen:
            retrieved.append(document)
            seen.add(key)
        if len(retrieved) == TOP_K:
            break

    expanded = expand_with_neighbors(retrieved, chunks)
    return {"documents": expanded}


def load_relevant_memories(state: AgentState) -> dict:
    memories = load_active_memories()

    if state["intent"] == "knowledge_query":
        memories = select_memories(
            state.get("rewritten_query") or state["user_input"],
            memories,
        )

    return {"memories": memories}


def answer_knowledge(state: AgentState) -> dict:
    question = state.get("rewritten_query") or state["user_input"]
    gate_metadata: dict = {}
    document_text, document_map = build_document_evidence(
        state["documents"]
    )
    memory_text, memory_map = build_memory_evidence(
        state.get("memories", [])
    )

    if (
        state.get("agent_version") == "v2"
        and requires_strict_evidence_gate(question)
    ):
        evidence_gate = get_model().with_structured_output(
            EvidenceSufficiencyDecision
        )
        gate_result = evidence_gate.invoke(
            [
                (
                    "system",
                    "你是回答生成前的证据充分性门控。"
                    "只判断给定证据是否直接回答用户所问的主体和问题类型。"
                    "检索结果主题相关不等于可以回答。"
                    "“资料未提及”不能证明“不存在、无需执行或未发生”。"
                    "若用户询问某主体如何操作，证据必须直接描述该主体的操作流程；"
                    "其他主体的流程不能替代。"
                    "若用户询问前台、页面或界面如何展示，"
                    "证据中对用户可见控件、颜色、状态、字段或交互的直接描述"
                    "可以视为充分证据，不要求原文必须出现“前台”二字。"
                    "在本项目中，“坐席个人状态与配置面板”、"
                    "点击在线图标、手动切换状态等描述，"
                    "属于坐席操作端的前台交互证据。"
                    "如果“前台”可能指客户侧或坐席侧，"
                    "可以基于证据回答坐席侧设计并明确限定范围，"
                    "不要因为客户侧没有对应描述而整体拒答。"
                    "只要证据足以回答当前询问的部分，就应判定可回答；"
                    "不要因为它不能回答未被用户询问的其他部分而拒答。"
                    "当问题询问需求设计、操作方式或交互流程时，"
                    "文档中明确的步骤、按钮、页面状态和交互规则就是直接证据；"
                    "不需要额外存在实施状态或已上线证明。"
                    "例如，只有合作伙伴入驻流程而没有客户小程序入驻流程时，"
                    "不能据此回答客户无需入驻，应判定 directly_answerable=false。"
                    "文档设计与已审批实施状态可以互补回答同一功能的问题。"
                    "只返回结构化结果。\n\n"
                    f"需求文档证据：\n{document_text or '无'}\n\n"
                    f"显式产品记忆：\n{memory_text or '无'}",
                ),
                ("user", question),
            ]
        )
        gate_metadata = {
            "directly_answerable": gate_result.directly_answerable,
            "reason": gate_result.reason,
            "missing_information": gate_result.missing_information,
        }
        if not gate_result.directly_answerable:
            refusal = MemoryAwareAnswer(
                answer=(
                    "现有资料不足以直接回答该问题。"
                    f"{gate_result.missing_information or gate_result.reason}"
                    "请补充直接相关资料或进行人工确认。"
                ),
                sufficient=False,
                document_citations=[],
                memory_citations=[],
                conflicts=[],
                needs_review=True,
            )
            return {
                "answer": refusal.answer,
                "needs_review": True,
                "structured_result": {
                    **refusal.model_dump(),
                    "evidence_gate": gate_metadata,
                },
            }

    selected_prompt = (
        grounded_v2_prompt
        if state.get("agent_version") == "v2"
        else answer_prompt
    )
    chain = selected_prompt | get_model().with_structured_output(
        MemoryAwareAnswer
    )
    answer = chain.invoke(
        {
            "question": question,
            "document_evidence": document_text or "无",
            "memory_evidence": memory_text or "无",
        }
    )

    answer.document_citations = [
        citation
        for citation in answer.document_citations
        if citation in document_map
    ]
    answer.memory_citations = [
        citation
        for citation in answer.memory_citations
        if citation in memory_map
    ]

    if not answer.sufficient:
        answer.answer = (
            "现有需求文档和已审批产品记忆不足以回答该问题，"
            "需要补充直接相关的资料或人工确认。"
        )
        answer.needs_review = True

    structured_result = answer.model_dump()
    if gate_metadata:
        structured_result["evidence_gate"] = gate_metadata

    return {
        "answer": answer.answer,
        "needs_review": answer.needs_review,
        "structured_result": structured_result,
    }


def detect_conflict(state: AgentState) -> dict:
    document_text, document_map = (
        build_conflict_document_evidence(state["documents"])
    )
    memory_text, memory_map = build_conflict_memory_evidence(
        state.get("memories", [])
    )

    chain = conflict_prompt | get_model().with_structured_output(
        ConflictReport
    )
    report = chain.invoke(
        {
            "new_requirement": state["user_input"],
            "document_evidence": document_text or "无",
            "memory_evidence": memory_text or "无",
        }
    )

    valid_document_ids = set(document_map)
    valid_memory_ids = set(memory_map)
    for conflict in report.conflicts:
        conflict.document_citations = [
            citation
            for citation in conflict.document_citations
            if citation in valid_document_ids
        ]
        conflict.memory_citations = [
            citation
            for citation in conflict.memory_citations
            if citation in valid_memory_ids
        ]

    return {
        "answer": report.summary,
        "needs_review": report.needs_review,
        "structured_result": report.model_dump(),
    }


def prepare_memory_input(state: AgentState) -> dict:
    """将总图输入适配为记忆草稿节点需要的字段。"""
    return {"raw_input": state["user_input"]}


builder = StateGraph(AgentState)
builder.add_node("route_intent", route_intent)
builder.add_node("answer_casual_chat", answer_casual_chat)
builder.add_node("rewrite_question", rewrite_question)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("load_relevant_memories", load_relevant_memories)
builder.add_node("answer_knowledge", answer_knowledge)
builder.add_node("detect_conflict", detect_conflict)
builder.add_node("prepare_memory_input", prepare_memory_input)
builder.add_node("create_draft", create_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("save_memory", save_memory)

builder.add_edge(START, "route_intent")
builder.add_conditional_edges(
    "route_intent",
    choose_intent,
    {
        "casual_chat": "answer_casual_chat",
        "knowledge_query": "rewrite_question",
        "conflict_check": "retrieve_documents",
        "memory_write": "prepare_memory_input",
    },
)
builder.add_edge("answer_casual_chat", END)
builder.add_edge("rewrite_question", "retrieve_documents")
builder.add_edge("retrieve_documents", "load_relevant_memories")
builder.add_conditional_edges(
    "load_relevant_memories",
    choose_intent,
    {
        "knowledge_query": "answer_knowledge",
        "conflict_check": "detect_conflict",
    },
)
builder.add_edge("answer_knowledge", END)
builder.add_edge("detect_conflict", END)
builder.add_edge("prepare_memory_input", "create_draft")
builder.add_edge("create_draft", "review_draft")
builder.add_edge("review_draft", "save_memory")
builder.add_edge("save_memory", END)


def print_result(state: dict) -> None:
    print("\n--- Agent Result ---")
    print("意图：", state.get("intent"))
    print("路由理由：", state.get("route_reason"))

    if state.get("answer"):
        print("回答：", state["answer"])
        print("需要复核：", state.get("needs_review"))

    if state.get("result"):
        print(state["result"])

    if state.get("structured_result"):
        print("结构化结果：")
        print(state["structured_result"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="产品知识与决策记忆 Agent 总流程"
    )
    parser.add_argument(
        "--thread",
        default="product-agent-demo",
        help="工作流会话 ID",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="提交查询、新需求或记忆写入请求",
    )
    run_parser.add_argument("text")

    resume_parser = subparsers.add_parser(
        "resume",
        help="恢复等待审批的记忆写入流程",
    )
    decision_group = resume_parser.add_mutually_exclusive_group(
        required=True
    )
    decision_group.add_argument("--approve", action="store_true")
    decision_group.add_argument("--reject", action="store_true")
    resume_parser.add_argument("--comment", default="")

    subparsers.add_parser("status", help="查看当前工作流状态")
    args = parser.parse_args()

    initialize_memory_database()
    config = build_run_config(args.thread)

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        if args.command == "run":
            state = graph.invoke(
                {"user_input": args.text},
                config=config,
            )
            print_result(state)

        elif args.command == "resume":
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print("当前工作流没有等待人工审批的任务。")
            else:
                state = graph.invoke(
                    Command(
                        resume={
                            "approved": args.approve,
                            "comment": args.comment,
                        }
                    ),
                    config=config,
                )
                print_result(state)

        snapshot = graph.get_state(config)
        print("\n下一节点：", snapshot.next or "无，流程已结束")
        if snapshot.interrupts:
            print("等待人工审批：")
            for item in snapshot.interrupts:
                print(item.value)

    flush_langfuse()


if __name__ == "__main__":
    main()

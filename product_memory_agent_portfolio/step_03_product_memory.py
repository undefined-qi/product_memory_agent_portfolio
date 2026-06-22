import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).parent
MEMORY_DB = BASE_DIR / "product_memories.sqlite"
CHECKPOINT_DB = BASE_DIR / "step_03_checkpoints.sqlite"


MemoryType = Literal[
    "decision",
    "implementation_status",
    "constraint",
    "historical_fact",
]


class ProductMemoryDraft(BaseModel):
    memory_type: MemoryType = Field(
        description=(
            "记忆类型：产品决策、实施状态、产品约束或历史事实"
        )
    )
    subject: str = Field(description="这条记忆描述的对象或功能")
    statement: str = Field(
        description="忠实保留原始语义的事实或决策陈述"
    )
    reason: str | None = Field(
        default=None,
        description="明确给出的原因；输入未提供时必须为 null",
    )
    scope: str = Field(description="该记忆适用的产品、用户或业务范围")
    as_of_date: str | None = Field(
        default=None,
        description="事实或状态对应的日期，格式 YYYY-MM-DD",
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="该记忆所依据的文档名称",
    )


class MemoryState(TypedDict, total=False):
    raw_input: str
    draft: dict
    approved: bool
    review_comment: str
    memory_id: int
    result: str


def get_draft_model():
    """仅在生成草稿时初始化模型，本地 list/status 不依赖 API。"""
    model = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return model.with_structured_output(ProductMemoryDraft)


def initialize_memory_database() -> None:
    with sqlite3.connect(MEMORY_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS product_memories_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                statement TEXT NOT NULL,
                reason TEXT,
                scope TEXT NOT NULL,
                as_of_date TEXT,
                status TEXT NOT NULL,
                source_documents TEXT NOT NULL,
                review_comment TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def create_draft(state: MemoryState) -> dict:
    """把自然语言描述转换成待审批的结构化记忆草稿。"""
    draft = get_draft_model().invoke(
        [
            (
                "system",
                "你负责把产品信息整理成产品记忆草稿。"
                "不得补充输入中没有的事实。"
                "必须区分以下类型："
                "decision 表示团队明确作出的选择；"
                "implementation_status 表示截至某时点的开发或上线状态；"
                "constraint 表示必须遵守的产品或技术限制；"
                "historical_fact 表示已经发生的历史事实。"
                "“尚未开发或上线”只是 implementation_status，"
                "不得改写为“决定不开发”或“暂不开发”。"
                "statement 必须忠实保留时间、否定和范围等原始语义。"
                "输入未提供原因时 reason 必须为 null，禁止猜测原因。"
                "输入未提供日期时 as_of_date 必须为 null。",
            ),
            ("user", state["raw_input"]),
        ]
    )
    return {"draft": draft.model_dump()}


def review_draft(state: MemoryState) -> dict:
    """暂停流程，等待人工批准、拒绝或修改草稿。"""
    review = interrupt(
        {
            "question": "是否将这条草稿写入产品记忆库？",
            "draft": state["draft"],
        }
    )

    revised_draft = review.get("draft") or state["draft"]
    return {
        "approved": review["approved"],
        "review_comment": review.get("comment", ""),
        "draft": revised_draft,
    }


def save_memory(state: MemoryState) -> dict:
    """只保存经过人工批准的记忆。"""
    if not state["approved"]:
        return {
            "result": (
                "记忆草稿已被拒绝，未写入产品记忆库。"
                f"审批意见：{state.get('review_comment') or '未填写'}"
            )
        }

    draft = ProductMemoryDraft.model_validate(state["draft"])
    with sqlite3.connect(MEMORY_DB) as connection:
        cursor = connection.execute(
            """
            INSERT INTO product_memories_v2 (
                memory_type,
                subject,
                statement,
                reason,
                scope,
                as_of_date,
                status,
                source_documents,
                review_comment,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft.memory_type,
                draft.subject,
                draft.statement,
                draft.reason,
                draft.scope,
                draft.as_of_date,
                "active",
                json.dumps(
                    draft.source_documents,
                    ensure_ascii=False,
                ),
                state.get("review_comment", ""),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        memory_id = cursor.lastrowid

    return {
        "memory_id": memory_id,
        "result": f"产品记忆已保存，ID={memory_id}。",
    }


builder = StateGraph(MemoryState)
builder.add_node("create_draft", create_draft)
builder.add_node("review_draft", review_draft)
builder.add_node("save_memory", save_memory)
builder.add_edge(START, "create_draft")
builder.add_edge("create_draft", "review_draft")
builder.add_edge("review_draft", "save_memory")
builder.add_edge("save_memory", END)


def list_memories() -> None:
    initialize_memory_database()
    with sqlite3.connect(MEMORY_DB) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                memory_type,
                subject,
                statement,
                reason,
                scope,
                as_of_date,
                status,
                source_documents,
                review_comment,
                created_at
            FROM product_memories_v2
            ORDER BY id
            """
        ).fetchall()

    if not rows:
        print("产品记忆库为空。")
        return

    for row in rows:
        print(f"\n--- Memory {row[0]} ---")
        print("类型：", row[1])
        print("主体：", row[2])
        print("陈述：", row[3])
        print("原因：", row[4] or "未提供")
        print("范围：", row[5])
        print("截至日期：", row[6] or "未提供")
        print("状态：", row[7])
        print("来源：", json.loads(row[8]))
        print("审批意见：", row[9] or "无")
        print("创建时间：", row[10])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="显式产品记忆：生成草稿、人工审批、持久化"
    )
    parser.add_argument(
        "--thread",
        default="memory-demo",
        help="记忆审批流程 ID",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    propose_parser = subparsers.add_parser(
        "propose",
        help="根据自然语言生成记忆草稿",
    )
    propose_parser.add_argument(
        "text",
        help="产品决策描述",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="提交人工审批结果",
    )
    decision_group = resume_parser.add_mutually_exclusive_group(
        required=True
    )
    decision_group.add_argument(
        "--approve",
        action="store_true",
        help="批准写入",
    )
    decision_group.add_argument(
        "--reject",
        action="store_true",
        help="拒绝写入",
    )
    resume_parser.add_argument(
        "--comment",
        default="",
        help="审批意见",
    )

    subparsers.add_parser("status", help="查看审批流程状态")
    subparsers.add_parser("list", help="查看已批准的产品记忆")
    args = parser.parse_args()

    initialize_memory_database()

    if args.command == "list":
        list_memories()
        return

    config = {
        "configurable": {
            "thread_id": args.thread,
        }
    }

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        if args.command == "propose":
            output = graph.invoke(
                {"raw_input": args.text},
                config=config,
            )
            print("--- 草稿已生成，等待人工审批 ---")
            print(output)

        elif args.command == "resume":
            snapshot_before = graph.get_state(config)
            if not snapshot_before.next:
                print("当前流程没有等待审批的草稿。")
            else:
                output = graph.invoke(
                    Command(
                        resume={
                            "approved": args.approve,
                            "comment": args.comment,
                        }
                    ),
                    config=config,
                )
                print(output["result"])

        snapshot = graph.get_state(config)
        print("\n--- 当前审批状态 ---")
        print(snapshot.values)
        print("下一节点：", snapshot.next or "无，流程已结束")

        if snapshot.interrupts:
            print("等待审批：")
            for item in snapshot.interrupts:
                print(item.value)


if __name__ == "__main__":
    main()

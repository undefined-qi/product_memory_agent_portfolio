import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from step_01_retrieval import expand_with_neighbors, load_chunks


BASE_DIR = Path(__file__).parent
MEMORY_DB = BASE_DIR / "product_memories.sqlite"
TOP_K = 5


ConflictType = Literal[
    "none",
    "direct_contradiction",
    "status_mismatch",
    "scope_overlap",
    "duplicate_requirement",
    "insufficient_evidence",
]


class ConflictItem(BaseModel):
    conflict_type: ConflictType
    description: str = Field(
        description="新需求与历史资料之间的具体关系"
    )
    document_citations: list[str] = Field(default_factory=list)
    memory_citations: list[str] = Field(default_factory=list)


class ConflictReport(BaseModel):
    has_conflict: bool
    summary: str
    conflicts: list[ConflictItem] = Field(default_factory=list)
    impact: list[str] = Field(
        default_factory=list,
        description="如果继续推进该需求，可能影响的已有设计或状态",
    )
    clarification_questions: list[str] = Field(
        default_factory=list,
        description="在需求评审前必须向提出方确认的问题",
    )
    needs_review: bool


def load_active_memories() -> list[dict]:
    if not MEMORY_DB.exists():
        return []

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
                source_documents,
                review_comment
            FROM product_memories_v2
            WHERE status = 'active'
            ORDER BY
                COALESCE(as_of_date, created_at) DESC,
                id DESC
            """
        ).fetchall()

    return [
        {
            "id": row[0],
            "memory_type": row[1],
            "subject": row[2],
            "statement": row[3],
            "reason": row[4],
            "scope": row[5],
            "as_of_date": row[6],
            "source_documents": json.loads(row[7]),
            "review_comment": row[8],
        }
        for row in rows
    ]


def build_document_evidence(
    documents: list[Document],
) -> tuple[str, dict[str, Document]]:
    evidence_map = {}
    blocks = []

    for index, document in enumerate(documents, start=1):
        evidence_id = f"D{index}"
        evidence_map[evidence_id] = document
        metadata = document.metadata
        blocks.append(
            f"[{evidence_id}]\n"
            f"文档：{metadata['source']}\n"
            f"章节：{metadata['section']}\n"
            f"段落：{metadata['paragraph_start']}-"
            f"{metadata['paragraph_end']}\n"
            f"内容：\n{document.page_content}"
        )

    return "\n\n".join(blocks), evidence_map


def build_memory_evidence(
    memories: list[dict],
) -> tuple[str, dict[str, dict]]:
    evidence_map = {}
    blocks = []

    for index, memory in enumerate(memories, start=1):
        evidence_id = f"M{index}"
        evidence_map[evidence_id] = memory
        blocks.append(
            f"[{evidence_id}]\n"
            f"类型：{memory['memory_type']}\n"
            f"主体：{memory['subject']}\n"
            f"陈述：{memory['statement']}\n"
            f"原因：{memory['reason'] or '未提供'}\n"
            f"范围：{memory['scope']}\n"
            f"截至日期：{memory['as_of_date'] or '未提供'}\n"
            f"来源：{memory['source_documents']}\n"
            f"人工审批：{memory['review_comment'] or '无'}"
        )

    return "\n\n".join(blocks), evidence_map


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是产品需求冲突审查助手。"
            "任务是判断新需求与历史需求文档、已审批产品记忆之间的关系。"
            "只能根据提供的证据判断，不得使用自身知识补充事实。"
            "冲突类型定义："
            "direct_contradiction 表示两项要求无法同时成立；"
            "status_mismatch 表示新需求假设的实施状态与已确认状态不一致；"
            "scope_overlap 表示范围重叠，可能需要统一规则；"
            "duplicate_requirement 表示历史资料已经存在相同或高度相似需求；"
            "insufficient_evidence 表示证据不足，无法完成判断；"
            "none 表示未发现冲突。"
            "“历史文档规划过，但当前尚未上线”不是直接矛盾；"
            "如果新需求要继续推进该能力，可判断为重复需求或状态关联，"
            "并要求确认是恢复旧需求、调整旧方案还是新增版本。"
            "引用只能使用给定的 D 或 M 编号。"
            "存在任何冲突、状态不一致或证据不足时 needs_review=true。\n\n"
            "历史文档证据：\n{document_evidence}\n\n"
            "已审批产品记忆：\n{memory_evidence}",
        ),
        (
            "user",
            "待审查的新需求：\n{new_requirement}",
        ),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查新需求与历史文档、产品记忆之间的冲突"
    )
    parser.add_argument(
        "requirement",
        nargs="?",
        default=(
            "本期上线坐席小憩状态。坐席进入小憩后继续分配新会话，"
            "但界面显示为黄色。"
        ),
    )
    args = parser.parse_args()

    chunks = load_chunks()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
        chunk_size=10,
    )
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retrieved = vector_store.similarity_search(
        args.requirement,
        k=TOP_K,
    )
    expanded = expand_with_neighbors(retrieved, chunks)
    document_text, document_map = build_document_evidence(expanded)

    memories = load_active_memories()
    memory_text, memory_map = build_memory_evidence(memories)

    model = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    chain = prompt | model.with_structured_output(ConflictReport)
    report = chain.invoke(
        {
            "new_requirement": args.requirement,
            "document_evidence": document_text or "无",
            "memory_evidence": memory_text or "无",
        }
    )

    valid_document_ids = set(document_map)
    valid_memory_ids = set(memory_map)
    invalid_citations = []

    for conflict in report.conflicts:
        invalid_citations.extend(
            citation
            for citation in conflict.document_citations
            if citation not in valid_document_ids
        )
        invalid_citations.extend(
            citation
            for citation in conflict.memory_citations
            if citation not in valid_memory_ids
        )

    print("--- 冲突检测报告 ---")
    print("新需求：", args.requirement)
    print("是否冲突：", report.has_conflict)
    print("需要复核：", report.needs_review)
    print("摘要：", report.summary)

    for index, conflict in enumerate(report.conflicts, start=1):
        print(f"\n冲突 {index}：{conflict.conflict_type}")
        print(conflict.description)
        print("文档引用：", conflict.document_citations)
        print("记忆引用：", conflict.memory_citations)

    if report.impact:
        print("\n潜在影响：")
        for item in report.impact:
            print("-", item)

    if report.clarification_questions:
        print("\n待确认问题：")
        for item in report.clarification_questions:
            print("-", item)

    if invalid_citations:
        print("\n警告：存在无效引用：", sorted(set(invalid_citations)))


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from step_01_retrieval import expand_with_neighbors, load_chunks


BASE_DIR = Path(__file__).parent
MEMORY_DB = BASE_DIR / "product_memories.sqlite"
TOP_K = 5


class MemoryAwareAnswer(BaseModel):
    answer: str = Field(
        description="完整回答用户问题中的每一个子问题"
    )
    sufficient: bool = Field(description="现有资料是否足以回答问题")
    document_citations: list[str] = Field(
        default_factory=list,
        description="使用的文档证据编号，例如 D1",
    )
    memory_citations: list[str] = Field(
        default_factory=list,
        description="使用的显式记忆编号，例如 M1",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "文档与显式记忆之间真正互相矛盾的陈述；"
            "设计与实施状态互补时不要填写"
        ),
    )
    needs_review: bool = Field(
        description="是否需要人工复核"
    )


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
                review_comment,
                created_at
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
            "created_at": row[9],
        }
        for row in rows
    ]


def select_memories(question: str, memories: list[dict]) -> list[dict]:
    """第一版使用主体和范围的确定性关键词匹配。"""
    normalized_question = "".join(question.lower().split())
    matched = []

    for memory in memories:
        subject = "".join(memory["subject"].lower().split())
        scope = "".join(memory["scope"].lower().split())
        if subject in normalized_question or any(
            term in normalized_question
            for term in subject.replace("功能", "").split()
            if term
        ):
            matched.append(memory)
        elif subject and subject[:2] in normalized_question:
            matched.append(memory)
        elif scope and scope[:4] in normalized_question:
            matched.append(memory)

    return matched


def build_document_evidence(
    documents: list[Document],
) -> tuple[str, dict[str, Document]]:
    evidence_map: dict[str, Document] = {}
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
    evidence_map: dict[str, dict] = {}
    blocks = []

    for index, memory in enumerate(memories, start=1):
        evidence_id = f"M{index}"
        evidence_map[evidence_id] = memory
        blocks.append(
            f"[{evidence_id}]\n"
            f"记忆 ID：{memory['id']}\n"
            f"类型：{memory['memory_type']}\n"
            f"主体：{memory['subject']}\n"
            f"陈述：{memory['statement']}\n"
            f"原因：{memory['reason'] or '未提供'}\n"
            f"范围：{memory['scope']}\n"
            f"截至日期：{memory['as_of_date'] or '未提供'}\n"
            f"来源文档：{memory['source_documents']}\n"
            f"人工审批：{memory['review_comment'] or '无'}"
        )

    return "\n\n".join(blocks), evidence_map


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是产品知识与决策记忆助手。"
            "只能根据需求文档证据和已审批的显式产品记忆回答。"
            "需求文档用于说明规划、设计和历史背景，"
            "不能单独证明功能已经开发或上线。"
            "显式记忆用于说明经人工确认的当前状态、约束或决策。"
            "当前实施状态冲突时，优先采用日期更新且状态为 active 的显式记忆，"
            "同时说明文档设计与当前状态的差异。"
            "文档描述设计、显式记忆描述实施状态时，二者是互补证据，"
            "不是资料不足，也不一定是真正冲突。"
            "如果文档足以说明设计，显式记忆足以说明当前状态，"
            "则应设置 sufficient=true，并综合回答两部分。"
            "必须逐项回答用户问题中的所有子问题。"
            "例如用户同时询问是否完成和如何设计，answer 中必须同时包含"
            "当前实施状态、截至日期和主要设计内容，不能把设计内容只放在 conflicts。"
            "只有问题中的关键部分在两类证据中都没有依据时，"
            "才设置 sufficient=false。"
            "conflicts 只记录真正无法同时成立的陈述。"
            "“文档已规划、显式记忆确认尚未上线”属于阶段信息互补，"
            "不应写入 conflicts。"
            "不得将“已设计”表述为“已上线”。"
            "引用只能使用给定的 D 或 M 编号。"
            "证据不足时设置 sufficient=false 和 needs_review=true。\n\n"
            "需求文档证据：\n{document_evidence}\n\n"
            "显式产品记忆：\n{memory_evidence}",
        ),
        ("user", "{question}"),
    ]
)

grounded_v2_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是产品知识与决策记忆助手。"
            "只能根据需求文档证据和已审批的显式产品记忆回答。"
            "需求文档用于说明规划、设计和历史背景，"
            "不能单独证明功能已经开发或上线。"
            "显式记忆用于说明经人工确认的当前状态、约束或决策。"
            "当前实施状态冲突时，优先采用日期更新且状态为 active 的显式记忆，"
            "同时说明文档设计与当前状态的差异。"
            "文档描述设计、显式记忆描述实施状态时，二者是互补证据，"
            "不是资料不足，也不一定是真正冲突。"
            "如果文档足以说明设计，显式记忆足以说明当前状态，"
            "则应设置 sufficient=true，并综合回答两部分。"
            "必须逐项回答用户问题中的所有子问题。"
            "例如用户同时询问是否完成和如何设计，answer 中必须同时包含"
            "当前实施状态、截至日期和主要设计内容，不能把设计内容只放在 conflicts。"
            "不得将“已设计”表述为“已上线”。"
            "\n\n严格证据规则："
            "\n1. 检索结果与问题词义相近，不代表它能回答问题。"
            "\n2. “证据没有提及某流程”不等于“该流程不存在或无需执行”。"
            "\n3. 只有证据直接陈述否定结论时，才能回答“不需要、不存在、未发生”。"
            "\n4. 必须核对主体、对象和业务范围；"
            "不得用合作伙伴、服务商或工程师的流程替代客户小程序的流程。"
            "\n5. 用户询问“如何操作、为什么、有哪些能力”时，"
            "必须存在直接说明对应流程、原因或能力的证据。"
            "如果证据已经明确描述页面、按钮、步骤、状态或交互规则，"
            "则足以回答需求设计和使用方式，应设置 sufficient=true。"
            "用户只询问设计或使用方式时，不得因为缺少实施状态或上线证明而拒答；"
            "只需避免把设计表述成已上线。"
            "\n6. 用户询问前台、页面或界面如何展示时，"
            "证据中对用户可见控件、颜色、状态、字段或交互的直接描述"
            "属于充分证据，不要求原文必须出现“前台”二字。"
            "“坐席个人状态与配置面板”、点击在线图标、"
            "手动切换状态属于坐席操作端的前台交互证据。"
            "如果“前台”可能指客户侧或坐席侧，"
            "应明确限定为证据能够支持的坐席侧，不得扩展到客户侧。"
            "\n7. 如果关键结论只能靠常识、猜测或从其他主体类推，"
            "设置 sufficient=false、needs_review=true，并说明缺失信息。"
            "\n8. 仅引用真正支撑 answer 中结论的 D 或 M 编号。"
            "\n\nconflicts 只记录真正无法同时成立的陈述。"
            "“文档已规划、显式记忆确认尚未上线”属于阶段信息互补，"
            "不应写入 conflicts。"
            "证据不足时设置 sufficient=false 和 needs_review=true。\n\n"
            "需求文档证据：\n{document_evidence}\n\n"
            "显式产品记忆：\n{memory_evidence}",
        ),
        ("user", "{question}"),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="联合需求文档与显式产品记忆回答问题"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="小憩状态做了吗？具体是怎么设计的？",
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

    retrieved_documents = vector_store.similarity_search(
        args.question,
        k=TOP_K,
    )
    expanded_documents = expand_with_neighbors(
        retrieved_documents,
        chunks,
    )

    memories = select_memories(
        args.question,
        load_active_memories(),
    )
    document_text, document_map = build_document_evidence(
        expanded_documents
    )
    memory_text, memory_map = build_memory_evidence(memories)

    model = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    answer_chain = prompt | model.with_structured_output(
        MemoryAwareAnswer
    )
    result = answer_chain.invoke(
        {
            "question": args.question,
            "document_evidence": document_text or "无",
            "memory_evidence": memory_text or "无",
        }
    )

    valid_document_citations = [
        citation
        for citation in result.document_citations
        if citation in document_map
    ]
    valid_memory_citations = [
        citation
        for citation in result.memory_citations
        if citation in memory_map
    ]

    if not result.sufficient:
        result.answer = (
            "现有需求文档和已审批产品记忆不足以回答该问题，"
            "需要补充直接相关的资料或人工确认。"
        )
        result.needs_review = True

    print("--- 联合回答 ---")
    print(result.answer)
    print("资料充分：", result.sufficient)
    print("需要复核：", result.needs_review)

    if result.conflicts:
        print("状态差异或冲突：")
        for conflict in result.conflicts:
            print("-", conflict)

    print("文档引用：")
    for citation in valid_document_citations:
        metadata = document_map[citation].metadata
        print(
            f"- {citation} | {metadata['source']} | "
            f"{metadata['section']} | "
            f"段落 {metadata['paragraph_start']}-"
            f"{metadata['paragraph_end']}"
        )

    print("记忆引用：")
    for citation in valid_memory_citations:
        memory = memory_map[citation]
        print(
            f"- {citation} | Memory {memory['id']} | "
            f"{memory['memory_type']} | "
            f"截至 {memory['as_of_date'] or '未知'}"
        )


if __name__ == "__main__":
    main()

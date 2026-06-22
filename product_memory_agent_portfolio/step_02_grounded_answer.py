import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from step_01_retrieval import (
    EVALUATION_CASES,
    expand_with_neighbors,
    load_chunks,
)


TOP_K = 5


class GroundedAnswer(BaseModel):
    answer: str = Field(description="仅根据证据生成的最终回答")
    sufficient: bool = Field(description="现有证据是否足以回答问题")
    missing_information: list[str] = Field(
        default_factory=list,
        description="证据不足时仍然缺少的信息",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="支持答案的证据编号，例如 E1、E2",
    )


def build_evidence_context(documents) -> tuple[str, dict[str, object]]:
    evidence_map: dict[str, object] = {}
    blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        evidence_id = f"E{index}"
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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是产品知识与决策记忆助手。"
            "只能根据提供的证据回答，禁止使用自身知识补充事实。"
            "每个结论必须由证据直接支持。"
            "需求文档中出现某项设计，不代表该功能已经开发或上线。"
            "如果证据不足，应设置 sufficient=false，明确缺少什么，"
            "并且只能说明无法从现有证据确定。"
            "不得因为资料未提及某个流程，就推断该流程不存在或无需执行。"
            "不得用相似但不同的流程或不同业务主体的流程代替答案。"
            "citations 只能填写实际使用的证据编号。\n\n"
            "证据：\n{evidence}",
        ),
        ("user", "{question}"),
    ]
)


def main() -> None:
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

    model = ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    answer_model = model.with_structured_output(GroundedAnswer)
    answer_chain = prompt | answer_model

    for index, case in enumerate(EVALUATION_CASES, start=1):
        question = case["question"]
        retrieved = vector_store.similarity_search(
            question,
            k=TOP_K,
        )
        expanded = expand_with_neighbors(retrieved, chunks)
        evidence_text, evidence_map = build_evidence_context(expanded)

        result = answer_chain.invoke(
            {
                "question": question,
                "evidence": evidence_text,
            }
        )

        valid_citations = [
            citation
            for citation in result.citations
            if citation in evidence_map
        ]
        invalid_citations = sorted(
            set(result.citations) - set(valid_citations)
        )

        # 资料不足时使用确定性失败话术，避免模型继续作无依据推断。
        if not result.sufficient:
            result.answer = (
                "现有资料不足以回答该问题，"
                "请补充与该问题直接相关的需求文档或产品决策记录。"
            )
            valid_citations = []

        print(f"\n{'=' * 70}")
        print(f"问题 {index}：{question}")
        print(f"资料充分：{result.sufficient}")
        print(f"回答：{result.answer}")

        if result.missing_information:
            print(
                "缺失信息："
                + "；".join(result.missing_information)
            )

        print("引用：")
        for citation in valid_citations:
            document = evidence_map[citation]
            metadata = document.metadata
            print(
                f"- {citation} | {metadata['source']} | "
                f"{metadata['section']} | "
                f"段落 {metadata['paragraph_start']}-"
                f"{metadata['paragraph_end']}"
            )

        if invalid_citations:
            print(f"警告：模型生成了无效引用 {invalid_citations}")


if __name__ == "__main__":
    main()

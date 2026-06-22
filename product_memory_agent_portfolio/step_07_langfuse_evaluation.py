import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 必须先加载环境变量，再初始化 Langfuse 或导入总 Agent。
load_dotenv(Path(__file__).parent / ".env")

from langfuse import get_client
from langfuse.experiment import Evaluation
from langfuse.langchain import CallbackHandler

from step_06_product_memory_agent import builder
from evaluation_dataset_v2 import (
    EVALUATION_ITEMS,
    items_for_split,
)


DATASET_NAME = "product-memory-agent-eval-v2"


def sync_dataset(client) -> None:
    """将本地固定测试集同步到 Langfuse Dataset。"""
    try:
        client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "产品知识与决策记忆 Agent 扩展评测集："
                "16 条开发集 + 9 条保留测试集"
            ),
            metadata={
                "version": "v2",
                "item_count": len(EVALUATION_ITEMS),
                "dev_count": len(items_for_split("dev")),
                "holdout_count": len(items_for_split("holdout")),
            },
        )

    for item in EVALUATION_ITEMS:
        stable_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{DATASET_NAME}:{item['id']}",
            )
        )
        client.create_dataset_item(
            id=stable_id,
            dataset_name=DATASET_NAME,
            input=item["input"],
            expected_output=item["expected_output"],
            metadata={
                "local_id": item["id"],
                "category": item["category"],
                "split": item["split"],
            },
        )

    client.flush()


graph = builder.compile()


class KeyPointJudgeResult(BaseModel):
    covered_indices: list[int] = Field(
        description="回答已覆盖的标准要点序号，从 1 开始"
    )
    reason: str = Field(description="简短判分理由")


class GroundednessJudgeResult(BaseModel):
    supported_claims: list[str] = Field(
        description="回答中可由证据支持的主要事实陈述"
    )
    unsupported_claims: list[str] = Field(
        description="回答中无法由证据支持的主要事实陈述"
    )
    score: float = Field(
        ge=0,
        le=1,
        description="有证据支持的主要事实陈述占比",
    )
    reason: str = Field(description="简短判分理由")


key_point_judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是严格的答案要点覆盖率评审员。"
            "逐条判断 Agent 回答是否在语义上明确覆盖标准要点。"
            "不能因为主题相关就算覆盖，也不能补充回答中没有的信息。"
            "同义表达可以算覆盖。"
            "covered_indices 只填写明确覆盖的要点序号，序号从 1 开始。",
        ),
        (
            "user",
            "问题：{question}\n\n"
            "标准要点：\n{key_points}\n\n"
            "Agent 回答：\n{answer}",
        ),
    ]
)


groundedness_judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是严格的事实依据评审员。"
            "只根据提供的检索证据判断 Agent 回答中的主要事实陈述是否有依据。"
            "不得使用常识或外部知识替 Agent 补充依据。"
            "“证据未提及”不能支持“不存在、无需执行、未发生”等否定结论。"
            "主体必须一致，不能用合作伙伴流程支持客户流程。"
            "如果回答只是谨慎说明资料不足，且没有额外事实断言，可得 1 分。"
            "score 为有证据支持的主要事实陈述占比；"
            "没有事实陈述时，根据回答是否恰当表达证据不足给出 0 或 1。",
        ),
        (
            "user",
            "问题：{question}\n\n"
            "检索证据：\n{evidence}\n\n"
            "Agent 回答：\n{answer}",
        ),
    ]
)


def get_judge_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        temperature=0,
    )


def invoke_with_retry(runnable, payload):
    last_error = None
    for attempt in range(1, 4):
        try:
            return runnable.invoke(payload)
        except Exception as error:
            last_error = error
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
    raise last_error


def build_experiment_task(
    agent_version: Literal["v1", "v2"],
):
    def run_agent(*, item, **kwargs):
        """对一个 DatasetItem 运行指定版本的 Agent。"""
        question = item.input["question"]
        conversation_history = item.input.get("history", [])
        last_error = None
        for attempt in range(1, 4):
            try:
                state = graph.invoke(
                    {
                        "user_input": question,
                        "conversation_history": conversation_history,
                        "agent_version": agent_version,
                    },
                    config={
                        "callbacks": [CallbackHandler()],
                        "run_name": (
                            f"product-memory-agent-eval-{agent_version}"
                        ),
                        "metadata": {
                            "dataset": DATASET_NAME,
                            "question": question,
                            "split": item.metadata.get("split"),
                            "category": item.metadata.get("category"),
                            "agent_version": agent_version,
                            "attempt": attempt,
                        },
                    },
                )
                break
            except Exception as error:
                last_error = error
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)
        else:
            raise last_error

        documents = state.get("documents", [])
        retrieved_sources = sorted(
            {
                document.metadata["source"]
                for document in documents
            }
        )
        document_evidence = [
            {
                "source": document.metadata.get("source"),
                "section": document.metadata.get("section"),
                "content": document.page_content,
            }
            for document in documents
        ]
        memory_evidence = [
            {
                "memory_type": memory.get("memory_type"),
                "subject": memory.get("subject"),
                "statement": memory.get("statement"),
                "scope": memory.get("scope"),
                "as_of_date": memory.get("as_of_date"),
            }
            for memory in state.get("memories", [])
        ]
        structured = state.get("structured_result", {})

        return {
            "answer": state.get("answer", ""),
            "intent": state.get("intent"),
            "rewritten_query": state.get("rewritten_query"),
            "sufficient": structured.get("sufficient"),
            "needs_review": state.get("needs_review", False),
            "document_citations": structured.get(
                "document_citations",
                [],
            ),
            "memory_citations": structured.get(
                "memory_citations",
                [],
            ),
            "retrieved_sources": retrieved_sources,
            "document_evidence": document_evidence,
            "memory_evidence": memory_evidence,
        }

    return run_agent


def run_agent(*, item, **kwargs):
    """Langfuse Experiment 对每个 DatasetItem 调用的任务函数。"""
    return build_experiment_task("v1")(item=item, **kwargs)


def refusal_evaluator(
    *,
    output,
    expected_output,
    **kwargs,
):
    if not expected_output.get("evaluate_refusal", True):
        return None
    should_refuse = expected_output["should_refuse"]
    actually_refused = output.get("sufficient") is False
    passed = should_refuse == actually_refused
    return Evaluation(
        name="refusal_accuracy",
        value=1.0 if passed else 0.0,
        comment=(
            f"expected_refuse={should_refuse}, "
            f"actual_refuse={actually_refused}"
        ),
    )


def review_evaluator(
    *,
    output,
    expected_output,
    **kwargs,
):
    expected = expected_output["should_review"]
    actual = bool(output.get("needs_review"))
    passed = expected == actual
    return Evaluation(
        name="review_accuracy",
        value=1.0 if passed else 0.0,
        comment=(
            f"expected_review={expected}, "
            f"actual_review={actual}"
        ),
    )


def source_recall_evaluator(
    *,
    output,
    expected_output,
    **kwargs,
):
    expected_sources = set(
        expected_output.get("expected_sources", [])
    )
    retrieved_sources = set(
        output.get("retrieved_sources", [])
    )

    if not expected_sources:
        return None

    matched = expected_sources & retrieved_sources
    score = len(matched) / len(expected_sources)
    return Evaluation(
        name="source_recall",
        value=score,
        comment=(
            f"matched={sorted(matched)}, "
            f"expected={sorted(expected_sources)}"
        ),
    )


def key_point_coverage_evaluator(
    *,
    input,
    output,
    expected_output,
    **kwargs,
):
    key_points = expected_output.get("key_points", [])
    should_refuse = expected_output.get("should_refuse", False)
    actually_refused = output.get("sufficient") is False

    if should_refuse and not actually_refused:
        return Evaluation(
            name="key_point_coverage",
            value=0.0,
            comment=(
                "确定性护栏：标准答案要求拒答，但 Agent 未拒答；"
                "不能视为覆盖“资料不足”类要点"
            ),
        )

    judge = (
        key_point_judge_prompt
        | get_judge_model().with_structured_output(
            KeyPointJudgeResult
        )
    )
    result = invoke_with_retry(
        judge,
        {
            "question": input["question"],
            "key_points": json.dumps(
                key_points,
                ensure_ascii=False,
                indent=2,
            ),
            "answer": output.get("answer", ""),
        },
    )

    # 防止 Judge 返回的 score 与它自己的覆盖列表不一致。
    covered_indices = sorted(
        {
            index
            for index in result.covered_indices
            if 1 <= index <= len(key_points)
        }
    )
    covered_points = [
        key_points[index - 1]
        for index in covered_indices
    ]
    missed_points = [
        point
        for index, point in enumerate(key_points, start=1)
        if index not in covered_indices
    ]
    score = (
        len(covered_indices) / len(key_points)
        if key_points
        else 1.0
    )
    return Evaluation(
        name="key_point_coverage",
        value=score,
        comment=(
            f"covered={covered_points}; "
            f"missed={missed_points}; "
            f"reason={result.reason}"
        ),
    )


def intent_accuracy_evaluator(
    *,
    output,
    expected_output,
    **kwargs,
):
    expected = expected_output["expected_intent"]
    actual = output.get("intent")
    return Evaluation(
        name="intent_accuracy",
        value=1.0 if expected == actual else 0.0,
        comment=f"expected_intent={expected}, actual_intent={actual}",
    )


def groundedness_evaluator(
    *,
    input,
    output,
    expected_output,
    **kwargs,
):
    if not expected_output.get("evaluate_groundedness", True):
        return None
    evidence = {
        "documents": output.get("document_evidence", []),
        "approved_memories": output.get("memory_evidence", []),
    }
    judge = (
        groundedness_judge_prompt
        | get_judge_model().with_structured_output(
            GroundednessJudgeResult
        )
    )
    result = invoke_with_retry(
        judge,
        {
            "question": input["question"],
            "evidence": json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2,
            ),
            "answer": output.get("answer", ""),
        },
    )
    claim_count = (
        len(result.supported_claims)
        + len(result.unsupported_claims)
    )
    score = (
        len(result.supported_claims) / claim_count
        if claim_count
        else result.score
    )
    return Evaluation(
        name="groundedness",
        value=score,
        comment=(
            f"supported={result.supported_claims}; "
            f"unsupported={result.unsupported_claims}; "
            f"reason={result.reason}"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="同步 Langfuse Dataset 并批量运行 Experiment"
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="只同步 Dataset，不运行实验",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Langfuse Dataset Run 名称",
    )
    parser.add_argument(
        "--agent-version",
        choices=["v1", "v2"],
        default="v2",
        help="v1 使用原 Prompt，v2 使用严格证据 Prompt",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "holdout", "all"],
        default="dev",
        help="dev 用于迭代；holdout 仅在版本冻结后运行",
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=[],
        help="只运行指定 local_id；可重复传入",
    )
    args = parser.parse_args()
    run_name = args.run_name or (
        f"{args.split}-baseline-v1"
        if args.agent_version == "v1"
        else f"{args.split}-current-v2"
    )

    client = get_client()
    if not client.auth_check():
        raise RuntimeError("Langfuse 认证失败，请检查 .env")

    sync_dataset(client)
    print(f"Dataset 已同步：{DATASET_NAME}")

    if args.sync_only:
        return

    dataset = client.get_dataset(DATASET_NAME)
    experiment_items = [
        item
        for item in dataset.items
        if (
            args.split == "all"
            or item.metadata.get("split") == args.split
        )
        and (
            not args.item_id
            or item.metadata.get("local_id") in args.item_id
        )
    ]
    if not experiment_items:
        raise RuntimeError("没有匹配到要运行的 Dataset Item")
    result = client.run_experiment(
        name="product-memory-agent-expanded-evaluation",
        run_name=run_name,
        description=(
            "同一 Dataset 和检索流程下，对比原始回答 Prompt 与严格证据 Prompt"
        ),
        data=experiment_items,
        task=build_experiment_task(args.agent_version),
        evaluators=[
            refusal_evaluator,
            review_evaluator,
            source_recall_evaluator,
            intent_accuracy_evaluator,
            key_point_coverage_evaluator,
            groundedness_evaluator,
        ],
        max_concurrency=1,
        metadata={
            "agent_version": args.agent_version,
            "dataset_split": args.split,
            "item_filter": args.item_id,
            "retrieval": "dense-top5-neighbor-expansion",
            "judge_model": "qwen-plus",
        },
    )
    client.flush()

    print("\n--- Experiment 完成 ---")
    print("Run：", result.run_name)
    print("Items：", len(result.item_results))
    if result.dataset_run_url:
        print("Langfuse：", result.dataset_run_url)


if __name__ == "__main__":
    main()

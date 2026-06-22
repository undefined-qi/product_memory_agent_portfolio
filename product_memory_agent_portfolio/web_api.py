import json
import sqlite3
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR
WEB_DIST = BASE_DIR / "web" / "dist"
WEB_DB = BASE_DIR / "web_app.sqlite"
CHECKPOINT_DB = BASE_DIR / "web_checkpoints.sqlite"
CONTEXT_MESSAGE_LIMIT = 6

load_dotenv(ROOT_DIR / ".env")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from step_03_product_memory import (  # noqa: E402
    MEMORY_DB,
    initialize_memory_database,
)
from step_06_product_memory_agent import (  # noqa: E402
    builder,
    build_run_config,
    flush_langfuse,
)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=10000)


class ReviewRequest(BaseModel):
    approved: bool
    comment: str = ""
    draft: dict[str, Any] | None = None


class ConversationRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def initialize_web_database() -> None:
    with sqlite3.connect(WEB_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
            """
        )


def create_conversation(title: str = "新对话") -> dict[str, str]:
    conversation_id = str(uuid.uuid4())
    created_at = now_iso()
    with sqlite3.connect(WEB_DB) as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, created_at, created_at),
        )
    return {
        "id": conversation_id,
        "title": title,
        "created_at": created_at,
        "updated_at": created_at,
    }


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> None:
    timestamp = now_iso()
    with sqlite3.connect(WEB_DB) as connection:
        connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, payload, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                json.dumps(payload, ensure_ascii=False) if payload else None,
                timestamp,
            ),
        )
        connection.execute(
            """
            UPDATE conversations SET updated_at = ? WHERE id = ?
            """,
            (timestamp, conversation_id),
        )


def load_recent_history(
    conversation_id: str,
    limit: int = CONTEXT_MESSAGE_LIMIT,
) -> list[dict[str, str]]:
    with sqlite3.connect(WEB_DB) as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    return [
        {"role": row[0], "content": row[1]}
        for row in reversed(rows)
    ]


def document_payload(documents: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"D{index}",
            "source": document.metadata.get("source", "未知文档"),
            "section": document.metadata.get("section", "未标注章节"),
            "paragraph_start": document.metadata.get("paragraph_start"),
            "paragraph_end": document.metadata.get("paragraph_end"),
            "content": document.page_content,
        }
        for index, document in enumerate(documents, start=1)
    ]


def memory_payload(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "citation_id": f"M{index}",
            **memory,
        }
        for index, memory in enumerate(memories, start=1)
    ]


def interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = state.get("__interrupt__", [])
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else {"value": value}


def serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    structured = state.get("structured_result") or {}
    pending_review = interrupt_payload(state)
    return {
        "answer": state.get("answer") or state.get("result") or (
            "已生成产品记忆草稿，请审批后写入。"
            if pending_review
            else "处理完成。"
        ),
        "intent": state.get("intent"),
        "route_reason": state.get("route_reason"),
        "rewritten_query": state.get("rewritten_query"),
        "needs_review": state.get("needs_review", bool(pending_review)),
        "structured_result": structured,
        "documents": document_payload(state.get("documents", [])),
        "memories": memory_payload(state.get("memories", [])),
        "pending_review": pending_review,
        "steps": build_steps(state, pending_review),
    }


def build_steps(
    state: dict[str, Any],
    pending_review: dict[str, Any] | None,
) -> list[dict[str, str]]:
    steps = []
    if state.get("intent"):
        steps.append(
            {
                "name": "识别意图",
                "status": "completed",
                "detail": state["intent"],
            }
        )
    if state.get("rewritten_query"):
        steps.append(
            {
                "name": "改写检索问题",
                "status": "completed",
                "detail": state["rewritten_query"],
            }
        )
    if state.get("intent") in {"knowledge_query", "conflict_check"}:
        steps.append(
            {
                "name": "检索需求文档",
                "status": "completed",
                "detail": f"获得 {len(state.get('documents', []))} 条证据",
            }
        )
    if state.get("intent") in {"knowledge_query", "conflict_check"}:
        steps.append(
            {
                "name": "读取产品记忆",
                "status": "completed",
                "detail": f"匹配 {len(state.get('memories', []))} 条记忆",
            }
        )
    if pending_review:
        steps.append(
            {
                "name": "等待人工审批",
                "status": "waiting",
                "detail": "产品记忆草稿已暂停",
            }
        )
    elif state.get("answer") or state.get("result"):
        steps.append(
            {
                "name": "生成结构化结果",
                "status": "completed",
                "detail": "流程执行完成",
            }
        )
    return steps


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_web_database()
    initialize_memory_database()
    saver_context = SqliteSaver.from_conn_string(str(CHECKPOINT_DB))
    checkpointer = saver_context.__enter__()
    app.state.graph = builder.compile(checkpointer=checkpointer)
    try:
        yield
    finally:
        flush_langfuse()
        saver_context.__exit__(None, None, None)


app = FastAPI(
    title="产品知识与决策记忆 Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/conversations")
def list_conversations() -> list[dict[str, Any]]:
    with sqlite3.connect(WEB_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/conversations")
def new_conversation() -> dict[str, str]:
    return create_conversation()


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str,
    request: ConversationRenameRequest,
) -> dict[str, str]:
    with sqlite3.connect(WEB_DB) as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (request.title.strip(), now_iso(), conversation_id),
        )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"id": conversation_id, "title": request.title.strip()}


@app.get("/api/conversations/{conversation_id}/messages")
def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(WEB_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, role, content, payload, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [
        {
            **dict(row),
            "payload": json.loads(row["payload"]) if row["payload"] else None,
        }
        for row in rows
    ]


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation = create_conversation(
            request.message.strip()[:28]
        )
        conversation_id = conversation["id"]
    else:
        with sqlite3.connect(WEB_DB) as connection:
            row = connection.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="对话不存在")
            if row[0] == "新对话":
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        request.message.strip()[:28],
                        now_iso(),
                        conversation_id,
                    ),
                )

    conversation_history = load_recent_history(conversation_id)
    save_message(conversation_id, "user", request.message.strip())
    config = build_run_config(conversation_id)
    config["metadata"]["surface"] = "web"
    try:
        state = app.state.graph.invoke(
            {
                "user_input": request.message.strip(),
                "conversation_history": conversation_history,
                "agent_version": "v2",
            },
            config=config,
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Agent 调用失败，请检查模型连接后重试",
        ) from error
    payload = serialize_state(state)
    save_message(
        conversation_id,
        "assistant",
        payload["answer"],
        payload,
    )
    flush_langfuse()
    return {
        "conversation_id": conversation_id,
        **payload,
    }


@app.post("/api/conversations/{conversation_id}/review")
def review_memory(
    conversation_id: str,
    request: ReviewRequest,
) -> dict[str, Any]:
    config = build_run_config(conversation_id)
    snapshot = app.state.graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="当前对话没有等待审批的任务",
        )

    try:
        state = app.state.graph.invoke(
            Command(
                resume={
                    "approved": request.approved,
                    "comment": request.comment,
                    "draft": request.draft,
                }
            ),
            config=config,
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="审批流程恢复失败，请稍后重试",
        ) from error
    payload = serialize_state(state)
    save_message(
        conversation_id,
        "assistant",
        payload["answer"],
        payload,
    )
    flush_langfuse()
    return {
        "conversation_id": conversation_id,
        **payload,
    }


@app.get("/api/memories")
def list_product_memories() -> list[dict[str, Any]]:
    initialize_memory_database()
    with sqlite3.connect(MEMORY_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                id, memory_type, subject, statement, reason, scope,
                as_of_date, status, source_documents, review_comment,
                created_at
            FROM product_memories_v2
            ORDER BY id DESC
            """
        ).fetchall()
    return [
        {
            **dict(row),
            "source_documents": json.loads(row["source_documents"]),
        }
        for row in rows
    ]


if WEB_DIST.exists():
    assets_dir = WEB_DIST / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        target = WEB_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from step_03_product_memory import MEMORY_DB, initialize_memory_database


def main() -> None:
    initialize_memory_database()
    with sqlite3.connect(MEMORY_DB) as connection:
        existing = connection.execute(
            """
            SELECT id FROM product_memories_v2
            WHERE subject = ? AND status = 'active'
            """,
            ("小憩状态",),
        ).fetchone()
        if existing:
            print(f"demo memory already exists: ID={existing[0]}")
            return

        cursor = connection.execute(
            """
            INSERT INTO product_memories_v2 (
                memory_type, subject, statement, reason, scope,
                as_of_date, status, source_documents,
                review_comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "implementation_status",
                "小憩状态",
                "小憩状态尚未开发或上线，目前只完成需求文档设计。",
                None,
                "星云服务平台坐席状态功能",
                "2026-06-20",
                "active",
                json.dumps(
                    [
                        "星云服务平台V2需求.docx",
                        "星云服务平台V3需求.docx",
                    ],
                    ensure_ascii=False,
                ),
                "合成演示数据",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        print(f"created demo memory: ID={cursor.lastrowid}")


if __name__ == "__main__":
    main()


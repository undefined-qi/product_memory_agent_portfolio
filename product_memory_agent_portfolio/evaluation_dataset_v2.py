"""产品记忆 Agent 扩展评测集。

dev 用于定位问题和迭代；holdout 只在版本冻结后运行，避免按测试题调 Prompt。
"""


def expected(
    *,
    intent: str = "knowledge_query",
    should_refuse: bool = False,
    should_review: bool = False,
    expected_sources: list[str] | None = None,
    key_points: list[str] | None = None,
    evaluate_groundedness: bool = True,
    evaluate_refusal: bool = True,
) -> dict:
    return {
        "expected_intent": intent,
        "should_refuse": should_refuse,
        "should_review": should_review,
        "expected_sources": expected_sources or [],
        "key_points": key_points or [],
        "evaluate_groundedness": evaluate_groundedness,
        "evaluate_refusal": evaluate_refusal,
    }


EVALUATION_ITEMS = [
    # -------------------- Dev: 已知核心问题 --------------------
    {
        "id": "dev-01",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "合作伙伴入驻有哪些注意事项？"},
        "expected_output": expected(
            expected_sources=[
                "星云服务平台合作伙伴入驻指南.docx"
            ],
            key_points=[
                "仅 伙伴平台 主账号角色和技术接口人角色具备操作权限",
                "添加机器人后预计等待 10 分钟完成初始化",
                "微信群名或个人微信名重名时无法绑定",
            ],
        ),
    },
    {
        "id": "dev-02",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "为什么要做 AI 智能回复？"},
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "渠道工程师专业能力存在差异",
                "帮助工程师自主闭环大多数 P3/P4 问题",
                "模拟专家指导并引导收集客户问题",
                "结合沟通上下文生成专业建议",
            ],
        ),
    },
    {
        "id": "dev-03",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "智能回复 Agent 需要具备哪些能力？"},
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "先总结分析会话并识别客户问题",
                "咨询类包含确认诉求、直接作答和开放式结尾",
                "配置类包含风险提示、步骤拆解和验证方法",
            ],
        ),
    },
    {
        "id": "dev-04",
        "split": "dev",
        "category": "knowledge",
        "input": {
            "question": "需求文档中是否规划了小憩状态？前后台分别如何设计？"
        },
        "expected_output": expected(
            expected_sources=[
                "星云服务平台V2需求.docx",
                "星云服务平台V3需求.docx",
            ],
            key_points=[
                "需求文档已规划小憩状态",
                "小憩期间不分配新会话",
                "坐席侧支持手动切换在线、小憩和离线",
                "后台展示状态人数、个人状态和持续时间",
                "需求设计不能证明已经开发或上线",
            ],
        ),
    },
    {
        "id": "dev-05",
        "split": "dev",
        "category": "refusal",
        "input": {"question": "客户小程序如何入驻？"},
        "expected_output": expected(
            should_refuse=True,
            should_review=True,
            key_points=[
                "现有资料没有客户小程序入驻流程",
                "不能用合作伙伴入驻流程替代客户流程",
                "需要补充直接相关资料或人工确认",
            ],
        ),
    },
    # -------------------- Dev: 文档事实扩展 --------------------
    {
        "id": "dev-06",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "合作伙伴完成入驻需要做哪两类关联？"},
        "expected_output": expected(
            expected_sources=[
                "星云服务平台合作伙伴入驻指南.docx"
            ],
            key_points=[
                "完成公司消息通知的微信群关联",
                "完成工程师个人微信与系统的关系关联",
                "公司群用于公共协同消息，个人微信用于责任人专属通知",
            ],
        ),
    },
    {
        "id": "dev-07",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "AI 智能回复在 Web 端是怎么交互的？"},
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "渠道端 Web 增加会话管理模块",
                "展示所有未解答问题并支持点击进入结果页",
                "问题支持更正后重新发送",
                "已回答问题在列表展示两行答案并可展开",
                "回答页底部支持继续提问",
            ],
        ),
    },
    {
        "id": "dev-08",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "移动端的 AI 智能辅助如何使用？"},
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "会话界面增加 AI 智能辅助按钮",
                "每次打开智能回复时刷新最新回复",
                "可查看未解答问题并更正问题",
                "结果页可编辑后发送或直接发送",
            ],
        ),
    },
    {
        "id": "dev-09",
        "split": "dev",
        "category": "knowledge",
        "input": {
            "question": "AI 智能回复规划了哪些体验、效能目标和核心观测指标？"
        },
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "结合上下文、产品知识库和历史相似工单生成专业回复",
                "关注首次响应时间和平均处理时长",
                "降低服务态度或表述不专业导致的客诉率",
                "AI 建议采纳率",
                "采纳后修改率",
            ],
        ),
    },
    {
        "id": "dev-10",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "强制下线坐席后，已有会话和新会话分别怎么处理？"},
        "expected_output": expected(
            expected_sources=[
                "星云服务平台V2需求.docx",
                "星云服务平台V3需求.docx",
            ],
            key_points=[
                "坐席状态立即变为离线",
                "不再分配新的会话",
                "已分配会话继续处理且不做流转",
                "被下线坐席页面显示提示",
            ],
        ),
    },
    {
        "id": "dev-11",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "把坐席最大并发从 7 调成 6 时会发生什么？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V3需求.docx"],
            key_points=[
                "页面可能暂时显示 7/6",
                "不会强制中断当前已接入会话",
                "暂停继续分配新会话",
                "当前并发回落到上限以内后恢复分配",
            ],
        ),
    },
    {
        "id": "dev-12",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "人工服务评价在什么情况下触发？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V2需求.docx"],
            key_points=[
                "结束服务或关闭会话时即时触发",
                "会话超过 24 小时无交互时自动关闭并静默触发",
                "渠道或原厂可点击邀请评价主动触发",
                "已经评价过的不能再次评价",
            ],
        ),
    },
    {
        "id": "dev-13",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "客户收到的服务评价卡片包含哪些内容？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V2需求.docx"],
            key_points=[
                "五星或满意度表情为必填项",
                "支持按星级选择快捷标签",
                "低于或等于 3 星时显示其他意见输入",
                "其他意见非必填且限制 200 字",
            ],
        ),
    },
    {
        "id": "dev-14",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "客户小程序的刷新按钮是怎么设计的？"},
        "expected_output": expected(
            expected_sources=[
                "星云服务平台V2需求.docx",
                "星云服务平台V3需求.docx",
            ],
            key_points=[
                "客户咨询和服务工单列表页增加刷新按钮",
                "点击刷新当前页",
                "默认位于右下角",
                "按钮可以拖动到任意位置",
            ],
        ),
    },
    {
        "id": "dev-15",
        "split": "dev",
        "category": "knowledge",
        "input": {"question": "客户小程序修改用户名时如何处理敏感词？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V3需求.docx"],
            key_points=[
                "修改用户名时进行敏感词过滤",
                "采用最简单的正则匹配",
                "词库位于需求不明确工作表",
            ],
        ),
    },
    {
        "id": "dev-16",
        "split": "dev",
        "category": "casual_chat",
        "input": {"question": "你好，请介绍一下你自己。"},
        "expected_output": expected(
            intent="casual_chat",
            key_points=[
                "说明自己是产品知识与决策记忆助手",
                "可以查询需求文档和产品记忆",
                "可以检查需求冲突并辅助记录产品记忆",
            ],
            evaluate_groundedness=False,
            evaluate_refusal=False,
        ),
    },
    # -------------------- Holdout: 版本冻结后再运行 --------------------
    {
        "id": "holdout-01",
        "split": "holdout",
        "category": "knowledge",
        "input": {"question": "服务质量告警一天推送几次，统计时间段怎么划分？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V2需求.docx"],
            key_points=[
                "每天推送两次",
                "上午 8:30 统计前一日 18:00 到当日 8:30",
                "下午 18:00 统计当日 8:30 到 18:00",
                "推送到渠道群",
            ],
        ),
    },
    {
        "id": "holdout-02",
        "split": "holdout",
        "category": "knowledge",
        "input": {"question": "服务异常监控支持排查哪些工单和会话异常？"},
        "expected_output": expected(
            expected_sources=["星云服务平台V2需求.docx"],
            key_points=[
                "工单支持超时未响应、即将逾期和已逾期",
                "会话支持响应超时、会话超时、满意度报警和情绪监控异常",
                "支持按客服名和企业名搜索",
                "可跳转到 工单系统 查看详情",
            ],
        ),
    },
    {
        "id": "holdout-03",
        "split": "holdout",
        "category": "multi_turn",
        "input": {
            "history": [
                {"role": "user", "content": "小憩状态做了吗？"},
                {
                    "role": "assistant",
                    "content": "截至 2026-06-20 尚未上线，但已有需求设计。",
                },
            ],
            "question": "那坐席侧怎么展示？",
        },
        "expected_output": expected(
            expected_sources=[
                "星云服务平台V2需求.docx",
                "星云服务平台V3需求.docx",
            ],
            key_points=[
                "补全指代为小憩状态",
                "点击在线图标可切换状态",
                "小憩使用黄色，在线绿色，离线灰色",
                "显示姓名和工号",
                "与 CTI 后台监控状态同步",
            ],
        ),
    },
    {
        "id": "holdout-04",
        "split": "holdout",
        "category": "multi_turn",
        "input": {
            "history": [
                {"role": "user", "content": "为什么要做 AI 智能回复？"},
                {
                    "role": "assistant",
                    "content": "为了提升渠道工程师自主解决问题的能力。",
                },
            ],
            "question": "那它需要具备什么能力？",
        },
        "expected_output": expected(
            expected_sources=["星云服务平台AI助手需求.docx"],
            key_points=[
                "补全指代为 AI 智能回复 Agent",
                "总结分析上下文并识别客户问题",
                "咨询类包含确认诉求、直接作答和开放式结尾",
                "配置类包含风险提示、步骤拆解和验证方法",
            ],
        ),
    },
    {
        "id": "holdout-05",
        "split": "holdout",
        "category": "refusal",
        "input": {"question": "敏感词词库具体包含哪些词？"},
        "expected_output": expected(
            should_refuse=True,
            should_review=True,
            key_points=[
                "现有资料只说明词库位置和过滤方式",
                "没有提供具体敏感词内容",
                "需要补充需求不明确工作表或人工确认",
            ],
        ),
    },
    {
        "id": "holdout-06",
        "split": "holdout",
        "category": "refusal",
        "input": {"question": "AI 智能回复当前真实采纳率是多少？"},
        "expected_output": expected(
            should_refuse=True,
            should_review=True,
            key_points=[
                "文档只规划了 AI 建议采纳率指标",
                "没有提供实际采纳率数值",
                "需要补充线上统计数据",
            ],
        ),
    },
    {
        "id": "holdout-07",
        "split": "holdout",
        "category": "refusal",
        "input": {"question": "客户小程序刷新按钮已经上线了吗？"},
        "expected_output": expected(
            should_refuse=True,
            should_review=True,
            key_points=[
                "需求文档存在刷新按钮设计",
                "需求文档不能证明已经开发或上线",
                "需要实施状态或发布记录确认",
            ],
        ),
    },
    {
        "id": "holdout-08",
        "split": "holdout",
        "category": "conflict",
        "input": {
            "question": "新需求：坐席进入小憩后仍继续分配新会话，请检查冲突。"
        },
        "expected_output": expected(
            intent="conflict_check",
            should_review=True,
            expected_sources=[
                "星云服务平台V2需求.docx",
                "星云服务平台V3需求.docx",
            ],
            key_points=[
                "与小憩暂时不分配新会话的历史设计直接矛盾",
                "影响坐席状态语义和调度逻辑",
                "当前显式记忆显示功能尚未上线",
                "需要人工复核或需求变更流程",
            ],
            evaluate_refusal=False,
        ),
    },
    {
        "id": "holdout-09",
        "split": "holdout",
        "category": "conflict",
        "input": {
            "question": "新需求：把最大并发从 7 调成 6 时，立即中断第 7 个已有会话，请检查冲突。"
        },
        "expected_output": expected(
            intent="conflict_check",
            should_review=True,
            expected_sources=["星云服务平台V3需求.docx"],
            key_points=[
                "与不强制中断当前已接入会话的设计冲突",
                "历史方案允许暂时显示 7/6",
                "历史方案只暂停新会话分配",
                "需要确认是否修改并发调整规则",
            ],
            evaluate_refusal=False,
        ),
    },
]


def items_for_split(split: str) -> list[dict]:
    if split == "all":
        return EVALUATION_ITEMS
    return [
        item
        for item in EVALUATION_ITEMS
        if item["split"] == split
    ]

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Archive,
  BookOpen,
  Bot,
  Check,
  ChevronDown,
  CircleDot,
  Clock3,
  FileText,
  FlaskConical,
  Menu,
  MessageSquareText,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RefreshCw,
  Send,
  ShieldAlert,
  Sparkles,
  X,
} from "lucide-react";

const API = "/api";
const LANGFUSE_URL =
  "https://cloud.langfuse.com";

const intentLabels = {
  knowledge_query: "知识查询",
  conflict_check: "冲突检测",
  memory_write: "记忆写入",
};

const memoryTypeLabels = {
  decision: "产品决策",
  implementation_status: "实施状态",
  constraint: "产品约束",
  historical_fact: "历史事实",
};

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "请求失败");
  }
  return response.json();
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function TextContent({ children }) {
  const lines = String(children || "").split("\n");
  return (
    <div className="message-text">
      {lines.map((line, index) => (
        <p key={`${line}-${index}`}>{line || "\u00a0"}</p>
      ))}
    </div>
  );
}

function StatusBadge({ payload }) {
  if (!payload) return null;
  const conflict = payload.structured_result?.has_conflict;
  const insufficient = payload.structured_result?.sufficient === false;
  let tone = "neutral";
  let label = intentLabels[payload.intent] || "Agent";

  if (conflict) {
    tone = "danger";
    label = "发现冲突";
  } else if (insufficient || payload.needs_review) {
    tone = "warning";
    label = insufficient ? "资料不足" : "需要复核";
  } else if (payload.intent) {
    tone = "success";
  }

  return <span className={`status-badge ${tone}`}>{label}</span>;
}

function ReviewCard({ review, busy, onReview }) {
  const draft = review?.draft || {};
  const [comment, setComment] = useState("");

  return (
    <section className="review-card">
      <div className="review-title">
        <Clock3 size={16} />
        <strong>产品记忆待审批</strong>
      </div>
      <dl className="draft-grid">
        <div>
          <dt>类型</dt>
          <dd>{memoryTypeLabels[draft.memory_type] || draft.memory_type}</dd>
        </div>
        <div>
          <dt>主体</dt>
          <dd>{draft.subject}</dd>
        </div>
        <div className="span-two">
          <dt>陈述</dt>
          <dd>{draft.statement}</dd>
        </div>
        <div>
          <dt>范围</dt>
          <dd>{draft.scope}</dd>
        </div>
        <div>
          <dt>截至日期</dt>
          <dd>{draft.as_of_date || "未提供"}</dd>
        </div>
      </dl>
      <input
        className="review-comment"
        value={comment}
        onChange={(event) => setComment(event.target.value)}
        placeholder="审批意见"
      />
      <div className="review-actions">
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => onReview(false, comment, draft)}
        >
          <X size={16} />
          拒绝
        </button>
        <button
          className="button primary"
          disabled={busy}
          onClick={() => onReview(true, comment, draft)}
        >
          <Check size={16} />
          批准写入
        </button>
      </div>
    </section>
  );
}

function ConflictDetails({ result }) {
  if (!result?.has_conflict) return null;
  return (
    <section className="conflict-block">
      <div className="conflict-heading">
        <ShieldAlert size={17} />
        <strong>冲突详情</strong>
      </div>
      {result.conflicts?.map((item, index) => (
        <div className="conflict-item" key={`${item.conflict_type}-${index}`}>
          <span>{item.conflict_type}</span>
          <p>{item.description}</p>
        </div>
      ))}
      {result.impact?.length > 0 && (
        <div className="plain-list">
          <strong>潜在影响</strong>
          {result.impact.map((item) => <p key={item}>• {item}</p>)}
        </div>
      )}
    </section>
  );
}

function Message({
  message,
  busy,
  onReview,
  onInspect,
}) {
  const payload = message.payload;
  return (
    <article className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === "assistant" ? <Bot size={17} /> : "你"}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>{message.role === "assistant" ? "产品记忆 Agent" : "你"}</strong>
          {message.role === "assistant" && <StatusBadge payload={payload} />}
          <time>{formatTime(message.created_at)}</time>
        </div>
        <TextContent>{message.content}</TextContent>
        {payload && (
          <>
            <ConflictDetails result={payload.structured_result} />
            {payload.pending_review && (
              <ReviewCard
                review={payload.pending_review}
                busy={busy}
                onReview={onReview}
              />
            )}
            <button
              className="inspect-button"
              onClick={() => onInspect(payload)}
            >
              <BookOpen size={15} />
              查看依据
            </button>
          </>
        )}
      </div>
    </article>
  );
}

function EmptyConversation() {
  return (
    <div className="empty-conversation">
      <div className="empty-mark"><Sparkles size={22} /></div>
      <h1>产品知识与决策记忆</h1>
      <p>询问历史需求、提交新方案，或记录一条产品事实。</p>
      <div className="example-row">
        <span>小憩状态做了吗？具体如何设计？</span>
        <span>本期让小憩继续分配会话，有什么风险？</span>
      </div>
    </div>
  );
}

function Sidebar({
  conversations,
  activeId,
  onNew,
  onSelect,
  onShowMemories,
  open,
  onClose,
}) {
  return (
    <aside className={`sidebar ${open ? "mobile-open" : ""}`}>
      <div className="brand">
        <div className="brand-mark">PM</div>
        <div>
          <strong>产品记忆</strong>
          <span>Decision workspace</span>
        </div>
        <button className="mobile-close icon-button" onClick={onClose}>
          <X size={18} />
        </button>
      </div>

      <button className="new-chat" onClick={onNew}>
        <Plus size={17} />
        新建对话
      </button>

      <div className="sidebar-label">最近对话</div>
      <nav className="conversation-list">
        {conversations.map((conversation) => (
          <button
            key={conversation.id}
            className={conversation.id === activeId ? "active" : ""}
            onClick={() => onSelect(conversation.id)}
          >
            <MessageSquareText size={16} />
            <span>{conversation.title}</span>
            <time>{formatTime(conversation.updated_at)}</time>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button onClick={onShowMemories}>
          <Archive size={17} />
          产品记忆
        </button>
        <a href={LANGFUSE_URL} target="_blank" rel="noreferrer">
          <FlaskConical size={17} />
          Langfuse 实验
        </a>
      </div>
    </aside>
  );
}

function EvidenceItem({ item }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="evidence-item">
      <button onClick={() => setExpanded(!expanded)}>
        <div>
          <span className="evidence-id">{item.id}</span>
          <strong>{item.source}</strong>
          <small>{item.section}</small>
        </div>
        <ChevronDown
          size={16}
          className={expanded ? "rotated" : ""}
        />
      </button>
      {expanded && <p>{item.content}</p>}
    </div>
  );
}

function ContextPanel({ payload, open, onToggle }) {
  const [tab, setTab] = useState("steps");
  const documents = payload?.documents || [];
  const memories = payload?.memories || [];

  return (
    <aside className={`context-panel ${open ? "" : "collapsed"}`}>
      <header>
        <div>
          <strong>上下文</strong>
          <span>{payload?.route_reason || "选择一条 Agent 回复"}</span>
        </div>
        <button className="icon-button" onClick={onToggle} title="折叠上下文">
          <PanelRightClose size={18} />
        </button>
      </header>
      <div className="context-tabs">
        <button
          className={tab === "steps" ? "active" : ""}
          onClick={() => setTab("steps")}
        >
          执行过程
        </button>
        <button
          className={tab === "docs" ? "active" : ""}
          onClick={() => setTab("docs")}
        >
          文档 {documents.length || ""}
        </button>
        <button
          className={tab === "memory" ? "active" : ""}
          onClick={() => setTab("memory")}
        >
          记忆 {memories.length || ""}
        </button>
      </div>

      <div className="context-content">
        {tab === "steps" && (
          <div className="step-list">
            {(payload?.steps || []).map((step, index) => (
              <div className="step" key={`${step.name}-${index}`}>
                <span className={step.status}>
                  {step.status === "waiting"
                    ? <Clock3 size={14} />
                    : <Check size={14} />}
                </span>
                <div>
                  <strong>{step.name}</strong>
                  <p>{step.detail}</p>
                </div>
              </div>
            ))}
            {!payload && <PanelEmpty text="暂无执行记录" />}
          </div>
        )}
        {tab === "docs" && (
          <div className="evidence-list">
            {documents.map((item) => (
              <EvidenceItem item={item} key={item.id} />
            ))}
            {documents.length === 0 && <PanelEmpty text="暂无文档证据" />}
          </div>
        )}
        {tab === "memory" && (
          <div className="memory-list compact">
            {memories.map((memory) => (
              <article key={`${memory.citation_id}-${memory.id}`}>
                <div>
                  <span>{memory.citation_id}</span>
                  <small>{memoryTypeLabels[memory.memory_type]}</small>
                </div>
                <strong>{memory.subject}</strong>
                <p>{memory.statement}</p>
                <time>截至 {memory.as_of_date || "未标注"}</time>
              </article>
            ))}
            {memories.length === 0 && <PanelEmpty text="暂无相关产品记忆" />}
          </div>
        )}
      </div>
    </aside>
  );
}

function PanelEmpty({ text }) {
  return (
    <div className="panel-empty">
      <CircleDot size={18} />
      <span>{text}</span>
    </div>
  );
}

function MemoryDrawer({ open, memories, loading, onClose, onRefresh }) {
  return (
    <div className={`drawer-layer ${open ? "visible" : ""}`}>
      <button className="drawer-backdrop" onClick={onClose} aria-label="关闭" />
      <aside className="memory-drawer">
        <header>
          <div>
            <span>Knowledge memory</span>
            <h2>产品记忆</h2>
          </div>
          <div>
            <button className="icon-button" onClick={onRefresh} title="刷新">
              <RefreshCw size={17} />
            </button>
            <button className="icon-button" onClick={onClose} title="关闭">
              <X size={18} />
            </button>
          </div>
        </header>
        <div className="memory-drawer-content">
          {loading && <div className="loading-row"><RefreshCw size={17} /> 正在读取</div>}
          {!loading && memories.map((memory) => (
            <article key={memory.id}>
              <div className="memory-card-head">
                <span>{memoryTypeLabels[memory.memory_type]}</span>
                <small className={memory.status}>{memory.status}</small>
              </div>
              <h3>{memory.subject}</h3>
              <p>{memory.statement}</p>
              <dl>
                <div><dt>范围</dt><dd>{memory.scope}</dd></div>
                <div><dt>截至</dt><dd>{memory.as_of_date || "未标注"}</dd></div>
                <div><dt>审批</dt><dd>{memory.review_comment || "无"}</dd></div>
              </dl>
            </article>
          ))}
        </div>
      </aside>
    </div>
  );
}

export function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedPayload, setSelectedPayload] = useState(null);
  const [contextOpen, setContextOpen] = useState(
    () => window.innerWidth > 860,
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [memories, setMemories] = useState([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const scrollRef = useRef(null);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeId),
    [conversations, activeId],
  );

  async function loadConversations(selectFirst = false) {
    const data = await request("/conversations");
    setConversations(data);
    if (selectFirst && !activeId && data[0]) {
      await selectConversation(data[0].id);
    }
  }

  async function selectConversation(id) {
    setActiveId(id);
    setSidebarOpen(false);
    const data = await request(`/conversations/${id}/messages`);
    setMessages(data);
    const lastPayload = [...data].reverse().find((item) => item.payload)?.payload;
    setSelectedPayload(lastPayload || null);
  }

  async function createConversation() {
    const conversation = await request("/conversations", { method: "POST" });
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setMessages([]);
    setSelectedPayload(null);
    setSidebarOpen(false);
  }

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;
    setError("");
    setInput("");
    setBusy(true);
    const optimistic = {
      id: `local-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((current) => [...current, optimistic]);

    try {
      const result = await request("/chat", {
        method: "POST",
        body: JSON.stringify({
          conversation_id: activeId,
          message: text,
        }),
      });
      setActiveId(result.conversation_id);
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.answer,
        payload: result,
        created_at: new Date().toISOString(),
      };
      setMessages((current) => [...current, assistantMessage]);
      setSelectedPayload(result);
      setContextOpen(true);
      await loadConversations();
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => current.filter((item) => item.id !== optimistic.id));
      setInput(text);
    } finally {
      setBusy(false);
    }
  }

  async function submitReview(approved, comment, draft) {
    if (!activeId || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await request(`/conversations/${activeId}/review`, {
        method: "POST",
        body: JSON.stringify({ approved, comment, draft }),
      });
      setMessages((current) => [
        ...current,
        {
          id: `review-${Date.now()}`,
          role: "assistant",
          content: result.answer,
          payload: result,
          created_at: new Date().toISOString(),
        },
      ]);
      setSelectedPayload(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadMemories() {
    setMemoriesLoading(true);
    try {
      setMemories(await request("/memories"));
    } finally {
      setMemoriesLoading(false);
    }
  }

  function openMemories() {
    setMemoryOpen(true);
    loadMemories();
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  useEffect(() => {
    loadConversations(true).catch((loadError) => setError(loadError.message));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, busy]);

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onNew={createConversation}
        onSelect={selectConversation}
        onShowMemories={openMemories}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="workspace">
        <header className="workspace-header">
          <button
            className="mobile-menu icon-button"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={19} />
          </button>
          <div>
            <strong>{activeConversation?.title || "新对话"}</strong>
            <span>自动路由 · 需求文档 · 显式记忆</span>
          </div>
          {!contextOpen && (
            <button
              className="icon-button"
              onClick={() => setContextOpen(true)}
              title="展开上下文"
            >
              <PanelRightOpen size={19} />
            </button>
          )}
        </header>

        <div className="conversation" ref={scrollRef}>
          {messages.length === 0 && <EmptyConversation />}
          <div className="message-stream">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message}
                busy={busy}
                onReview={submitReview}
                onInspect={(payload) => {
                  setSelectedPayload(payload);
                  setContextOpen(true);
                }}
              />
            ))}
            {busy && (
              <div className="agent-running">
                <RefreshCw size={16} />
                Agent 正在检索并判断
              </div>
            )}
          </div>
        </div>

        <footer className="composer-wrap">
          {error && (
            <div className="error-banner">
              <AlertTriangle size={15} />
              {error}
            </div>
          )}
          <div className="composer">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="询问历史需求，提交新方案，或记录一条产品事实..."
              rows={1}
            />
            <button
              className="send-button"
              disabled={!input.trim() || busy}
              onClick={sendMessage}
              title="发送"
            >
              <Send size={18} />
            </button>
          </div>
        </footer>
      </main>

      <ContextPanel
        payload={selectedPayload}
        open={contextOpen}
        onToggle={() => setContextOpen(false)}
      />

      <MemoryDrawer
        open={memoryOpen}
        memories={memories}
        loading={memoriesLoading}
        onClose={() => setMemoryOpen(false)}
        onRefresh={loadMemories}
      />
    </div>
  );
}

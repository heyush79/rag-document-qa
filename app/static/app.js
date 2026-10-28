// The web UI: a plain JavaScript client for the API in this repository. No build step.
"use strict";

const API = "/api/v1";
const POLL_MS = 1500;
const SUGGESTIONS = [
  "What does the handbook say about paid leave?",
  "Who leads the Data team and what is their on-call channel?",
  "How do I reset my password?",
  "Summarise the documents in this collection",
];

// ---------------------------------------------------------------------------
// state and small helpers
// ---------------------------------------------------------------------------

const state = {
  apiKey: localStorage.getItem("rag_api_key") || "",
  theme: localStorage.getItem("rag_theme") || "system",
  libraryOpen: localStorage.getItem("rag_library") !== "closed",
  sidebarOpen: false,
  scope: "", // collection id, or "" for everything
  conversationId: "", // "" means a new chat that is created on the first question
  conversation: null,
  collections: [],
  conversations: [],
  documents: [],
  pollTimer: null,
  busy: false,
  seq: 0,
};

const $ = (selector) => document.querySelector(selector);

const ICONS = {
  plus: '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
  x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  menu: '<line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>',
  settings:
    '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
  sun: '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
  moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
  folder: '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
  send: '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
  layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
  refresh: '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
};

const svg = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function timeAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

function collectionName(id) {
  if (!id) return "All collections";
  const found = state.collections.find((c) => c.id === id);
  return found ? found.name : id;
}

function toast(message, kind = "info") {
  const node = el("div", { class: `toast ${kind}`, text: message, role: "status" });
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "bad" ? 7000 : 4000);
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

function headers(extra = {}) {
  const result = { ...extra };
  if (state.apiKey) result.Authorization = `Bearer ${state.apiKey}`;
  return result;
}

async function readError(response) {
  try {
    const body = await response.json();
    if (body.message) return body.message;
    if (body.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch (_) {
    // not JSON
  }
  return `${response.status} ${response.statusText}`;
}

async function api(path, { method = "GET", json, form } = {}) {
  const options = { method, headers: headers(json ? { "Content-Type": "application/json" } : {}) };
  if (json) options.body = JSON.stringify(json);
  if (form) options.body = form;
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) throw new Error(await readError(response));
  if (response.status === 204) return null;
  const body = await response.json();
  body._status = response.status;
  return body;
}

// Reads a text/event-stream response and calls onEvent with each `data:` object.
async function streamEvents(path, json, onEvent) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify(json),
  });
  if (!response.ok) throw new Error(await readError(response));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data:")) onEvent(JSON.parse(line.slice(5).trim()));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Markdown: enough for model answers (paragraphs, lists, emphasis, code, headings)
// ---------------------------------------------------------------------------

function inline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
    .replace(/\[S(\d+)\]/g, (_, n) => `<a class="cite" href="#" data-source="${n}" title="Source ${n}">${n}</a>`);
}

function renderMarkdown(source) {
  const lines = escapeHtml(source).split("\n");
  let html = "";
  let paragraph = [];
  let list = null;
  let inCode = false;
  const flushParagraph = () => {
    if (paragraph.length) html += `<p>${inline(paragraph.join(" "))}</p>`;
    paragraph = [];
  };
  const closeList = () => {
    if (list) html += `</${list}>`;
    list = null;
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      flushParagraph();
      closeList();
      html += inCode ? "</code></pre>" : "<pre><code>";
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      html += `${raw}\n`;
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = Math.min(heading[1].length + 2, 6);
      html += `<h${level}>${inline(heading[2])}</h${level}>`;
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) {
      flushParagraph();
      closeList();
      html += "<hr>";
      continue;
    }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      const type = bullet ? "ul" : "ol";
      if (list !== type) {
        closeList();
        html += `<${type}>`;
        list = type;
      }
      html += `<li>${inline((bullet || numbered)[1])}</li>`;
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      closeList();
      continue;
    }
    paragraph.push(line.trim());
  }
  flushParagraph();
  closeList();
  if (inCode) html += "</code></pre>";
  return html;
}

// ---------------------------------------------------------------------------
// layout, theme, settings
// ---------------------------------------------------------------------------

function syncLayout() {
  const app = $("#app");
  app.dataset.library = state.libraryOpen ? "open" : "closed";
  app.dataset.sidebar = state.sidebarOpen ? "open" : "closed";
  $("#toggle-library").setAttribute("aria-expanded", String(state.libraryOpen));
  const narrowLibrary = window.matchMedia("(max-width: 1180px)").matches;
  const narrowSidebar = window.matchMedia("(max-width: 820px)").matches;
  const overlay = (state.libraryOpen && narrowLibrary) || (state.sidebarOpen && narrowSidebar);
  $("#scrim").hidden = !overlay;
}

function setLibrary(open) {
  state.libraryOpen = open;
  localStorage.setItem("rag_library", open ? "open" : "closed");
  syncLayout();
}

function setSidebar(open) {
  state.sidebarOpen = open;
  syncLayout();
}

function effectiveTheme() {
  if (state.theme !== "system") return state.theme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme() {
  const root = document.documentElement;
  if (state.theme === "system") delete root.dataset.theme;
  else root.dataset.theme = state.theme;
  $("#toggle-theme").innerHTML = svg(effectiveTheme() === "dark" ? "sun" : "moon");
  $("#theme").value = state.theme;
}

function setTheme(theme) {
  state.theme = theme;
  localStorage.setItem("rag_theme", theme);
  applyTheme();
}

async function loadHealth(deep = false) {
  const dot = $("#health-dot");
  const text = $("#health-text");
  text.textContent = deep ? "Deep check…" : "Checking…";
  dot.className = "dot";
  try {
    const health = await api(`/health${deep ? "?deep=true" : ""}`);
    delete health._status;
    dot.className = `dot ${health.status === "ok" ? "ok" : "warn"}`;
    const problems = [];
    if (health.sqlite !== "ok") problems.push("SQLite");
    if (health.chroma !== "ok") problems.push("Chroma");
    if (health.embedder && health.embedder.status !== "ok") problems.push("embedder");
    for (const provider of health.llm || []) if (provider.status !== "ok") problems.push(provider.model);
    text.textContent = problems.length ? `Degraded: ${problems.join(", ")}` : `Healthy · ${health.embedding_backend}`;
    if (deep) {
      $("#health-detail").textContent = JSON.stringify(health, null, 2);
      toast(health.status === "ok" ? "Deep check passed" : `Deep check: ${problems.join(", ")} not ok`, health.status === "ok" ? "ok" : "bad");
    }
  } catch (error) {
    dot.className = "dot bad";
    text.textContent = "Unreachable";
    if (deep) $("#health-detail").textContent = error.message;
  }
}

async function loadStats() {
  try {
    const stats = await api("/stats");
    const usage = stats.llm_usage;
    const pairs = [
      ["Documents", stats.document_count],
      ["Chunks", stats.vector_count],
      ["Collections", stats.collection_count],
      ["Chats", stats.conversation_count],
      ["Queries", stats.query_count],
      ["LLM calls", `${usage.calls}${usage.failed_calls ? ` (${usage.failed_calls} failed)` : ""}`],
      ["Prompt tokens", usage.prompt_tokens.toLocaleString()],
      ["Answer tokens", usage.completion_tokens.toLocaleString()],
    ];
    $("#stats").replaceChildren(...pairs.map(([label, value]) => el("div", {}, [el("strong", { text: String(value) }), el("span", { text: ` ${label}` })])));
  } catch (error) {
    $("#stats").replaceChildren(el("div", { text: `Stats unavailable: ${error.message}` }));
  }
}

// ---------------------------------------------------------------------------
// collections and scope
// ---------------------------------------------------------------------------

async function loadCollections() {
  const data = await api("/collections");
  state.collections = data.items;
  renderCollections();
}

function renderCollections() {
  const total = state.collections.reduce((n, c) => n + c.document_count, 0);
  const item = (id, name, count, extra) =>
    el("li", { class: `side-item${state.scope === id ? " active" : ""}`, onclick: () => selectScope(id) }, [
      el("span", { class: "label", text: name }),
      el("span", { class: "count", text: String(count) }),
      extra,
    ]);
  $("#collection-list").replaceChildren(
    item("", "All collections", total, null),
    ...state.collections.map((collection) =>
      item(
        collection.id,
        collection.name,
        collection.document_count,
        collection.id === "default"
          ? null
          : el("button", {
              class: "icon-btn danger",
              html: svg("trash"),
              "aria-label": `Delete ${collection.name}`,
              onclick: (event) => {
                event.stopPropagation();
                deleteCollection(collection);
              },
            }),
      ),
    ),
  );
  const scope = $("#scope");
  scope.replaceChildren(
    el("option", { value: "", text: "All collections" }),
    ...state.collections.map((c) => el("option", { value: c.id, text: c.name })),
  );
  scope.value = state.scope;
  syncScopeControl();
}

function syncScopeControl() {
  const bound = state.conversation && state.conversation.collection_id;
  const scope = $("#scope");
  scope.disabled = Boolean(bound);
  scope.title = bound ? "This chat is bound to its collection" : "";
  $("#library-scope").textContent = state.scope ? `· ${collectionName(state.scope)}` : "· all";
  $("#thread-subtitle").textContent = state.scope ? `Searching ${collectionName(state.scope)}` : "Searching all collections";
}

function selectScope(id) {
  state.scope = id;
  // A chat bound to another collection can't search this one; start a new chat instead.
  if (state.conversation && state.conversation.collection_id && state.conversation.collection_id !== id) {
    startNewChat();
  }
  renderCollections();
  loadDocuments().catch((e) => toast(e.message, "bad"));
  if (window.matchMedia("(max-width: 820px)").matches) setSidebar(false);
}

async function createCollection(event) {
  event.preventDefault();
  const input = $("#collection-name");
  try {
    const created = await api("/collections", { method: "POST", json: { name: input.value.trim() } });
    input.value = "";
    $("#collection-form").classList.add("hidden");
    toast(`Collection "${created.name}" created`, "ok");
    state.scope = created.id;
    await loadCollections();
    await loadDocuments();
    loadStats();
  } catch (error) {
    toast(error.message, "bad");
  }
}

async function deleteCollection(collection) {
  const cascade = collection.document_count > 0;
  const question = cascade
    ? `Delete "${collection.name}" with its ${collection.document_count} document(s), their vectors and its chats?`
    : `Delete the empty collection "${collection.name}"?`;
  if (!confirm(question)) return;
  try {
    await api(`/collections/${collection.id}${cascade ? "?cascade=true" : ""}`, { method: "DELETE" });
    if (state.scope === collection.id) state.scope = "";
    if (state.conversation && state.conversation.collection_id === collection.id) startNewChat();
    await Promise.all([loadCollections(), loadDocuments(), loadConversations(), loadStats()]);
    toast(`Collection "${collection.name}" deleted`, "ok");
  } catch (error) {
    toast(error.message, "bad");
  }
}

// ---------------------------------------------------------------------------
// documents
// ---------------------------------------------------------------------------

async function loadDocuments() {
  const query = state.scope ? `?collection_id=${encodeURIComponent(state.scope)}&limit=200` : "?limit=200";
  const data = await api(`/documents${query}`);
  state.documents = data.items;
  renderDocuments();
  schedulePolling();
}

function renderDocuments() {
  const list = $("#document-list");
  if (!state.documents.length) {
    list.replaceChildren(el("li", { class: "side-empty", text: "No documents here yet. Upload one above." }));
    return;
  }
  list.replaceChildren(
    ...state.documents.map((doc) => {
      let meta;
      if (doc.status === "ready") meta = `${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"} · ${collectionName(doc.collection_id)}`;
      else if (doc.status === "processing") meta = "Indexing…";
      else meta = `Failed: ${doc.error_message || "unknown error"}`;
      return el("li", { class: "doc", title: `${doc.source}\n${doc.id}` }, [
        el("span", { class: `pill ${doc.status}`, title: doc.status }),
        el("span", { class: "title", text: doc.title }),
        el("span", { class: "meta", text: meta, title: meta }),
        el("button", {
          class: "icon-btn danger",
          html: svg("trash"),
          "aria-label": `Delete ${doc.title}`,
          onclick: () => deleteDocument(doc),
        }),
      ]);
    }),
  );
}

// While any document is still processing, refresh until it is ready or failed.
function schedulePolling() {
  clearTimeout(state.pollTimer);
  if (state.documents.some((doc) => doc.status === "processing")) {
    state.pollTimer = setTimeout(() => {
      loadDocuments().then(loadCollections).then(loadStats).catch((e) => toast(e.message, "bad"));
    }, POLL_MS);
  }
}

function reportIngestion(document) {
  if (document.duplicate) toast(`Already indexed as "${document.title}"`, "info");
  else if (document._status === 202) toast(`"${document.title}" accepted; indexing in the background`, "ok");
  else toast(`"${document.title}" indexed (${document.chunk_count} chunks)`, "ok");
}

async function uploadFiles(files) {
  const dropzone = $("#dropzone");
  const background = $("#background").checked ? "?background=true" : "";
  dropzone.classList.add("busy");
  try {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      const title = $("#upload-title").value.trim();
      if (title && files.length === 1) form.append("title", title);
      if (state.scope) form.append("collection_id", state.scope);
      try {
        reportIngestion(await api(`/documents/upload${background}`, { method: "POST", form }));
      } catch (error) {
        toast(`${file.name}: ${error.message}`, "bad");
      }
    }
    $("#upload-title").value = "";
    $("#file-input").value = "";
    await Promise.all([loadDocuments(), loadCollections(), loadStats()]);
  } finally {
    dropzone.classList.remove("busy");
  }
}

async function ingestText(event) {
  event.preventDefault();
  const button = event.submitter;
  const background = $("#background").checked ? "?background=true" : "";
  const json = { title: $("#text-title").value.trim(), text: $("#text-body").value, source: "ui" };
  if (state.scope) json.collection_id = state.scope;
  if (button) button.disabled = true;
  try {
    reportIngestion(await api(`/documents/text${background}`, { method: "POST", json }));
    $("#text-title").value = "";
    $("#text-body").value = "";
    await Promise.all([loadDocuments(), loadCollections(), loadStats()]);
  } catch (error) {
    toast(error.message, "bad");
  } finally {
    if (button) button.disabled = false;
  }
}

async function deleteDocument(doc) {
  if (!confirm(`Delete "${doc.title}" and its vectors?`)) return;
  try {
    await api(`/documents/${doc.id}`, { method: "DELETE" });
    await Promise.all([loadDocuments(), loadCollections(), loadStats()]);
  } catch (error) {
    toast(error.message, "bad");
  }
}

// ---------------------------------------------------------------------------
// conversations
// ---------------------------------------------------------------------------

async function loadConversations() {
  const data = await api("/conversations?limit=100");
  state.conversations = data.items;
  renderConversations();
}

function renderConversations() {
  const list = $("#conversation-list");
  if (!state.conversations.length) {
    list.replaceChildren(el("li", { class: "side-empty", text: "No chats yet." }));
    return;
  }
  list.replaceChildren(
    ...state.conversations.map((conversation) =>
      el(
        "li",
        {
          class: `side-item${conversation.id === state.conversationId ? " active" : ""}`,
          onclick: () => openConversation(conversation.id),
          title: conversation.collection_id ? `In ${collectionName(conversation.collection_id)}` : "Across all collections",
        },
        [
          el("span", { class: "label", text: conversation.title || "New chat" }),
          el("span", { class: "count", text: timeAgo(conversation.updated_at) }),
          el("button", {
            class: "icon-btn danger",
            html: svg("trash"),
            "aria-label": "Delete chat",
            onclick: (event) => {
              event.stopPropagation();
              deleteConversation(conversation);
            },
          }),
        ],
      ),
    ),
  );
}

function startNewChat() {
  state.conversationId = "";
  state.conversation = null;
  $("#thread-title").textContent = "New chat";
  showEmptyThread();
  renderConversations();
  syncScopeControl();
  $("#question").focus();
}

async function openConversation(id) {
  try {
    const detail = await api(`/conversations/${id}`);
    state.conversationId = id;
    state.conversation = detail;
    if (detail.collection_id) state.scope = detail.collection_id;
    $("#thread-title").textContent = detail.title || "New chat";
    renderConversations();
    renderCollections();
    loadDocuments().catch(() => {});
    const thread = $("#thread");
    thread.replaceChildren();
    if (!detail.messages.length) showEmptyThread();
    for (const message of detail.messages) {
      if (message.role === "user") addUserMessage(message.content);
      else addAssistantMessage().finish({ text: message.content, sources: [], note: "earlier answer" });
    }
    thread.scrollTop = thread.scrollHeight;
    if (window.matchMedia("(max-width: 820px)").matches) setSidebar(false);
  } catch (error) {
    toast(error.message, "bad");
  }
}

async function deleteConversation(conversation) {
  if (!confirm(`Delete "${conversation.title || "this chat"}" and its messages?`)) return;
  try {
    await api(`/conversations/${conversation.id}`, { method: "DELETE" });
    if (state.conversationId === conversation.id) startNewChat();
    await loadConversations();
    loadStats();
  } catch (error) {
    toast(error.message, "bad");
  }
}

// Creates the conversation for a new chat on its first question.
async function ensureConversation() {
  if (state.conversationId) return state.conversationId;
  const json = state.scope ? { collection_id: state.scope } : {};
  const created = await api("/conversations", { method: "POST", json });
  state.conversationId = created.id;
  state.conversation = created;
  return created.id;
}

// ---------------------------------------------------------------------------
// the thread
// ---------------------------------------------------------------------------

function showEmptyThread() {
  $("#thread").replaceChildren(
    el("div", { class: "empty" }, [
      el("h3", { text: "Ask your documents" }),
      el("p", { text: "Answers come only from what you have uploaded, with a citation for every statement. Follow-up questions use the earlier turns of the chat." }),
      el(
        "div",
        { class: "chips" },
        SUGGESTIONS.map((text) =>
          el("button", {
            type: "button",
            class: "chip",
            text,
            onclick: () => {
              $("#question").value = text;
              $("#composer").requestSubmit();
            },
          }),
        ),
      ),
    ]),
  );
}

function clearEmptyThread() {
  $("#thread").querySelector(".empty")?.remove();
}

function addUserMessage(text) {
  clearEmptyThread();
  const node = el("div", { class: "msg user" }, [el("div", { class: "bubble", text })]);
  $("#thread").append(node);
  return node;
}

function sourceCard(source, index) {
  const percent = Math.round(Math.max(0, Math.min(1, source.score)) * 100);
  return el("div", { class: "source", "data-n": String(index + 1) }, [
    el("div", { class: "source-head" }, [
      el("span", { class: "cite n", text: String(index + 1) }),
      el("span", { class: "title", text: source.document_title, title: source.document_title }),
      el("span", { class: "score", text: source.score.toFixed(2) }),
    ]),
    el("div", { class: "bar" }, [el("span", { style: `width:${percent}%` })]),
    el("div", { class: "source-text", text: source.text }),
    el("div", { class: "source-meta", text: `chunk ${source.chunk_index} · ${source.source}` }),
  ]);
}

function addAssistantMessage() {
  clearEmptyThread();
  const id = `m${++state.seq}`;
  const content = el("div", { class: "content cursor" });
  const bubble = el("div", { class: "bubble" }, [content]);
  const foot = el("div", { class: "msg-foot hidden" });
  const sources = el("div", { class: "sources" });
  const body = el("div", { class: "body" }, [bubble, foot, sources]);
  const node = el("div", { class: "msg assistant", id }, [el("div", { class: "avatar", text: "AI" }), body]);
  const thread = $("#thread");
  thread.append(node);
  thread.scrollTop = thread.scrollHeight;

  const handle = {
    node,
    setText(text) {
      content.innerHTML = renderMarkdown(text);
      thread.scrollTop = thread.scrollHeight;
    },
    finish({ text, sources: hits = [], searchQuery, question, latencyMs, note }) {
      content.classList.remove("cursor");
      content.innerHTML = renderMarkdown(text || "");
      if (!text) content.textContent = "(empty answer)";
      sources.replaceChildren(...hits.map(sourceCard));
      const parts = [];
      if (hits.length) {
        const toggle = el("button", { type: "button", class: "toggle", text: `${hits.length} source${hits.length === 1 ? "" : "s"}` });
        toggle.addEventListener("click", () => {
          sources.classList.toggle("open");
          toggle.textContent = `${hits.length} source${hits.length === 1 ? "" : "s"}${sources.classList.contains("open") ? " ▴" : ""}`;
        });
        parts.push(toggle);
      } else if (!note) {
        parts.push(el("span", { text: "No relevant sources found" }));
      }
      if (searchQuery && question && searchQuery !== question) {
        parts.push(el("span", { class: "rewrite", text: `Searched as “${searchQuery}”`, title: "The follow-up was rewritten into a standalone query for retrieval" }));
      }
      if (latencyMs !== undefined) parts.push(el("span", { text: latencyMs >= 1000 ? `${(latencyMs / 1000).toFixed(1)} s` : `${Math.round(latencyMs)} ms` }));
      if (note) parts.push(el("span", { text: note }));
      foot.replaceChildren(...parts);
      foot.classList.toggle("hidden", !parts.length);
      content.querySelectorAll("a.cite").forEach((link) =>
        link.addEventListener("click", (event) => {
          event.preventDefault();
          const card = sources.querySelector(`.source[data-n="${link.dataset.source}"]`);
          if (!card) return;
          sources.classList.add("open");
          sources.querySelectorAll(".source").forEach((c) => c.classList.remove("flash"));
          card.classList.add("flash");
          card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }),
      );
      thread.scrollTop = thread.scrollHeight;
    },
    fail(message) {
      content.classList.remove("cursor");
      bubble.classList.add("error");
      content.textContent = `Error: ${message}`;
      foot.classList.add("hidden");
    },
  };
  return handle;
}

async function ask(event) {
  event.preventDefault();
  if (state.busy) return;
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  state.busy = true;
  $("#send").disabled = true;
  input.value = "";
  autosize(input);
  addUserMessage(question);
  const answer = addAssistantMessage();
  const started = performance.now();
  try {
    const conversationId = await ensureConversation();
    const payload = { question, top_k: Number($("#top-k").value) || 4, conversation_id: conversationId };
    if (state.scope && !(state.conversation && state.conversation.collection_id)) payload.collection_id = state.scope;

    if ($("#stream").checked) {
      let text = "";
      let sources = [];
      let searchQuery = question;
      let ended = false;
      await streamEvents("/query/stream", payload, (item) => {
        if (item.type === "source") {
          sources = item.sources;
          searchQuery = item.search_query;
        } else if (item.type === "answer") {
          text += item.answer;
          answer.setText(text);
        } else if (item.type === "end") {
          ended = true;
          answer.finish({ text, sources, searchQuery, question, latencyMs: item.latency_ms });
        } else if (item.type === "error") {
          throw new Error(item.message);
        }
      });
      if (!ended) answer.finish({ text, sources, searchQuery, question, note: "stream ended early", latencyMs: performance.now() - started });
    } else {
      const result = await api("/query", { method: "POST", json: payload });
      answer.finish({ text: result.answer, sources: result.sources, searchQuery: result.search_query, question, latencyMs: result.latency_ms });
    }
    if (!state.conversation?.title) {
      // The server titles a new chat after its first question.
      state.conversation = await api(`/conversations/${conversationId}`);
      $("#thread-title").textContent = state.conversation.title || "New chat";
    }
    loadConversations().catch(() => {});
    loadStats();
  } catch (error) {
    answer.fail(error.message);
    toast(error.message, "bad");
  } finally {
    state.busy = false;
    $("#send").disabled = false;
    input.focus();
  }
}

function autosize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

function refreshAll() {
  loadHealth();
  loadCollections()
    .then(() => Promise.all([loadDocuments(), loadConversations()]))
    .catch((e) => toast(e.message, "bad"));
  loadStats();
}

function init() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    node.innerHTML = svg(node.dataset.icon);
  });
  applyTheme();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  if (localStorage.getItem("rag_library") === null && window.matchMedia("(max-width: 1180px)").matches) state.libraryOpen = false;
  syncLayout();
  window.addEventListener("resize", syncLayout);
  showEmptyThread();

  // sidebar
  $("#open-sidebar").addEventListener("click", () => setSidebar(true));
  $("#close-sidebar").addEventListener("click", () => setSidebar(false));
  $("#new-chat").addEventListener("click", () => {
    startNewChat();
    if (window.matchMedia("(max-width: 820px)").matches) setSidebar(false);
  });
  $("#add-collection").addEventListener("click", () => {
    $("#collection-form").classList.toggle("hidden");
    $("#collection-name").focus();
  });
  $("#collection-form").addEventListener("submit", createCollection);
  $("#health").addEventListener("click", () => loadHealth(true));
  $("#open-settings").addEventListener("click", () => $("#settings").showModal());

  // main
  $("#toggle-theme").addEventListener("click", () => setTheme(effectiveTheme() === "dark" ? "light" : "dark"));
  $("#toggle-library").addEventListener("click", () => setLibrary(!state.libraryOpen));
  $("#scrim").addEventListener("click", () => {
    setSidebar(false);
    if (window.matchMedia("(max-width: 1180px)").matches) setLibrary(false);
  });
  $("#composer").addEventListener("submit", ask);
  $("#scope").addEventListener("change", (event) => selectScope(event.target.value));
  const question = $("#question");
  question.addEventListener("input", () => autosize(question));
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#composer").requestSubmit();
    }
  });

  // library
  $("#refresh-library").addEventListener("click", () => Promise.all([loadDocuments(), loadCollections(), loadStats()]).catch((e) => toast(e.message, "bad")));
  $("#close-library").addEventListener("click", () => setLibrary(false));
  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) uploadFiles([...fileInput.files]);
  });
  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("over");
    });
  }
  dropzone.addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length) uploadFiles([...event.dataTransfer.files]);
  });
  $("#text-form").addEventListener("submit", ingestText);

  // settings
  $("#api-key").value = state.apiKey;
  $("#api-key").addEventListener("change", (event) => {
    state.apiKey = event.target.value.trim();
    localStorage.setItem("rag_api_key", state.apiKey);
    refreshAll();
  });
  $("#theme").addEventListener("change", (event) => setTheme(event.target.value));
  $("#deep-health").addEventListener("click", () => loadHealth(true));

  refreshAll();
}

init();

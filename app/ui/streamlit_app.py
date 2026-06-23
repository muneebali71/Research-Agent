"""Streamlit front-end — Research Agent chatbot with PDF chunks viewer."""
from __future__ import annotations
import streamlit as st
from app.ui.api_client import (
    ApiError, create_session, get_chunks, get_documents,
    get_messages, list_sessions, send_chat, upload_pdf,
)
from app.ui.components import render_badges, render_message
from app.ui.state import (
    append_doc, append_message, current_session_title,
    docs_loaded, get_current_docs, get_current_messages,
    init_state, messages_loaded, set_current_docs,
    set_current_messages, set_current_session,
)

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
init_state()

# ── Pin chat input to bottom of viewport, always visible ──────────────────────
st.markdown("""
<style>
div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 21rem;      /* sidebar width */
    right: 0;
    width: auto;
    z-index: 999;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
    background: var(--background-color);
    border-top: 1px solid rgba(128,128,128,0.2);
}

.main .block-container {
    padding-bottom: 110px;
}
</style>
""", unsafe_allow_html=True)

# ── Page tabs ─────────────────────────────────────────────────────────────────
TAB_CHAT, TAB_CHUNKS = "💬 Chat", "📚 PDF Chunks"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = TAB_CHAT

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 Research Agent")
    st.caption("Multi-agent research & report chatbot")

    with st.expander("⚙️ Settings"):
        st.session_state.api_base_url = st.text_input(
            "Backend URL", value=st.session_state.api_base_url
        )

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        try:
            session = create_session(st.session_state.api_base_url)
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state.sessions.insert(0, session)
            st.session_state.session_messages[session["id"]] = []
            st.session_state.session_docs[session["id"]]     = []
            set_current_session(session["id"])
            st.session_state.active_tab = TAB_CHAT
            st.rerun()

    st.divider()

    if not st.session_state.sessions_loaded:
        try:
            st.session_state.sessions = list_sessions(st.session_state.api_base_url)
            st.session_state.sessions_loaded = True
        except ApiError as exc:
            st.error(str(exc))

    st.caption("Your chats")
    if not st.session_state.sessions:
        st.caption("No sessions yet — create one above.")

    for s in st.session_state.sessions:
        is_active = s["id"] == st.session_state.current_session_id
        if st.button(
            ("▶ " if is_active else "") + (s.get("title") or "New chat"),
            key=f"sess-{s['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                set_current_session(s["id"])
                st.session_state.active_tab = TAB_CHAT
                st.rerun()

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────────────
    st.caption("Documents")
    if st.session_state.current_session_id is None:
        st.info("Create or select a chat first.")
    else:
        sid = st.session_state.current_session_id

        if not docs_loaded(sid):
            try:
                set_current_docs(get_documents(st.session_state.api_base_url, sid))
            except ApiError:
                set_current_docs([])

        pdf_file = st.file_uploader("Upload PDF", type="pdf", key=f"pdf-{sid}")
        if pdf_file and st.button("📤 Index PDF", use_container_width=True):
            with st.spinner("Chunking and embedding..."):
                try:
                    result = upload_pdf(
                        st.session_state.api_base_url, sid,
                        pdf_file.name, pdf_file.getvalue(),
                    )
                except ApiError as exc:
                    st.error(str(exc))
                else:
                    append_doc({"filename": result["filename"], "chunks": result["chunks"]})
                    # Clear cached chunks so viewer reloads fresh
                    st.session_state.pop(f"chunks_{sid}", None)
                    st.success(f"✅ **{result['filename']}** — {result['chunks']} chunks indexed")

        docs = get_current_docs()
        if docs:
            st.caption("Indexed PDFs")
            for doc in docs:
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"📄 **{doc['filename']}**")
                c2.markdown(
                    f"<span style='color:#6366f1;font-size:0.75rem;font-weight:700'>"
                    f"{doc['chunks']} chunks</span>",
                    unsafe_allow_html=True,
                )
            # Button to jump to chunks viewer
            if st.button("🔍 View chunks", use_container_width=True):
                st.session_state.active_tab = TAB_CHUNKS
                st.rerun()


# ── Main panel ────────────────────────────────────────────────────────────────
if st.session_state.current_session_id is None:
    st.title("Welcome 👋")
    st.write("Create a new chat from the sidebar. Upload a PDF and ask questions about it, "
             "or ask any research question — the agent decides how to answer.")
    st.info("**Routing:**\n"
            "- 💬 Greetings → direct reply\n"
            "- 📄 PDF questions → answered from your PDF\n"
            "- 🌐 Research questions → web search\n"
            "- 🧩 Mixed → PDF + web combined")
else:
    sid   = st.session_state.current_session_id
    title = current_session_title() or "Chat"

    tab_chat, tab_chunks = st.tabs([TAB_CHAT, TAB_CHUNKS])

    # ── CHAT TAB ──────────────────────────────────────────────────────────────
    with tab_chat:
        if not messages_loaded(sid):
            try:
                set_current_messages(get_messages(st.session_state.api_base_url, sid))
            except ApiError as exc:
                st.error(str(exc))
                set_current_messages([])

        for msg in get_current_messages():
            render_message(msg)

        query = st.chat_input("Ask a question or say hello...")
        if query:
            append_message({"role": "user", "content": query})
            render_message({"role": "user", "content": query})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = send_chat(st.session_state.api_base_url, sid, query)
                    except ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.markdown(result["answer"])
                        render_badges(result.get("route"), result.get("score"))
                        append_message({
                            "role": "assistant", "content": result["answer"],
                            "route": result.get("route"), "score": result.get("score"),
                        })

    # ── CHUNKS TAB ────────────────────────────────────────────────────────────
    with tab_chunks:
        st.markdown("### 📚 Indexed PDF Chunks")
        st.caption("All text chunks stored in Qdrant for this session.")

        # Load chunks (cached per session in st.session_state)
        cache_key = f"chunks_{sid}"
        if cache_key not in st.session_state:
            with st.spinner("Loading chunks from Qdrant..."):
                try:
                    st.session_state[cache_key] = get_chunks(
                        st.session_state.api_base_url, sid
                    )
                except ApiError as exc:
                    st.error(str(exc))
                    st.session_state[cache_key] = []

        chunks = st.session_state.get(cache_key, [])

        if not chunks:
            st.info("No chunks yet. Upload a PDF from the sidebar first.")
        else:
            # ── Stats row ─────────────────────────────────────────────────────
            sources = sorted(set(c["source_file"] for c in chunks))
            pages   = sorted(set(c["page"] for c in chunks if c["page"] is not None))

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Chunks", len(chunks))
            col2.metric("PDF Files",    len(sources))
            col3.metric("Pages Covered", len(pages))

            st.divider()

            # ── Filters ───────────────────────────────────────────────────────
            fc1, fc2, fc3 = st.columns([2, 2, 3])
            with fc1:
                sel_file = st.selectbox(
                    "Filter by file",
                    ["All files"] + sources,
                    key="chunk_filter_file",
                )
            with fc2:
                page_opts = ["All pages"] + [f"Page {p+1}" for p in sorted(
                    set(c["page"] for c in chunks if c["page"] is not None)
                )]
                sel_page = st.selectbox("Filter by page", page_opts, key="chunk_filter_page")
            with fc3:
                search_q = st.text_input("🔍 Search in chunks", placeholder="Type to filter...",
                                          key="chunk_search")

            # ── Apply filters ─────────────────────────────────────────────────
            filtered = chunks
            if sel_file != "All files":
                filtered = [c for c in filtered if c["source_file"] == sel_file]
            if sel_page != "All pages":
                pg = int(sel_page.replace("Page ", "")) - 1
                filtered = [c for c in filtered if c["page"] == pg]
            if search_q:
                q = search_q.lower()
                filtered = [c for c in filtered if q in c["text"].lower()]

            st.caption(f"Showing **{len(filtered)}** of **{len(chunks)}** chunks")
            st.divider()

            # ── Chunk cards ───────────────────────────────────────────────────
            if not filtered:
                st.warning("No chunks match your filters.")
            else:
                for chunk in filtered:
                    page_label = f"p.{chunk['page'] + 1}" if chunk["page"] is not None else "p.?"
                    header = f"**Chunk #{chunk['id']}** · `{chunk['source_file']}` · `{page_label}`"

                    with st.expander(header, expanded=False):
                        # Word count badge
                        word_count = len(chunk["text"].split())
                        st.markdown(
                            f"<div style='margin-bottom:8px'>"
                            f"<span style='background:#6366f11a;color:#6366f1;padding:2px 10px;"
                            f"border-radius:999px;font-size:0.72rem;font-weight:600'>"
                            f"📄 {chunk['source_file']}</span>&nbsp;"
                            f"<span style='background:#0ea5e91a;color:#0ea5e9;padding:2px 10px;"
                            f"border-radius:999px;font-size:0.72rem;font-weight:600'>"
                            f"Page {chunk['page'] + 1 if chunk['page'] is not None else '?'}</span>&nbsp;"
                            f"<span style='background:#10b9811a;color:#10b981;padding:2px 10px;"
                            f"border-radius:999px;font-size:0.72rem;font-weight:600'>"
                            f"{word_count} words</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # Highlight search term if present
                        text = chunk["text"]
                        if search_q:
                            highlighted = text.replace(
                                search_q,
                                f"**{search_q}**",
                            )
                            st.markdown(highlighted)
                        else:
                            st.markdown(
                                f"<div style='background:#f8fafc;border-left:3px solid #6366f1;"
                                f"padding:12px 16px;border-radius:0 6px 6px 0;"
                                f"font-size:0.9rem;line-height:1.6;color:#1e293b;"
                                f"white-space:pre-wrap'>{text}</div>",
                                unsafe_allow_html=True,
                            )
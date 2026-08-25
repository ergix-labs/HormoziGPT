from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ergix_hormozi.config import Settings
from ergix_hormozi.daily import generate_daily_motions
from ergix_hormozi.ingest import ingest_paths
from ergix_hormozi.ollama import ModelClient, ModelClientError
from ergix_hormozi.prompts import SYSTEM_MESSAGE, grounded_user_prompt
from ergix_hormozi.store import KnowledgeStore


st.set_page_config(page_title="Ergix Operator", page_icon="↗", layout="wide")
settings = Settings.from_env()
store = KnowledgeStore(settings.database_path, max_vector_scan=settings.max_vector_scan)
client = ModelClient(
    settings.ollama_base_url,
    settings.chat_model,
    settings.embedding_model,
    chat_provider=settings.chat_provider,
    chat_base_url=settings.chat_base_url,
    chat_api_key=settings.chat_api_key,
    reasoning_effort=settings.reasoning_effort,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Ergix Operator")
st.caption("Grounded business motions powered by Kimi K3 with a private local knowledge index.")

with st.sidebar:
    st.subheader("Model runtime")
    st.code(
        f"Chat: {settings.chat_model} via {settings.chat_provider}\n"
        f"Reasoning: {settings.reasoning_effort}\n"
        f"Embeddings: {settings.embedding_model} (local)"
    )
    if settings.chat_provider == "moonshot" and not settings.chat_api_key:
        st.error(
            "Moonshot API key missing. Pull `kimi` from Infisical in the Ergix harness "
            "or add MOONSHOT_API_KEY to .env, then restart the app."
        )
    stats = store.stats()
    st.metric("Sources", stats["documents"])
    st.metric("Knowledge chunks", stats["chunks"])

    uploads = st.file_uploader(
        "Add knowledge",
        type=["md", "txt", "pdf", "csv", "json", "jsonl", "html", "htm"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files", use_container_width=True):
        with st.status("Indexing locally…", expanded=True) as status:
            with tempfile.TemporaryDirectory(prefix="ergix-operator-") as temp_dir:
                paths = []
                for upload in uploads:
                    target = Path(temp_dir) / Path(upload.name).name
                    target.write_bytes(upload.getvalue())
                    paths.append(target)
                result = ingest_paths(paths, store, client.embed, settings.embedding_model)
                st.write(result)
                status.update(label="Knowledge indexed", state="complete")
        st.rerun()

    st.divider()
    if st.button("Generate today's motions", type="primary", use_container_width=True):
        profile = settings.business_profile_path.read_text("utf-8") if settings.business_profile_path.exists() else ""
        try:
            with st.spinner("Finding today's highest-leverage moves…"):
                result = generate_daily_motions(store, client, profile, settings.motions_dir)
            st.session_state.daily_motions = result
        except ModelClientError as exc:
            st.error(str(exc))

if result := st.session_state.get("daily_motions"):
    st.subheader(f"Suggested motions · {result['date']}")
    columns = st.columns(len(result["motions"]))
    for column, motion in zip(columns, result["motions"]):
        with column:
            st.markdown(f"### {motion['title']}")
            st.write(motion["why_now"])
            st.markdown(f"**Move:** {motion['action']}")
            st.markdown(f"**Scoreboard:** {motion['metric']} — target **{motion['target']}**")
            st.caption(f"Timebox: {motion['timebox_minutes']} minutes")

st.divider()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about offers, leads, sales, retention, or today's bottleneck"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching the Ergix knowledge base…"):
            hits = store.search(query, embedder=client.embed, top_k=6)
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            messages.extend(st.session_state.messages[-8:])
            messages.append({"role": "user", "content": grounded_user_prompt(query, hits)})
            try:
                answer = client.chat(messages)
            except ModelClientError as exc:
                answer = f"Model connection error: {exc}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

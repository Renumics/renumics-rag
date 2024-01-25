import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, get_args

import streamlit as st
from langchain_core.runnables import Runnable

from assistant import (
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    format_doc,
    get_chat_model,
    get_chromadb,
    get_embeddings_model,
    get_rag_chain,
    get_retriever,
)


Role = Literal["user", "assistant", "source"]


AVATARS: Dict[Role, Any] = {"user": "🧐", "assistant": "🤖", "source": "🔖"}


@dataclasses.dataclass
class Message:
    role: Role
    content: str
    sources: Optional[List[str]] = None


def set_embeddings_model_name(value: str) -> None:
    st.session_state.embeddings_model_name = value


st.set_page_config(page_title="F1 RAG Demo", page_icon="🏎️")
st.title("F1 RAG Demo 🤖➕🔖❤️🏎️")

st.header("Chat with the F1 docs")


# if "embeddings_model_name" not in st.session_state.keys():
#     st.session_state.embeddings_model_name = "text-embedding-ada-002"
# if "chat_model_name" not in st.session_state.keys():
#     st.session_state.chat_model_name = "gpt-4"
if "messages" not in st.session_state.keys():
    st.session_state.messages = [Message("assistant", "Ask me a question about F1", [])]


@st.cache_resource(show_spinner=False)
def get_chain(
    embeddings_model_name: str,
    chat_model_name: str,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
) -> Runnable:
    print("Load chain", embeddings_model_name, chat_model_name, relevance_score_fn)
    with st.spinner(text="Loading database and LLMs..."):
        # embeddings_model = get_embeddings_model("text-embedding-ada-002")
        embeddings_model = get_embeddings_model(embeddings_model_name)
        chat_model = get_chat_model(chat_model_name)
        vectorstore = get_chromadb(
            persist_directory=Path("./db-docs"),
            embeddings_model=embeddings_model,
            collection_name="docs_store",
            relevance_score_fn=relevance_score_fn,
        )
        retriever = get_retriever(vectorstore, k)

        chain = get_rag_chain(retriever, chat_model)
        return chain


with st.sidebar:
    st.header("Settings")
    st.subheader("LLM models")
    st.text_input(
        "Embeddings model", value="text-embedding-ada-002", key="embeddings_model_name"
    )
    st.text_input("Chat model", value="gpt-4", key="chat_model_name")
    st.subheader("Retriever")
    st.selectbox(
        "Relevance score",
        get_args(PredefinedRelevanceScoreFn),
        get_args(PredefinedRelevanceScoreFn).index("l2"),
        key="relevance_score_fn",
    )
    st.slider("k", 0, 100, 4, 1, key="k")


chain = get_chain(
    st.session_state.embeddings_model_name,
    st.session_state.chat_model_name,
    st.session_state.relevance_score_fn,
    st.session_state.k,
)


if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append(Message("user", prompt))

for message in st.session_state.messages:  # Display the prior chat messages
    if message.role == "assistant":
        for source in message.sources:
            with st.chat_message("source", avatar=AVATARS.get("source")):
                st.write(source)
        with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
            st.write(message.content)
    else:
        with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
            st.write(message.content)

if st.session_state.messages[-1].role == "user":
    with st.spinner("Thinking..."):
        answer = chain.invoke(prompt)
        message = Message(
            "assistant",
            answer["answer"],
            [format_doc(doc) for doc in answer["source_documents"]],
        )
        for source in message.sources:
            with st.chat_message("source", avatar=AVATARS.get("source")):
                st.write(source)
        with st.chat_message("assistant", avatar=AVATARS.get("assistant")):
            st.write(message.content)
        st.session_state.messages.append(message)

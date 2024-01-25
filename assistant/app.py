import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import streamlit as st
from langchain_core.runnables import Runnable

from assistant import (
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


st.set_page_config(page_title="F1 RAG Demo", page_icon="🏎️")
st.title("F1 RAG Demo 🤖➕🔖❤️🏎️")

st.header("Chat with the F1 docs")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [Message("assistant", "Ask me a question about F1", [])]


@st.cache_resource(show_spinner=False)
def get_chain() -> Runnable:
    with st.spinner(text="Loading database and LLMs..."):
        embeddings_model = get_embeddings_model("text-embedding-ada-002")
        chat_model = get_chat_model("gpt-4")
        vectorstore = get_chromadb(
            persist_directory=Path("./db-docs"),
            embeddings_model=embeddings_model,
            collection_name="docs_store",
        )
        retriever = get_retriever(vectorstore)

        chain = get_rag_chain(retriever, chat_model)
        return chain


chain = get_chain()


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

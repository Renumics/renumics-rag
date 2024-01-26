import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Literal, get_args

import streamlit as st
from langchain_core.runnables import Runnable
from langchain_core.documents import Document


from assistant import (
    MODEL_TYPES,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RETRIEVER_SEARCH_TYPES,
    ModelType,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
    format_doc,
    get_chat_model,
    get_chromadb,
    get_embeddings_model,
    get_rag_chain,
    get_retriever,
    stable_hash,
)


Role = Literal["user", "assistant", "source"]


AVATARS: Dict[Role, Any] = {"user": "🧐", "assistant": "🤖", "source": "📚"}


@dataclasses.dataclass
class Message:
    role: Role
    content: str


def set_embeddings_model_name(value: str) -> None:
    st.session_state.embeddings_model_name = value


st.set_page_config(page_title="F1 RAG Demo", page_icon="🏎️", layout="wide")
st.title("F1 RAG Demo 🤖➕📚❤️🏎️")

st.header("Chat with the F1 docs")


if "messages" not in st.session_state.keys():
    st.session_state.messages = [Message("assistant", "Ask me a question about F1")]


@st.cache_resource(show_spinner=False)
def _get_rag_chain(
    embeddings_model_type: ModelType,
    embeddings_model_name: str,
    chat_model_type: ModelType,
    chat_model_name: str,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
    search_type: RetrieverSearchType,
    score_threshold: float,
    fetch_k: int,
    lambda_mult: float,
) -> Runnable:
    print(
        "Load chain",
        embeddings_model_type,
        embeddings_model_name,
        chat_model_type,
        chat_model_name,
        relevance_score_fn,
        k,
        search_type,
        score_threshold,
        fetch_k,
        lambda_mult,
    )
    with st.spinner(text="Loading database and LLMs..."):
        embeddings_model = get_embeddings_model(
            embeddings_model_name, embeddings_model_type
        )
        chat_model = get_chat_model(chat_model_name, chat_model_type)
        vectorstore = get_chromadb(
            persist_directory=Path("./db-docs"),
            embeddings_model=embeddings_model,
            collection_name="docs_store",
            relevance_score_fn=relevance_score_fn,
        )
        retriever = get_retriever(
            vectorstore, k, search_type, score_threshold, fetch_k, lambda_mult
        )
        chain = get_rag_chain(retriever, chat_model)
        return chain


_get_chroma_db = st.cache_resource(show_spinner=False)(get_chromadb)


with st.sidebar:
    st.header("Settings")
    st.subheader("LLM models")
    st.selectbox(
        "Embeddings model type",
        get_args(ModelType),
        get_args(ModelType).index("openai"),
        format_func=lambda x: MODEL_TYPES.get(x, x),
        key="embeddings_model_type",
    )
    st.text_input(
        "Embeddings model", value="text-embedding-ada-002", key="embeddings_model_name"
    )
    st.selectbox(
        "Chat model type",
        get_args(ModelType),
        get_args(ModelType).index("openai"),
        format_func=lambda x: MODEL_TYPES.get(x, x),
        key="chat_model_type",
    )
    st.text_input("Chat model", value="gpt-4", key="chat_model_name")
    with st.expander("Advanced"):
        st.subheader("Retriever")
        st.selectbox(
            "Relevance score function",
            get_args(PredefinedRelevanceScoreFn),
            get_args(PredefinedRelevanceScoreFn).index("l2"),
            format_func=lambda x: PREDEFINED_RELEVANCE_SCORE_FNS.get(x, x),
            key="relevance_score_fn",
            help="Distance function in the embedding space "
            "([more](https://docs.trychroma.com/usage-guide#changing-the-distance-function))",
        )
        k = st.slider("k", 0, 100, 4, key="k", help="Amount of documents to return")
        search_type = st.selectbox(
            "Search type",
            get_args(RetrieverSearchType),
            get_args(RetrieverSearchType).index("similarity"),
            format_func=lambda x: RETRIEVER_SEARCH_TYPES.get(x, x),
            key="search_type",
            help="Type of search",
        )
        st.slider(
            "Score threshold",
            0.0,
            1.0,
            0.5,
            key="score_threshold",
            help="Minimum relevance threshold",
            disabled=search_type != "similarity_score_threshold",
        )
        st.slider(
            "Fetch k",
            k,
            max(200, k * 2),
            max(20, k + 10),
            key="fetch_k",
            help="Amount of documents to pass to MMR",
            disabled=search_type != "mmr",
        )
        st.slider(
            "MMR λ",
            0.0,
            1.0,
            0.5,
            key="lambda_mult",
            help="Diversity of results returned by MMR. 1 for minimum diversity and 0 for maximum.",
            disabled=search_type != "mmr",
        )


chain = _get_rag_chain(
    st.session_state.embeddings_model_type,
    st.session_state.embeddings_model_name,
    st.session_state.chat_model_type,
    st.session_state.chat_model_name,
    st.session_state.relevance_score_fn,
    st.session_state.k,
    st.session_state.search_type,
    st.session_state.score_threshold,
    st.session_state.fetch_k,
    st.session_state.lambda_mult,
)

questions_vectorstore = _get_chroma_db(
    persist_directory=Path("./db-out"),
    embeddings_model=get_embeddings_model(
        st.session_state.embeddings_model_name, st.session_state.embeddings_model_type
    ),
    collection_name=f"questions_docs_store",
)

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append(Message("user", prompt))

for message in st.session_state.messages:
    with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
        st.write(message.content)

if st.session_state.messages[-1].role == "user":
    with st.spinner("Thinking..."):
        answer = chain.invoke(prompt)

        # store question and answer in db
        questions_vectorstore.add_documents(
            [
                Document(
                    page_content=prompt,
                    metadata={
                        "answer": answer["answer"],
                        "sources": ",".join(
                            map(stable_hash, answer["source_documents"])
                        ),
                    },
                )
            ]
        )

        messages: List[Message] = []
        for doc in answer["source_documents"]:
            messages.append(Message("source", format_doc(doc)))
        messages.append(Message("assistant", answer["answer"]))
        for message in messages:
            with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
                st.write(message.content)
        st.session_state.messages.extend(messages)

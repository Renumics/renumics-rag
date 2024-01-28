import dataclasses
from typing import Any, Dict, List, Literal, get_args

import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_core.runnables import Runnable

from assistant import (
    format_doc,
    get_chromadb,
    get_embeddings_model,
    get_llm,
    get_rag_chain,
    get_retriever,
    question_as_doc,
)
from assistant.settings import settings
from assistant.types import (
    MODEL_TYPES,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RETRIEVER_SEARCH_TYPES,
    ModelType,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
)

Role = Literal["user", "assistant", "source"]


AVATARS: Dict[Role, Any] = {"user": "🧐", "assistant": "🤖", "source": "📚"}


@dataclasses.dataclass
class Message:
    role: Role
    content: str


st.set_page_config(page_title="F1 RAG Demo", page_icon="🏎️", layout="wide")
st.title("F1 RAG Demo 🤖➕📚❤️🏎️")

st.header("Chat with the F1 docs")


if "messages" not in st.session_state.keys():
    st.session_state.messages = [Message("assistant", "Ask me a question about F1")]


@st.cache_resource(show_spinner=False)
def _get_rag_chain(
    llm_type: ModelType,
    llm_name: str,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
    search_type: RetrieverSearchType,
    score_threshold: float,
    fetch_k: int,
    lambda_mult: float,
    embeddings_model_type: ModelType,
    embeddings_model_name: str,
) -> Runnable:
    print(
        "Load chain",
        llm_type,
        llm_name,
        relevance_score_fn,
        k,
        search_type,
        score_threshold,
        fetch_k,
        lambda_mult,
        embeddings_model_type,
        embeddings_model_name,
    )
    with st.spinner(text="Loading database and LLMs..."):
        embeddings_model = get_embeddings_model(
            embeddings_model_name, embeddings_model_type
        )
        llm = get_llm(llm_name, llm_type)
        vectorstore = get_chromadb(
            persist_directory=settings.docs_db_directory,
            embeddings_model=embeddings_model,
            collection_name=settings.docs_db_collection,
            relevance_score_fn=relevance_score_fn,
        )
        retriever = get_retriever(
            vectorstore, k, search_type, score_threshold, fetch_k, lambda_mult
        )
        chain = get_rag_chain(retriever, llm)
        return chain


@st.cache_resource(show_spinner=False)
def get_questions_chromadb(
    embeddings_model_type: ModelType, embeddings_model_name: str
) -> Chroma:
    embeddings_model = get_embeddings_model(
        embeddings_model_name, embeddings_model_type
    )
    vectorstore = get_chromadb(
        persist_directory=settings.questions_db_directory,
        embeddings_model=embeddings_model,
        collection_name=settings.questions_db_collection,
    )
    return vectorstore


with st.sidebar:
    st.header("Settings")
    st.subheader("LLM")
    st.selectbox(
        "type",
        get_args(ModelType),
        get_args(ModelType).index(settings.llm_type),
        format_func=lambda x: MODEL_TYPES.get(x, x),
        key="llm_type",
    )
    st.text_input("name", value=settings.llm_name, key="llm_name")
    with st.expander("Advanced"):
        st.subheader("Retriever")
        st.selectbox(
            "Relevance score function",
            get_args(PredefinedRelevanceScoreFn),
            get_args(PredefinedRelevanceScoreFn).index(settings.relevance_score_fn),
            format_func=lambda x: PREDEFINED_RELEVANCE_SCORE_FNS.get(x, x),
            key="relevance_score_fn",
            help="Distance function in the embedding space "
            "([more](https://docs.trychroma.com/usage-guide#changing-the-distance-function))",
        )
        k = st.slider(
            "k",
            1,
            max(100, settings.k + 20),
            settings.k,
            key="k",
            help="Amount of documents to return",
        )
        search_type = st.selectbox(
            "Search type",
            get_args(RetrieverSearchType),
            get_args(RetrieverSearchType).index(settings.search_type),
            format_func=lambda x: RETRIEVER_SEARCH_TYPES.get(x, x),
            key="search_type",
            help="Type of search",
        )
        st.slider(
            "Score threshold",
            0.0,
            1.0,
            settings.score_threshold,
            key="score_threshold",
            help="Minimum relevance threshold",
            disabled=search_type != "similarity_score_threshold",
        )
        st.slider(
            "Fetch k",
            k,
            max(200, k * 2),
            max(settings.fetch_k, k + 10),
            key="fetch_k",
            help="Amount of documents to pass to MMR",
            disabled=search_type != "mmr",
        )
        st.slider(
            "MMR λ",
            0.0,
            1.0,
            settings.lambda_mult,
            key="lambda_mult",
            help="Diversity of results returned by MMR. 1 for minimum diversity and 0 for maximum.",
            disabled=search_type != "mmr",
        )
        st.subheader("Embeddings model")
        st.write(
            f"Be sure to replace the vectorstore at '{settings.docs_db_directory}' "
            f"with one indexed by the respective embeddings model and stored in "
            f"the '{settings.docs_db_collection}' collection."
        )
        st.selectbox(
            "type",
            get_args(ModelType),
            get_args(ModelType).index(settings.embeddings_model_type),
            format_func=lambda x: MODEL_TYPES.get(x, x),
            key="embeddings_model_type",
        )
        st.text_input(
            "name", value=settings.embeddings_model_name, key="embeddings_model_name"
        )


chain = _get_rag_chain(
    st.session_state.llm_type,
    st.session_state.llm_name,
    st.session_state.relevance_score_fn,
    st.session_state.k,
    st.session_state.search_type,
    st.session_state.score_threshold,
    st.session_state.fetch_k,
    st.session_state.lambda_mult,
    st.session_state.embeddings_model_type,
    st.session_state.embeddings_model_name,
)
questions_vectorstore = get_questions_chromadb(
    st.session_state.embeddings_model_type, st.session_state.embeddings_model_name
)

if question := st.chat_input("Your question"):
    st.session_state.messages.append(Message("user", question))

for message in st.session_state.messages:
    with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
        st.write(message.content)

if st.session_state.messages[-1].role == "user":
    with st.spinner("Thinking..."):
        rag_answer = chain.invoke(question)

        questions_vectorstore.add_documents(
            [question_as_doc(st.session_state.messages[-1].content, rag_answer)]
        )

        messages: List[Message] = []
        for doc in rag_answer["source_documents"]:
            messages.append(Message("source", format_doc(doc)))
        messages.append(Message("assistant", rag_answer["answer"]))
        for message in messages:
            with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
                st.write(message.content)
        st.session_state.messages.extend(messages)

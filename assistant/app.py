import dataclasses
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, get_args

import duckdb
import streamlit as st
import typer
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

try:
    from renumics import spotlight
except ImportError:
    spotlight = None  # type: ignore

from assistant import (
    get_chromadb,
    get_embeddings_model,
    get_embeddings_model_config,
    get_llm,
    get_llm_config,
    get_rag_chain,
    get_retriever,
    question_as_doc,
)
from assistant.settings import Settings, settings
from assistant.sql_rag import get_sql_chain
from assistant.types import (
    LLM,
    MODEL_TYPES,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RAG_MODES,
    RETRIEVER_SEARCH_TYPES,
    ModelType,
    PredefinedRelevanceScoreFn,
    RAGMode,
    RelevanceScoreFn,
    RetrieverSearchType,
)

app = typer.Typer()

Role = Literal["user", "assistant", "source", "query", "link"]


AVATARS: Dict[Role, Any] = {
    "user": "🧐",
    "assistant": "🤖",
    "source": "📚",
    "query": "🔃",
    "link": "🔗",
}


@dataclasses.dataclass
class Message:
    role: Role
    content: str


@dataclasses.dataclass
class SourceMessage(Message):
    role: Role
    content: str
    sources: List[str]


def hash_model(model: Union[Embeddings, LLM]) -> int:
    if isinstance(model, Embeddings):
        name, model_type = get_embeddings_model_config(model)
    else:
        name, model_type = get_llm_config(model)
    return hash(name) ^ hash(model_type)


HASH_FUNCS: Dict[Union[str, Type], Callable[[Any], Any]] = {
    AzureOpenAIEmbeddings: hash_model,
    OpenAIEmbeddings: hash_model,
    HuggingFaceEmbeddings: hash_model,
    AzureChatOpenAI: hash_model,
    ChatOpenAI: hash_model,
    HuggingFacePipeline: hash_model,
}


_get_llm = st.cache_resource(max_entries=1, show_spinner=False)(get_llm)
_get_embeddings_model = st.cache_resource(max_entries=1, show_spinner=False)(
    get_embeddings_model
)
_get_sql_chain = st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)(
    get_sql_chain
)


@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_rag_chain(
    llm: LLM,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
    search_type: RetrieverSearchType,
    score_threshold: float,
    fetch_k: int,
    lambda_mult: float,
    embeddings_model: Embeddings,
) -> Runnable:
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


@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def get_questions_chromadb(embeddings_model: Embeddings) -> Chroma:
    vectorstore = get_chromadb(
        persist_directory=settings.questions_db_directory,
        embeddings_model=embeddings_model,
        collection_name=settings.questions_db_collection,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_db_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


@st.cache_resource(show_spinner=False)
def get_or_create_spotlight_viewer() -> Any:
    if spotlight is None:
        return None
    viewers = spotlight.viewers()
    if viewers:
        for viewer in viewers[:-1]:
            viewer.close()
        return spotlight.viewers()[-1]
    return spotlight.show(no_browser=True, wait=False)


def st_settings(
    default_settings: Settings,
) -> None:
    st.header("Settings")
    st.radio(
        "RAG mode",
        get_args(RAGMode),
        get_args(RAGMode).index(default_settings.rag_mode),
        format_func=lambda x: RAG_MODES.get(x, x),
        key="rag_mode",
    )
    st.subheader("LLM")
    st.selectbox(
        "type",
        get_args(ModelType),
        get_args(ModelType).index(default_settings.llm_type),
        format_func=lambda x: MODEL_TYPES.get(x, x),
        key="llm_type",
    )
    st.text_input("name", value=default_settings.llm_name, key="llm_name")
    with st.expander("Advanced"):
        st.subheader("Retriever")
        st.selectbox(
            "Relevance score function",
            get_args(PredefinedRelevanceScoreFn),
            get_args(PredefinedRelevanceScoreFn).index(
                default_settings.relevance_score_fn
            ),
            format_func=lambda x: PREDEFINED_RELEVANCE_SCORE_FNS.get(x, x),
            key="relevance_score_fn",
            help="Distance function in the embedding space "
            "([more](https://docs.trychroma.com/usage-guide#changing-the-distance-function))",
        )
        k = st.slider(
            "k",
            1,
            max(100, default_settings.k + 20),
            default_settings.k,
            key="k",
            help="Amount of documents to return",
        )
        search_type = st.selectbox(
            "Search type",
            get_args(RetrieverSearchType),
            get_args(RetrieverSearchType).index(default_settings.search_type),
            format_func=lambda x: RETRIEVER_SEARCH_TYPES.get(x, x),
            key="search_type",
            help="Type of search",
        )
        st.slider(
            "Score threshold",
            0.0,
            1.0,
            default_settings.score_threshold,
            key="score_threshold",
            help="Minimum relevance threshold",
            disabled=search_type != "similarity_score_threshold",
        )
        st.slider(
            "Fetch k",
            k,
            max(200, k * 2),
            max(default_settings.fetch_k, k + 10),
            key="fetch_k",
            help="Amount of documents to pass to MMR",
            disabled=search_type != "mmr",
        )
        st.slider(
            "MMR λ",
            0.0,
            1.0,
            default_settings.lambda_mult,
            key="lambda_mult",
            help="Diversity of results returned by MMR. 1 for minimum diversity and 0 for maximum.",
            disabled=search_type != "mmr",
        )
        st.subheader("Embeddings model")
        st.write(
            f"Be sure to replace the vectorstore at '{default_settings.docs_db_directory}' "
            f"with one indexed by the respective embeddings model and stored in "
            f"the '{default_settings.docs_db_collection}' collection."
        )
        st.selectbox(
            "type",
            get_args(ModelType),
            get_args(ModelType).index(default_settings.embeddings_model_type),
            format_func=lambda x: MODEL_TYPES.get(x, x),
            key="embeddings_model_type",
        )
        st.text_input(
            "name",
            value=default_settings.embeddings_model_name,
            key="embeddings_model_name",
        )


def st_chat_messages(messages: List[Message]) -> None:
    for message in messages:
        with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
            if message.role == "source":
                with st.expander("Sources"):
                    for source in message.sources:  # type: ignore
                        st.write(source)
            elif message.role == "query":
                st.write(f"```sql\n{message.content}\n```")
            else:
                st.write(message.content)


def st_docs_chat(chain: Runnable, questions_vectorstore: Chroma) -> None:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [Message("assistant", "Ask me a question")]

    if question := st.chat_input("Your question"):
        st.session_state.messages.append(Message("user", question))

    st_chat_messages(st.session_state.messages)

    if st.session_state.messages[-1].role == "user":
        with st.spinner("Thinking..."):
            rag_answer = chain.invoke(question)

            questions_vectorstore.add_documents(
                [question_as_doc(st.session_state.messages[-1].content, rag_answer)]
            )
            questions_vectorstore.persist()

            messages: List[Message] = []
            sources: List[str] = []
            for doc in rag_answer["source_documents"]:
                sources.append(f"Content: {doc.page_content}")
                sources.append(f"Source: \"{doc.metadata['source']}\"")
            messages.append(SourceMessage("source", "Sources", sources))
            # for doc in rag_answer["source_documents"]:
            #     messages.append(Message("source", format_doc(doc)))
            messages.append(Message("assistant", rag_answer["answer"]))
            st_chat_messages(messages)
            st.session_state.messages.extend(messages)


def st_sql_chat(chain: Runnable, viewer: Any) -> None:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [Message("assistant", "Ask me a question")]

    if question := st.chat_input("Your question"):
        st.session_state.messages.append(Message("user", question))

    st_chat_messages(st.session_state.messages)

    if st.session_state.messages[-1].role == "user":
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            try:
                query, explanation = answer.split("END_QUERY", 1)
                _, query = query.split("QUERY:")
            except ValueError:
                query = "⚠️Unparsable SQL query"
                explanation = answer
            else:
                query = query.strip().strip("`")
                if query.startswith("sql"):
                    query = query[3:]
                query = query.strip()
            messages = [Message("query", query), Message("assistant", explanation)]
        with st.spinner("Preparing visualization..."):
            if viewer is not None and not query.startswith("⚠️"):
                try:
                    df = duckdb.sql(query).df()
                    viewer.show(
                        df, no_browser=True, allow_filebrowsing=False, wait=False
                    )
                except Exception:
                    messages[0] = Message(
                        "query", f"⚠️Invalid SQL query:\n{messages[0].content}"
                    )
                else:
                    messages.insert(1, Message("link", f"Explore: {viewer}"))
        st_chat_messages(messages)
        st.session_state.messages.extend(messages)


def st_app(
    title: str = "RAG Demo",
    favicon: str = "🤖",
    image: Optional[str] = None,
    h1: str = "RAG Demo",
    h2: str = "Chat with your docs",
) -> None:
    st.set_page_config(
        page_title=title,
        page_icon=favicon,
        layout="wide",
        menu_items={
            # "Get Help": "https://github.com/Renumics/rag-demo",
            # "Report a bug": "https://github.com/Renumics/rag-demo/issues",
            # "About": "https://github.com/Renumics/rag-demo",
        },
    )
    with st.sidebar:
        st_settings(settings)

    if image:
        st.image(image)
    if h1:
        st.title(h1)
    if h2:
        st.header(h2, divider=True)

    # All variables used in `get_embeddings_model`, `_get_rag_chain` and
    # `get_questions_chromadb` should be set before, either with `st_settings` or fixed.
    if st.session_state.rag_mode == "docs":
        with st.spinner("Loading RAG database, models and chain..."):
            llm = _get_llm(st.session_state.llm_name, st.session_state.llm_type)
            embeddings_model = _get_embeddings_model(
                st.session_state.embeddings_model_name,
                st.session_state.embeddings_model_type,
            )
            chain = _get_rag_chain(
                llm,
                st.session_state.relevance_score_fn,
                st.session_state.k,
                st.session_state.search_type,
                st.session_state.score_threshold,
                st.session_state.fetch_k,
                st.session_state.lambda_mult,
                embeddings_model,
            )
            questions_vectorstore = get_questions_chromadb(embeddings_model)
        st_docs_chat(chain, questions_vectorstore)
    elif st.session_state.rag_mode == "sql":
        with st.spinner("Loading llm, chain and explorer..."):
            llm = _get_llm(st.session_state.llm_name, st.session_state.llm_type)
            chain = _get_sql_chain(llm)
            viewer = get_or_create_spotlight_viewer()
            if viewer is None:
                st.warning(
                    "To explore results, first install Spotlight: "
                    "`pip install renumics-spotlight`",
                    icon="⚠️",
                )
        st_sql_chat(chain, viewer)
    else:
        raise ValueError(f"Unknown RAG mode: '{st.session_state.rag_mode}'.")


app.command()(st_app)


if __name__ == "__main__":
    app()

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_args

import duckdb
import pandas as pd
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
from assistant.exploration import get_docs_questions_df
from assistant.settings import Settings, settings
from assistant.sql_rag import get_sql_chain
from assistant.types import (
    AVATARS,
    LLM,
    MODEL_TYPES,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RAG_MODES,
    RETRIEVER_SEARCH_TYPES,
    Message,
    ModelType,
    NestedMessage,
    PredefinedRelevanceScoreFn,
    RAGMode,
    RelevanceScoreFn,
    RetrieverSearchType,
)

app = typer.Typer()


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
        embeddings_model,
        settings.docs_db_directory,
        settings.docs_db_collection,
        relevance_score_fn,
    )
    retriever = get_retriever(
        vectorstore, k, search_type, score_threshold, fetch_k, lambda_mult
    )
    chain = get_rag_chain(retriever, llm)
    return chain


@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_questions_chromadb(embeddings_model: Embeddings) -> Chroma:
    vectorstore = get_chromadb(
        embeddings_model,
        settings.questions_db_directory,
        settings.questions_db_collection,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_or_create_spotlight_viewer(port: Optional[int] = None) -> Any:
    try:
        from renumics import spotlight
        from renumics.spotlight import dtypes as spotlight_dtypes
    except ImportError:
        return None
    viewers = spotlight.viewers()
    if port is None:
        if viewers:
            return spotlight.viewers()[-1]
    else:
        for viewer in viewers:
            if viewer.port == port:
                return viewer
    return spotlight.show(
        None,
        port="auto" if port is None else port,
        no_browser=True,
        dtype={
            "used_by_questions": spotlight_dtypes.SequenceDType(
                spotlight_dtypes.str_dtype
            )
        },
        wait=False,
    )


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
            if isinstance(message, NestedMessage):
                with st.expander(message.content):
                    for content in message.subcontents:
                        st.write(content)
            elif message.role == "query":
                st.write(f"```sql\n{message.content}\n```")
            else:
                st.write(message.content)


def st_chat(on_question: Callable[[str], List[Message]]) -> None:
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [Message("assistant", "Ask me a question")]

    if question := st.chat_input("Your question"):
        st.session_state.messages.append(Message("user", question))

    st_chat_messages(st.session_state.messages)

    if st.session_state.messages[-1].role == "user":
        with st.spinner("Thinking..."):
            messages = on_question(st.session_state.messages[-1].content)
            st_chat_messages(messages)
            st.session_state.messages.extend(messages)


def init_docs_rag() -> Tuple[Callable[[str], List[Message]], bool, Callable[[], None]]:
    embeddings_model = _get_embeddings_model(
        st.session_state.embeddings_model_name,
        st.session_state.embeddings_model_type,
        device=settings.device,
        trust_remote_code=settings.trust_remote_code,
    )
    llm = _get_llm(
        st.session_state.llm_name,
        st.session_state.llm_type,
        device=settings.device,
        trust_remote_code=settings.trust_remote_code,
        torch_dtype=settings.torch_dtype,
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
    questions_vectorstore = _get_questions_chromadb(embeddings_model)

    def on_question(question: str) -> List[Message]:
        rag_answer: dict = chain.invoke(question)

        questions_vectorstore.add_documents([question_as_doc(question, rag_answer)])
        questions_vectorstore.persist()

        messages: List[Message] = []
        sources: List[str] = []
        for doc in rag_answer["source_documents"]:
            sources.append(f"**Content**: {doc.page_content}")
            sources.append(f"**Source**: \"{doc.metadata['source']}\"")
        messages.append(NestedMessage("source", "Sources", sources))
        messages.append(Message("assistant", rag_answer["answer"]))
        return messages

    viewer = get_or_create_spotlight_viewer()

    def explore() -> None:
        df = get_docs_questions_df(
            settings.docs_db_directory,
            settings.docs_db_collection,
            settings.questions_db_directory,
            settings.questions_db_collection,
        )
        viewer.show(df, wait=False)

    return on_question, viewer is not None, explore


def init_sql_rag() -> Tuple[Callable[[str], List[Message]], bool, Callable[[], None]]:
    llm = _get_llm(
        st.session_state.llm_name,
        st.session_state.llm_type,
        device=settings.device,
        trust_remote_code=settings.trust_remote_code,
        torch_dtype=settings.torch_dtype,
    )
    chain = _get_sql_chain(llm)

    viewer_table = get_or_create_spotlight_viewer(5000)
    viewer_ts = get_or_create_spotlight_viewer(5001)

    def on_question(question: str) -> List[Message]:
        answer: str = chain.invoke(question)
        try:
            prefix, query = answer.split("```sql", 1)
            query, explanation = query.split("```", 1)
        except ValueError:
            query = "⚠️Unparsable SQL query"
            explanation = answer
        else:
            explanation = prefix + explanation
            assert "dataset" in query
            query = query.strip().replace(
                "dataset", "'" + str(settings.sql_dataset_path) + "'"
            )
            try:
                df = duckdb.sql(query).df()
            except Exception:
                query = f"⚠️Invalid SQL query:\n{query}"
            else:
                viewer_table.show(
                    df, no_browser=True, allow_filebrowsing=False, wait=False
                )
                data = {}
                for column in df:
                    if column == "_timestamp":
                        continue
                    if not pd.api.types.is_numeric_dtype(df.dtypes[column]):  # type: ignore
                        continue
                    data[column] = [df[column].values]
                st.dataframe(pd.DataFrame(data))
                viewer_ts.show(
                    pd.DataFrame(data),
                    no_browser=True,
                    allow_filebrowsing=False,
                    wait=False,
                    dtype={column: "sequence1d" for column in data},
                )
        messages = [Message("query", query)]
        if explanation:
            messages.append(Message("assistant", explanation))
        return messages

    def explore() -> None:
        viewer_table.open_browser()
        viewer_ts.open_browser()

    return on_question, viewer_table is not None, explore


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
            # "Get Help": "https://github.com/Renumics/renumics-rag",
            # "Report a bug": "https://github.com/Renumics/renumics-rag/issues",
            # "About": "https://github.com/Renumics/renumics-rag",
        },
    )
    with st.sidebar:
        sidebar_container = st.container()
        st_settings(settings)

    col1, col2 = st.columns([7, 1])
    with col1:
        if h1:
            st.title(h1)
        if h2:
            st.header(h2)
    with col2:
        if image:
            st.image(image)
    st.divider()

    # All variables used here should be set before, either with `st_settings` or fixed.
    with st.spinner("Loading LLMs, chains and databases..."):
        if st.session_state.rag_mode == "docs":
            on_question, can_explore, explore = init_docs_rag()
        elif st.session_state.rag_mode == "sql":
            on_question, can_explore, explore = init_sql_rag()
        else:
            raise ValueError

    with sidebar_container:
        if can_explore:
            st.button(
                "Explore",
                help="Explore documents and questions interactivaly in Renumics Spotlight.",
                on_click=explore,
                type="primary",
                disabled=not can_explore,
            )
        else:
            st.warning(
                "Install [Renumics Spotlight](https://github.com/Renumics/spotlight) "
                "to explore vectorstores interactively: "
                "`pip install renumics-rag[exploration]` or `pip install renumics-spotlight`",
                icon="⚠️",
            )

    st_chat(on_question)


app.command()(st_app)


if __name__ == "__main__":
    app()

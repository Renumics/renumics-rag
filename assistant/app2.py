from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Type, Union, get_args

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
)
from assistant.settings import Settings, settings
from assistant.types import (
    AVATARS,
    LLM,
    PREDEFINED_RELEVANCE_SCORE_FNS,
    RETRIEVER_SEARCH_TYPES,
    Message,
    NestedMessage,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
)

Mode = Literal["Chat", "RAG"]
LLMName = Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]


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


@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_rag_chain(
    llm: LLM,
    docs_db_directory: str,
    relevance_score_fn: RelevanceScoreFn,
    k: int,
    search_type: RetrieverSearchType,
    score_threshold: float,
    fetch_k: int,
    lambda_mult: float,
    embeddings_model: Embeddings,
) -> Runnable:
    print(docs_db_directory)
    vectorstore = get_chromadb(
        embeddings_model,
        Path(docs_db_directory),
        settings.docs_db_collection,
        relevance_score_fn,
    )
    retriever = get_retriever(
        vectorstore, k, search_type, score_threshold, fetch_k, lambda_mult
    )
    chain = get_rag_chain(retriever, llm)
    return chain


@st.cache_resource(show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_chat_chain(llm: LLM) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm
    return chain


def st_settings(default_settings: Settings) -> None:
    st.header("Settings")
    st.radio("mode", get_args(Mode), key="mode", horizontal=True)
    st.selectbox(
        "LLM",
        get_args(LLMName),
        get_args(LLMName).index(default_settings.llm_name),
        key="llm_name",
        help="OpenAI model to use for text generation.",
    )
    if st.session_state.mode == "Chat":

        def end_chat() -> None:
            st.session_state.messages[st.session_state.mode].extend(
                [
                    Message("end", "Chat ended"),
                    Message("assistant", "Ask me a question"),
                ]
            )

        st.button("End chat", on_click=end_chat)
    elif st.session_state.mode == "RAG":
        st.selectbox(
            "Vectorstore",
            ["Fulterer", "Hensoldt"],
            key="docs_db",
            help="Indexed vectorstore datatbase to use in RAG context.",
        )
        with st.expander("Advanced RAG settings"):
            st.selectbox(
                "Relevance score function",
                get_args(PredefinedRelevanceScoreFn),
                get_args(PredefinedRelevanceScoreFn).index(
                    default_settings.relevance_score_fn
                ),
                format_func=lambda x: PREDEFINED_RELEVANCE_SCORE_FNS.get(x, x),
                key="relevance_score_fn",
                help="Distance function in the embedding space",
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
            st.text_input(
                "Embeddings model",
                value=default_settings.embeddings_model_name,
                key="embeddings_model_name",
                help="OpenAI model to use for embedding extraction.",
                disabled=True,
            )


def st_chat_messages(messages: List[Message]) -> None:
    for message in messages:
        with st.chat_message(message.role, avatar=AVATARS.get(message.role)):
            if isinstance(message, NestedMessage):
                with st.expander(message.content):
                    for content in message.subcontents:
                        st.write(content)
            else:
                st.write(message.content)


def st_chat(on_question: Callable[[str], List[Message]]) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if st.session_state.mode not in st.session_state.messages:
        st.session_state.messages[st.session_state.mode] = [
            Message("assistant", "Ask me a question")
        ]
    messages = st.session_state.messages[st.session_state.mode]

    if question := st.chat_input("Your question"):
        messages.append(Message("user", question))

    st_chat_messages(messages)

    if messages[-1].role == "user":
        with st.spinner("Thinking..."):
            answer_messages = on_question(messages[-1].content)
            st_chat_messages(answer_messages)
            messages.extend(answer_messages)


def st_app() -> None:
    st.set_page_config(page_title="LLM Chat", page_icon="🤖")

    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"

    if st.session_state.mode == "Chat":
        st.header("Chat with LLM")
    elif st.session_state.mode == "RAG":
        st.header("Chat with documents")

    with st.sidebar:
        st_settings(settings)

    with st.spinner("Loading models..."):
        if st.session_state.mode == "Chat":
            llm = _get_llm(st.session_state.llm_name, settings.llm_type)
            chain = _get_chat_chain(llm)

            def on_question(question: str) -> List[Message]:
                start_index = next(
                    (
                        i
                        for i, message in enumerate(
                            reversed(st.session_state.messages[st.session_state.mode])
                        )
                        if message.role == "end"
                    ),
                    0,
                )

                messages: List[BaseMessage] = []
                for message in st.session_state.messages[st.session_state.mode][
                    -start_index:
                ]:
                    if message.role == "user":
                        messages.append(HumanMessage(content=message.content))
                    elif message.role == "assistant":
                        messages.append(AIMessage(content=message.content))
                messages.append(HumanMessage(content=question))
                chat_answer = chain.invoke({"messages": messages})
                return [Message("assistant", chat_answer.content)]

        elif st.session_state.mode == "RAG":
            embeddings_model = _get_embeddings_model(
                st.session_state.embeddings_model_name, settings.embeddings_model_type
            )
            llm = _get_llm(st.session_state.llm_name, settings.llm_type)
            if st.session_state.docs_db == "Fulterer":
                docs_db_directory = "./db-docs-fulterer"
            if st.session_state.docs_db == "Hensoldt":
                docs_db_directory = "./db-docs-hensoldt"
            chain = _get_rag_chain(
                llm,
                docs_db_directory,
                st.session_state.relevance_score_fn,
                st.session_state.k,
                st.session_state.search_type,
                st.session_state.score_threshold,
                st.session_state.fetch_k,
                st.session_state.lambda_mult,
                embeddings_model,
            )

            def on_question(question: str) -> List[Message]:
                rag_answer = chain.invoke(question)

                messages: List[Message] = []
                sources: List[str] = []
                for doc in rag_answer["source_documents"]:
                    sources.append(f"**Content**: {doc.page_content}")
                    sources.append(f"**Source**: \"{doc.metadata['source']}\"")
                messages.append(NestedMessage("source", "Sources", sources))
                messages.append(Message("assistant", rag_answer["answer"]))
                return messages

        else:
            raise ValueError("Invalid mode: '{st.session_state.mode}'.")

    st_chat(on_question)


if __name__ == "__main__":
    st_app()

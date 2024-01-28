import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast, get_args

import chromadb
import chromadb.api.types
import dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

from .exceptions import HFImportError
from .settings import guess_model_type, settings
from .types import (
    LLM,
    ModelType,
    PredefinedRelevanceScoreFn,
    RelevanceScoreFn,
    RetrieverSearchType,
)

dotenv.load_dotenv(override=True)

if sys.platform == "linux":
    # For Linux, `sqlite3` can be outdated, so use the installed Python package.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def parse_model_name(name: str) -> Tuple[str, ModelType]:
    if ":" in name:
        model_type, name = name.split(":", 1)
        assert model_type in get_args(ModelType)
    else:
        model_type = guess_model_type()
    model_type = cast(ModelType, model_type)
    return name, model_type


def get_hf_embeddings_model(name: str) -> HuggingFaceEmbeddings:
    try:
        import sentence_transformers  # noqa: F401
        import torch
    except ImportError as e:
        raise HFImportError() from e
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encode_kwargs = {"normalize_embeddings": True}
    mode_kwargs = {"device": device}
    return HuggingFaceEmbeddings(
        model_name=name, encode_kwargs=encode_kwargs, model_kwargs=mode_kwargs
    )


def get_embeddings_model(name: str, model_type: ModelType) -> Embeddings:
    if model_type == "azure":
        return AzureOpenAIEmbeddings(azure_deployment=name)
    if model_type == "openai":
        return OpenAIEmbeddings(model=name)
    if model_type == "hf":
        return get_hf_embeddings_model(name)
    raise TypeError(f"Unknown model type '{model_type}'.")


def get_embeddings_model_config(embeddings_model: Embeddings) -> Tuple[str, ModelType]:
    if isinstance(embeddings_model, AzureOpenAIEmbeddings):
        assert embeddings_model.deployment is not None
        return embeddings_model.deployment, "azure"
    if isinstance(embeddings_model, OpenAIEmbeddings):
        return embeddings_model.model, "openai"
    if isinstance(embeddings_model, HuggingFaceEmbeddings):
        return embeddings_model.model_name, "hf"
    raise TypeError("Unknown model type.")


def get_hf_llm(name: str) -> HuggingFacePipeline:
    """
    [Pipeline tasks](https://huggingface.co/docs/transformers/v4.37.1/en/main_classes/pipelines#transformers.pipeline.task):
    - "conversational",
    - "document-question-answering",
    - ["question-answering"](https://huggingface.co/models?pipeline_tag=question-answering),
    - "summarization",
    - "table-question-answering",
    - ["text2text-generation"](https://huggingface.co/models?pipeline_tag=text2text-generation)
    - "text-generation"
    """
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise HFImportError() from e
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = pipeline(model=name, device=device)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
    # tokenizer = AutoTokenizer.from_pretrained(name)
    # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    # base_model = AutoModelForCausalLM.from_pretrained(  # AutoModelForCausalLM.from_pretrained( ##AutoModel
    #     name,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     quantization_config=quantization_config,
    # )
    # pipe = pipeline(
    #     "text-generation",
    #     model=base_model,
    #     tokenizer=tokenizer,
    #     max_length=256,
    #     temperature=0.0,
    #     top_p=0.95,
    #     repetition_penalty=1.2,
    # )
    # local_llm = HuggingFacePipeline(pipeline=pipe)
    # return local_llm  # type: ignore


def get_llm(name: str, model_type: ModelType) -> LLM:
    if model_type == "azure":
        return AzureChatOpenAI(azure_deployment=name, temperature=0.0)
    if model_type == "openai":
        return ChatOpenAI(model=name, temperature=0.0)
    if model_type == "hf":
        return get_hf_llm(name)
    raise TypeError("Unknown model type.")


def get_chromadb(
    persist_directory: Path,
    embeddings_model: Embeddings,
    collection_name: str,
    relevance_score_fn: RelevanceScoreFn = "l2",
) -> Chroma:
    """
    https://docs.trychroma.com/usage-guide#changing-the-distance-function
    """
    model_name, model_type = get_embeddings_model_config(embeddings_model)
    if persist_directory.exists():
        client_settings = chromadb.Settings(
            is_persistent=True, persist_directory=str(settings.docs_db_directory)
        )
        client = chromadb.Client(client_settings)
        try:
            collection = client.get_collection(collection_name, embedding_function=None)
        except ValueError:
            ...  # Collection doesn't exist, so nothing to check.
        else:
            if collection.metadata:
                try:
                    collection_model_type = collection.metadata["model_type"]
                except KeyError:
                    ...  # Model type isn't defined on the collection, do not check.
                else:
                    if model_type != collection_model_type and {
                        model_type,
                        collection_model_type,
                    } != {"azure", "openai"}:
                        raise RuntimeError(
                            f"Given embeddings model type '{model_type}' doesn't "
                            f"match with the embeddings model type "
                            f"'{collection.metadata['model_type']}' of the "
                            f"collection '{collection_name}' of the database."
                        )
                if (
                    "model_name" in collection.metadata
                    and collection.metadata["model_name"] != model_name
                ):
                    raise RuntimeError(
                        f"Given embeddings model name '{model_name}' doesn't "
                        f"match with the embeddings model name "
                        f"'{collection.metadata['model_name']}' of the "
                        f"collection '{collection_name}' of the database."
                    )

    collection_metadata = {"model_name": model_name, "model_type": model_type}
    if isinstance(relevance_score_fn, str):
        assert relevance_score_fn in get_args(PredefinedRelevanceScoreFn)
        collection_metadata["hnsw:space"] = relevance_score_fn
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_model,
            persist_directory=str(persist_directory),
            collection_metadata=collection_metadata,
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=str(persist_directory),
        collection_metadata=collection_metadata,
        relevance_score_fn=relevance_score_fn,
    )


def get_retriever(
    vectorstore: VectorStore,
    k: Optional[int] = None,
    search_type: RetrieverSearchType = "similarity",
    score_threshold: Optional[float] = None,
    fetch_k: Optional[int] = None,
    lambda_mult: Optional[float] = None,
) -> VectorStoreRetriever:
    kwargs: Dict[str, Any] = {"search_type": search_type}
    search_kwargs: Dict[str, Any] = {}
    if k is not None:
        search_kwargs["k"] = k
    if search_type == "similarity_score_threshold":
        assert score_threshold is not None
        search_kwargs["score_threshold"] = score_threshold
    elif search_type == "mmr":
        if fetch_k is not None:
            search_kwargs["fetch_k"] = fetch_k
        if lambda_mult is not None:
            search_kwargs["lambda_mult"] = lambda_mult
    kwargs["search_kwargs"] = search_kwargs
    return vectorstore.as_retriever(**kwargs)


def format_doc(doc: Document) -> str:
    return f"Content: {doc.page_content}\nSource: {doc.metadata['source']}"


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata['source']}" for doc in docs
    )


def get_rag_chain(retriever: VectorStoreRetriever, llm: LLM) -> Runnable:
    template = """You are an assistant for question-answering tasks on Formula One (F1) documentation.
Given the following extracted parts of a long document and a question, create a final answer with used references (named "sources").
Keep the answer concise. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return used sources in your answer.

QUESTION: {question}
=========
{source_documents}
=========
FINAL ANSWER: """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            source_documents=(lambda x: format_docs(x["source_documents"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {  # type: ignore
            "source_documents": retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_with_source


def stable_hash(doc: Document) -> str:
    """
    Stable hash document based on its metadata.
    """
    return hashlib.sha1(json.dumps(doc.metadata, sort_keys=True).encode()).hexdigest()


def question_as_doc(question: str, rag_answer: Dict[str, Any]) -> Document:
    return Document(
        page_content=question,
        metadata={
            "answer": rag_answer["answer"],
            "sources": ",".join(map(stable_hash, rag_answer["source_documents"])),
        },
    )

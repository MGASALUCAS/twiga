import logging
from typing import List, Literal

from app.tools.utils.ChromaDBLoader import ChromaDBLoader
from app.tools.utils.groq_requests import async_groq_request
from app.tools.utils.models import Metadata, RetrievedDocSchema, ChunkSchema
from app.tools.utils.openai_requests import async_openai_request
from app.tools.utils.prompt_templates import REWRITE_QUERY_PROMPT

"""
This is the modules file, which will contain the modular components that can be used by the RAG pipelines I build.
"""

logger = logging.getLogger(__name__)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


async def query_rewriter(
    query: str,
    llm: Literal["openai", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
) -> str:
    """Enhances a user query by rewriting it in better English and adding more detail."""
    try:
        messages = [
            {
                "role": REWRITE_QUERY_PROMPT.role,
                "content": REWRITE_QUERY_PROMPT.content,
            },
            {"role": "user", "content": f"Query: ({query})"},
        ]

        # Send the prompt to API using a suitable engine
        if llm == "openai":
            res = await async_openai_request(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                max_tokens=80,  # Adjust based on the expected length of the enhanced query
                n=1,  # Number of completions to generate (default is 1)
                stop=None,  # Optional stopping character or sequence
            )
        else:
            res = await async_groq_request(
                llm=llm,
                verbose=False,
                messages=messages,
                max_tokens=80,
            )

        # Extract the enhanced query text from the response
        enhanced_query = res.choices[0].message.content

        return enhanced_query

    except Exception as e:
        logger.error(f"An error occurred when rewriting the query: {e}")
        return query  # Return the original query in case of an error


async def chromadb_retriever(
    model_class: ChromaDBLoader,
    retrieval_msg: str,
    size: int,
    doc_type: Literal["Content", "Exercise"],
    book_title: str = "Geography for Secondary Schools Student's Book Form Two",
    verbose: bool = False,
) -> List[RetrievedDocSchema]:

    # Search for documents
    res = model_class.search(
        query=retrieval_msg, n_results=size, where={"doc_type": doc_type}
    )

    documents = res.get("documents")[0]
    metadatas = res.get("metadatas")[0]
    ids = res.get("ids")[0]

    docs: List[RetrievedDocSchema] = []
    for doc, metadata, id in zip(documents, metadatas, ids):
        for key, value in metadata.items():
            if value == "":
                metadata[key] = None
        md: Metadata = Metadata(**metadata)
        chunk: ChunkSchema = ChunkSchema(chunk=doc, metadata=md)
        retrieved_doc: RetrievedDocSchema = RetrievedDocSchema(
            retrieval_type="dense", id=id, source=chunk
        )
        docs.append(retrieved_doc)

    # This prints out the response in a pretty format if the caller desires
    if verbose:
        for i, document in enumerate(docs):
            print(f"------Document {i}-----")
            print(document.source.chunk)

    return docs


def rerank(
    eval_query: str,
    documents: List[RetrievedDocSchema],
    num_results: int,
    verbose: bool = False,
) -> List[RetrievedDocSchema]:

    # # Extract text content from Document objects and convert to strings
    # document_texts = [doc.source.chunk for doc in documents]
    # query_text = eval_query

    # # Create pairs as strings
    # pairs = [[query_text, doc_text] for doc_text in document_texts]
    # # Predict scores for pairs
    # scores = cross_encoder.predict(pairs)

    # # Print scores (these are pretty useless right now, some small modifications would make the logs much better)
    # if verbose:
    #     logger.info("Scores:")
    #     for score in scores:
    #         logger.info(score)

    # # Combine documents with their scores
    # doc_scores = list(zip(documents, scores))

    # # Sort documents by scores in descending order
    # doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    # # Extract the sorted documents
    # sorted_documents = [doc for doc, _ in doc_scores_sorted]

    # Return the top 'num_results' documents
    # return sorted_documents[:num_results]
    return None

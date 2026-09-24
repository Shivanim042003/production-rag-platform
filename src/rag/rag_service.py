import logging
import time
from pathlib import Path
from typing import Any

from rag.bm25 import BM25Retriever
from rag.cross_encoder import CrossEncoderScorer
from rag.embeddings import EmbeddingModel
from rag.graph import build_rag_graph
from rag.graph_nodes import (
    GenerateNode,
    GradeNode,
    GroundingNode,
    QueryTransformNode,
    RerankNode,
    RetrieveNode,
)
from rag.grader import RelevanceGrader
from rag.grounding import GroundingChecker
from rag.hybrid import HybridRetriever
from rag.ingestion import ingest_txt
from rag.local_grader import LocalRelevanceScorer
from rag.observability import RAGTrace
from rag.qwen_generator import QwenLocalGenerator
from rag.qwen_grounding import QwenGroundingScorer
from rag.query_transformer import IdentityQueryTransformer
from rag.reranker import Reranker
from rag.vector_store import FaissVectorStore


logger = logging.getLogger("rag.service")


DATA_PATH = Path("benchmarks/data")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CHUNK_SIZE = 35
CHUNK_OVERLAP = 5


class RAGService:
    """Application service for the production RAG pipeline."""

    def __init__(self) -> None:
        self.chunks = self._load_chunks()

        self.embedding_model = EmbeddingModel()

        embeddings = self.embedding_model.embed_documents(
            self.chunks
        )

        self.vector_store = FaissVectorStore(
            dimension=self.embedding_model.dimension
        )

        self.vector_store.add(
            self.chunks,
            embeddings,
        )

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        self.hybrid_retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
        )

        self.cross_encoder = CrossEncoderScorer()

        self.reranker = Reranker(
            scorer=self.cross_encoder,
        )

        self.relevance_scorer = LocalRelevanceScorer(
            scorer=self.cross_encoder,
            threshold=0.0,
        )

        self.relevance_grader = RelevanceGrader(
            scorer=self.relevance_scorer,
        )

        self.generator = QwenLocalGenerator(
            model_name=MODEL_NAME,
            max_new_tokens=128,
        )

        self.grounding_scorer = QwenGroundingScorer(
            model_name=MODEL_NAME,
        )

        self.grounding_checker = GroundingChecker(
            scorer=self.grounding_scorer,
        )

    def _load_chunks(self):
        chunks = []

        for path in sorted(DATA_PATH.glob("*.txt")):
            chunks.extend(
                ingest_txt(
                    path,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )
            )

        if not chunks:
            raise RuntimeError(
                "No RAG documents were found."
            )

        return chunks

    def _build_graph(
        self,
        trace: RAGTrace,
    ):
        retrieve_node = RetrieveNode(
            retriever=self.hybrid_retriever,
            top_k=10,
            dense_k=10,
            bm25_k=10,
            trace=trace,
        )

        rerank_node = RerankNode(
            reranker=self.reranker,
            top_k=5,
            trace=trace,
        )

        grade_node = GradeNode(
            grader=self.relevance_grader,
            min_relevant=1,
            trace=trace,
        )

        generate_node = GenerateNode(
            generator=self.generator,
            trace=trace,
        )

        grounding_node = GroundingNode(
            checker=self.grounding_checker,
            trace=trace,
        )

        query_transform_node = QueryTransformNode(
            transformer=IdentityQueryTransformer(),
            trace=trace,
        )

        return build_rag_graph(
            retrieve_node=retrieve_node,
            rerank_node=rerank_node,
            grade_node=grade_node,
            generate_node=generate_node,
            grounding_node=grounding_node,
            max_retries=2,
            query_transform_node=query_transform_node,
        )

    def query(
        self,
        question: str,
    ) -> dict[str, Any]:
        """Run a question through the RAG graph."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        trace = RAGTrace(
            query=question,
        )

        start_time = time.perf_counter()

        graph = self._build_graph(
            trace=trace,
        )

        initial_state = {
            "query": question,
            "transformed_queries": [],
            "candidates": [],
            "reranked_results": [],
            "graded_results": [],
            "context": [],
            "answer": None,
            "grounded": None,
            "retry_count": 0,
            "sufficient_context": False,
        }

        final_state = graph.invoke(
            initial_state
        )

        total_duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        trace.finish(
            total_duration_ms=total_duration_ms,
        )

        reranked_results = final_state.get(
            "reranked_results",
            [],
        )

        graded_results = final_state.get(
            "graded_results",
            [],
        )

        relevant_count = sum(
            1
            for _, grade in graded_results
            if grade.relevant
        )

        trace.metadata.update(
            {
                "candidate_count": len(
                    final_state.get(
                        "candidates",
                        [],
                    )
                ),
                "reranked_count": len(
                    reranked_results
                ),
                "relevant_context_count": relevant_count,
                "retry_count": final_state.get(
                    "retry_count",
                    0,
                ),
                "grounded": bool(
                    final_state.get(
                        "grounded",
                        False,
                    )
                ),
            }
        )

        logger.info(
            "RAG request completed",
            extra={
                "rag_trace": {
                    "query": trace.query,
                    "total_duration_ms": (
                        trace.total_duration_ms
                    ),
                    "stages": [
                        {
                            "name": stage.name,
                            "duration_ms": (
                                stage.duration_ms
                            ),
                        }
                        for stage in trace.stages
                    ],
                    "metadata": trace.metadata,
                }
            },
        )

        sources = [
            {
                "chunk_id": result.chunk.chunk_id,
                "score": float(result.score),
            }
            for result in reranked_results
        ]

        return {
            "query": question,
            "answer": final_state.get(
                "answer",
                "",
            ),
            "grounded": bool(
                final_state.get(
                    "grounded",
                    False,
                )
            ),
            "sources": sources,
        }
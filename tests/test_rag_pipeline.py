"""Tests for RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from app.models.llm import LLMClient
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = [
        {
            "score": 0.9,
            "chunk_text": "Section 302: Penalty for theft is imprisonment.",
            "doc_id": "BNS",
            "metadata": {"section": "302"},
        }
    ]
    return retriever


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = MagicMock(spec=LLMClient)
    llm.model_name = "gpt2"
    llm.generate.return_value = "The penalty for theft is imprisonment as stated in Section 302."
    return llm


def test_rag_pipeline_init(mock_retriever, mock_llm):
    """Test RAG pipeline initialization."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)

    assert pipeline.retriever == mock_retriever
    assert pipeline.llm == mock_llm


def test_rag_pipeline_answer(mock_retriever, mock_llm):
    """Test answering a question through the pipeline."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)

    result = pipeline.answer("What is the penalty for theft?")

    assert "answer" in result
    assert "context" in result
    assert "num_chunks" in result
    assert result["num_chunks"] == 1
    assert len(result["context"]) == 1

    # Verify retriever was called
    mock_retriever.retrieve.assert_called_once()
    # Verify LLM was called
    mock_llm.generate.assert_called_once()


def test_rag_pipeline_answer_empty_question(mock_retriever, mock_llm):
    """Test answering with empty question."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.answer("")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.answer("   ")


def test_rag_pipeline_answer_no_chunks(mock_retriever, mock_llm):
    """Test answering when no chunks are retrieved."""
    mock_retriever.retrieve.return_value = []

    pipeline = RAGPipeline(mock_retriever, mock_llm)

    result = pipeline.answer("Test question")

    assert result["answer"] == "Not found in the provided legal text."
    assert result["context"] == []
    assert result["num_chunks"] == 0
    # LLM should not be called if no chunks
    mock_llm.generate.assert_not_called()


def test_rag_pipeline_answer_custom_k(mock_retriever, mock_llm):
    """Test answering with custom k value."""
    mock_retriever.retrieve.return_value = [
        {"score": 0.9, "chunk_text": "Chunk 1", "doc_id": "doc1", "metadata": {}},
        {"score": 0.8, "chunk_text": "Chunk 2", "doc_id": "doc2", "metadata": {}},
    ]

    pipeline = RAGPipeline(mock_retriever, mock_llm)
    result = pipeline.answer("Test question", k=2)

    # Verify k was passed to retriever
    call_kwargs = mock_retriever.retrieve.call_args[1]
    assert call_kwargs["k"] == 2
    assert result["num_chunks"] == 2


def test_rag_pipeline_answer_min_score(mock_retriever, mock_llm):
    """Test answering with minimum score threshold."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    pipeline.answer("Test question", min_score=0.5)

    # Verify min_score was passed
    call_kwargs = mock_retriever.retrieve.call_args[1]
    assert call_kwargs["min_score"] == 0.5


def test_rag_pipeline_answer_simple_prompt(mock_retriever, mock_llm):
    """Test answering with simple prompt format."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    pipeline.answer("Test question", use_simple_prompt=True)

    # Verify LLM was called (prompt building happens internally)
    mock_llm.generate.assert_called_once()
    # Check that prompt contains the question
    prompt = mock_llm.generate.call_args[0][0]
    assert "Test question" in prompt


def test_rag_pipeline_answer_custom_tokens(mock_retriever, mock_llm):
    """Test answering with custom max_new_tokens."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    pipeline.answer("Test question", max_new_tokens=100)

    # Verify max_new_tokens was passed to LLM
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs["max_new_tokens"] == 100


def test_rag_pipeline_answer_custom_temperature(mock_retriever, mock_llm):
    """Test answering with custom temperature."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    pipeline.answer("Test question", temperature=0.7)

    # Verify temperature was passed to LLM
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs["temperature"] == 0.7


def test_rag_pipeline_answer_retrieval_error(mock_retriever, mock_llm):
    """Test error handling during retrieval."""
    mock_retriever.retrieve.side_effect = Exception("Retrieval error")

    pipeline = RAGPipeline(mock_retriever, mock_llm)

    with pytest.raises(Exception, match="Retrieval error"):
        pipeline.answer("Test question")


def test_rag_pipeline_answer_llm_error(mock_retriever, mock_llm):
    """Test error handling during LLM generation."""
    mock_llm.generate.side_effect = Exception("LLM error")

    pipeline = RAGPipeline(mock_retriever, mock_llm)

    with pytest.raises(Exception, match="LLM error"):
        pipeline.answer("Test question")


def test_rag_pipeline_answer_response_format(mock_retriever, mock_llm):
    """Test that response has correct format."""
    mock_retriever.retrieve.return_value = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk",
            "doc_id": "doc1",
            "metadata": {"section": "1"},
        }
    ]

    pipeline = RAGPipeline(mock_retriever, mock_llm)
    result = pipeline.answer("Test question")

    # Check response structure
    assert isinstance(result, dict)
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert "context" in result
    assert isinstance(result["context"], list)
    assert "num_chunks" in result
    assert isinstance(result["num_chunks"], int)

    # Check context structure
    assert len(result["context"]) > 0
    context_item = result["context"][0]
    assert "score" in context_item
    assert "chunk_text" in context_item
    assert "doc_id" in context_item
    assert "metadata" in context_item


def test_rag_pipeline_answer_batch(mock_retriever, mock_llm):
    """Test batch answering."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)

    questions = ["Question 1", "Question 2", "Question 3"]
    results = pipeline.answer_batch(questions)

    assert len(results) == 3
    assert all("answer" in r for r in results)
    assert all("context" in r for r in results)
    # Verify retriever was called for each question
    assert mock_retriever.retrieve.call_count == 3


def test_rag_pipeline_answer_batch_with_error(mock_retriever, mock_llm):
    """Test batch answering with one error."""
    # Make second call fail
    def side_effect(*args, **kwargs):
        if mock_retriever.retrieve.call_count == 2:
            raise Exception("Error on question 2")
        return [
            {
                "score": 0.9,
                "chunk_text": "Test chunk",
                "doc_id": "doc1",
                "metadata": {},
            }
        ]

    mock_retriever.retrieve.side_effect = side_effect

    pipeline = RAGPipeline(mock_retriever, mock_llm)

    questions = ["Question 1", "Question 2", "Question 3"]
    results = pipeline.answer_batch(questions)

    assert len(results) == 3
    # First and third should succeed
    assert "answer" in results[0]
    assert "answer" in results[2]
    # Second should have error message
    assert "Error processing question" in results[1]["answer"]


def test_rag_pipeline_repr(mock_retriever, mock_llm):
    """Test string representation."""
    pipeline = RAGPipeline(mock_retriever, mock_llm)
    repr_str = repr(pipeline)

    assert "RAGPipeline" in repr_str
    assert "gpt2" in repr_str


def test_rag_pipeline_integration_flow(mock_retriever, mock_llm):
    """Test complete integration flow."""
    # Setup mocks
    mock_retriever.retrieve.return_value = [
        {
            "score": 0.95,
            "chunk_text": "Section 302: Whoever commits theft shall be punished.",
            "doc_id": "BNS",
            "metadata": {"section": "302", "page": 45},
        },
        {
            "score": 0.85,
            "chunk_text": "Section 303: Additional provisions apply.",
            "doc_id": "BNS",
            "metadata": {"section": "303", "page": 46},
        },
    ]

    mock_llm.generate.return_value = (
        "According to Section 302, theft is punishable. "
        "Section 303 provides additional provisions."
    )

    pipeline = RAGPipeline(mock_retriever, mock_llm)
    result = pipeline.answer("What is the penalty for theft?")

    # Verify complete flow
    assert result["answer"] is not None
    assert len(result["context"]) == 2
    assert result["num_chunks"] == 2

    # Verify retriever was called with question
    assert mock_retriever.retrieve.called
    # Verify LLM was called with a prompt
    assert mock_llm.generate.called
    prompt = mock_llm.generate.call_args[0][0]
    assert "Section 302" in prompt
    assert "What is the penalty for theft?" in prompt


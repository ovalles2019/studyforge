from studyforge.rag import normalize_questions


def test_normalize_questions_dict_choices():
    raw = [
        {
            "question": "What stores embeddings?",
            "choices": {"A": "ChromaDB", "B": "PyMuPDF", "C": "Streamlit", "D": "vLLM"},
            "answer": "a",
            "explanation": "Vector store.",
            "source_page": 2,
        }
    ]
    out = normalize_questions(raw)
    assert len(out) == 1
    assert out[0]["answer"] == "A"
    assert out[0]["source_page"] == 2


def test_normalize_questions_list_choices_and_junk():
    raw = [
        {
            "question": "What is RAG?",
            "choices": ["Retrieval", "Training from scratch", "A GPU", "A PDF parser"],
            "answer": "A",
            "source_page": "1",
        },
        "not a question",
        {"choices": {"A": "x"}},
    ]
    out = normalize_questions(raw)
    assert len(out) == 1
    assert out[0]["choices"]["A"] == "Retrieval"
    assert out[0]["source_page"] == 1

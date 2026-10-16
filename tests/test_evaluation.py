from course_assistant.eval.answer_quality import evaluate_answer
from course_assistant.qa.schema import Answer, SourceRef
from course_assistant.retrieval.evidence import Evidence


class _Answerer:
    def answer(self, question, evidence):
        return Answer(
            answer="Speculative decoding uses a draft model to reduce latency.",
            sources=[
                SourceRef(
                    document="MBAX 6418 - Week 2.pptx",
                    page=13,
                    excerpt="A draft model proposes tokens.",
                )
            ],
        )


def test_evaluate_answer_records_correctness_review_support_and_timing():
    spec = {
        "question": "What is speculative decoding?",
        "doc": "Week 2",
        "page": 13,
        "kind": "text",
    }
    evidence = [
        Evidence(
            doc_id="d1",
            document="MBAX 6418 - Week 2.pptx",
            page=13,
            section="Speculative decoding",
            text="A draft model proposes tokens.",
        )
    ]

    result = evaluate_answer(spec, evidence, _Answerer())

    assert result["answer"].startswith("Speculative decoding")
    assert result["source_support"] is True
    assert result["answer_correct"] is None  # explicit manual-review field
    assert result["answer_seconds"] >= 0
    assert result["cited_documents"] == ["MBAX 6418 - Week 2.pptx"]
    assert result["cited_pages"] == [13]


def test_evaluate_answer_scores_a_supported_missing_information_response():
    class MissingAnswerer:
        def answer(self, question, evidence):
            return Answer(answer="The materials do not say.", missing_info=True)

    spec = {"question": "Unknown?", "doc": None, "page": None, "kind": "unanswerable"}
    result = evaluate_answer(spec, [], MissingAnswerer())

    assert result["source_support"] is True
    assert result["missing_info"] is True
    assert result["answer_correct"] is None

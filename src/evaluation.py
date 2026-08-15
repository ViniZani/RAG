from .models import (AnsweredQuestion, MinimalSource,
                     RagDataset, StudentSearchResults)
import json


def _overlap_ratio(retrieved: MinimalSource, correct: MinimalSource) -> float:
    """Calcula a fração do intervalo `correct` coberta por `retrieved`."""

    r_path = retrieved.file_path.replace(
        "data/raw/vllm-0.10.1/", "").replace("\\", "/")
    c_path = correct.file_path.replace(
        "data/raw/vllm-0.10.1/", "").replace("\\", "/")

    if r_path != c_path:
        return 0.0

    overlap_start = max(retrieved.first_character_index,
                        correct.first_character_index)
    overlap_end = min(retrieved.last_character_index,
                      correct.last_character_index)

    if overlap_end <= overlap_start:
        return 0.0

    intersection = overlap_end - overlap_start
    correct_len = correct.last_character_index - correct.first_character_index

    if correct_len <= 0:
        return 0.0

    return intersection / correct_len


def is_source_found(retrieved: MinimalSource, correct: MinimalSource) -> bool:
    """A source do gabarito é 'found' se o overlap for >= 5% (regra VI.1.1)."""
    return _overlap_ratio(retrieved, correct) >= 0.05


def recall_at_k(
    retrieved_sources: list[MinimalSource],
    correct_sources: list[MinimalSource],
) -> float:
    """Calcula recall@k para uma única pergunta.
    Args:
        retrieved_sources: Fontes recuperadas (já cortadas no k desejado).
        correct_sources: Fontes do gabarito.
    Returns:
        Proporção de fontes corretas que foram encontradas. Se não houver
        nenhuma fonte de gabarito, retorna 1.0 (não há nada a recuperar,
        então não há erro possível).
    """
    if not correct_sources:
        return 1.0

    found_count = 0
    for correct in correct_sources:
        if any(is_source_found(retrieved, correct) for retrieved in retrieved_sources): # noqa
            found_count += 1

    return found_count / len(correct_sources)


def evaluate(student_answer_path: str, dataset_path: str,
             k: int = 10) -> dict[int, float]:
    """Compara os resultados de busca do aluno contra o gabarito.

    Args:
        student_answer_path: Caminho do JSON salvo por search_dataset.
        dataset_path: Caminho do dataset original com gabarito
            (AnsweredQuestions/, não UnansweredQuestions/).
        k: k máximo considerado (avalia recall@1, @3, @5, @10 até esse limite).

    Returns:
        Dict mapeando cada k avaliado para o recall médio naquele k.
    """
    with open(student_answer_path, encoding="utf-8") as f:
        student_results = StudentSearchResults(**json.load(f))

    with open(dataset_path, encoding="utf-8") as f:
        gabarito = RagDataset(**json.load(f))

    gabarito_by_id: dict[str, AnsweredQuestion] = {
        q.question_id: q
        for q in gabarito.rag_questions
        if isinstance(q, AnsweredQuestion)
    }

    k_values = [kv for kv in (1, 3, 5, 10) if kv <= k]
    scores_by_k: dict[int, list[float]] = {kv: [] for kv in k_values}

    for result in student_results.search_results:
        gabarito_pergunta = gabarito_by_id.get(result.question_id)
        if gabarito_pergunta is None:
            continue

        for kv in k_values:
            top_k_sources = result.retrieved_sources[:kv]
            score = recall_at_k(top_k_sources, gabarito_pergunta.sources)
            scores_by_k[kv].append(score)

    averages = {
        kv: (sum(scores) / len(scores) if scores else 0.0)
        for kv, scores in scores_by_k.items()
    }

    print("Evaluation Results")
    print("=" * 40)
    print(f"Questions evaluated: {len(student_results.search_results)}")
    for kv, avg in averages.items():
        print(f"Recall@{kv}: {avg:.3f}")

    return averages

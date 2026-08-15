# mypy: ignore-errors
from pathlib import Path
import json

import bm25s
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .retrieval import search

_MODEL_NAME = "Qwen/Qwen3-0.6B"
_tokenizer: PreTrainedTokenizerBase | None = None
_model: PreTrainedModel | None = None

_SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas sobre o código e a "
    "documentação do projeto vLLM, usando APENAS o contexto fornecido "
    "abaixo. Se o contexto não tiver informação suficiente, diga isso "
    "explicitamente em vez de inventar uma resposta. Sempre que possível, "
    "cite o arquivo de onde a informação vem."
)


def _get_model_and_tokenizer() -> tuple[
    PreTrainedModel, PreTrainedTokenizerBase
]:
    """Carrega o modelo uma única vez e
    reaproveita entre chamadas (lazy load)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)
    assert _model is not None
    assert _tokenizer is not None
    return _model, _tokenizer


def build_prompt(
    question: str, context_chunks: list[dict[str, str | int]]
) -> str:
    """Monta o texto de contexto a partir dos chunks recuperados.

    Args:
        question: A pergunta do usuário.
        context_chunks: Chunks já dentro do orçamento de tokens.

    Returns:
        O texto de contexto formatado, pronto para entrar no prompt.
    """
    blocks: list[str] = []
    for chunk in context_chunks:
        blocks.append(f"[Fonte: {chunk['file_path']}]\n{chunk['text']}")

    context_text = "\n\n".join(blocks)
    return f"Contexto:\n{context_text}\n\nPergunta: {question}"


def fit_chunks_to_budget(
    chunks: list[dict[str, str | int]], max_tokens: int
) -> list[dict[str, str | int]]:
    """Seleciona os chunks mais relevantes que cabem no orçamento de tokens.

    Args:
        chunks: Chunks já ordenados por relevância (saída de search()).
        max_tokens: Limite de tokens disponível para o contexto.

    Returns:
        Sublista de chunks que cabe no orçamento, preservando a ordem.
    """
    _, tokenizer = _get_model_and_tokenizer()

    selected: list[dict[str, str | int]] = []
    tokens_used = 0
    for chunk in chunks:
        chunk_tokens = len(tokenizer(chunk["text"])["input_ids"])
        if tokens_used + chunk_tokens > max_tokens:
            continue
        selected.append(chunk)
        tokens_used += chunk_tokens

    return selected


def answer(
    query: str,
    retriever: bm25s.BM25,
    k: int = 10,
    max_context_tokens: int = 1500,
) -> str:
    """Gera uma resposta em linguagem natural para uma pergunta, usando RAG.

    Args:
        query: A pergunta do usuário.
        retriever: Um BM25 retriever já carregado.
        k: Quantos chunks recuperar antes de filtrar pelo orçamento de tokens.
        max_context_tokens: Orçamento de tokens reservado ao contexto.

    Returns:
        O texto da resposta gerada pelo modelo.
    """
    model, tokenizer = _get_model_and_tokenizer()

    chunks = search(query, retriever, k)
    fitted_chunks = fit_chunks_to_budget(chunks, max_context_tokens)
    user_content = build_prompt(query, fitted_chunks)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs: dict[str, torch.Tensor] = tokenizer(
        prompt_text, return_tensors="pt"
    )
    output_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded: str = tokenizer.decode(  # type: ignore[assignment]
        generated_ids, skip_special_tokens=True
    )
    return decoded.strip()


def answer_dataset(
    student_search_results_path: Path, save_directory: str
) -> None:
    """Gera respostas para resultados de busca já salvos por search_dataset.

    Args:
        student_search_results_path: Caminho do JSON de StudentSearchResults.
        save_directory: Diretório onde salvar o resultado com as respostas.
    """
    with open(student_search_results_path, encoding="utf-8") as f:
        data = StudentSearchResults(**json.load(f))

    model, tokenizer = _get_model_and_tokenizer()
    minimal_answers: list[MinimalAnswer] = []

    for result in tqdm(data.search_results, desc="Gerando respostas"):
        chunks: list[dict[str, str | int]] = []
        for s in result.retrieved_sources:
            chunk_text = ""
            try:
                path_obj = Path(s.file_path)
                if path_obj.exists():
                    file_content = path_obj.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    start = s.first_character_index
                    end = s.last_character_index
                    chunk_text = file_content[start:end]
            except OSError:
                chunk_text = ""

            chunks.append(
                {
                    "file_path": s.file_path,
                    "first_character_index": s.first_character_index,
                    "last_character_index": s.last_character_index,
                    "text": chunk_text,
                }
            )

        fitted_chunks = fit_chunks_to_budget(chunks, max_tokens=64)
        user_content = build_prompt(result.question, fitted_chunks)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs: dict[str, torch.Tensor] = tokenizer(
            prompt_text, return_tensors="pt"
        )
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text: str = tokenizer.decode(  # type: ignore[assignment]
            generated_ids, skip_special_tokens=True
        )

        minimal_answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generated_text.strip(),
            )
        )

    output = StudentSearchResultsAndAnswer(
        search_results=minimal_answers, k=data.k
    )

    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / Path(student_search_results_path).name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"Saved student_search_results_and_answer to {output_path}")

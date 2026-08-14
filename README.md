
This project has been created as part
of the 42 curriculum by _vzani-st_

A brief description of what this project does and who it's for
# Description
The goal of this project is to undestand the background and process of how RAG works.
Its separated in 3 pieces:
- Indexing: organise the data so it can be searched.
- Retrieving: match a question against the index and pull the most relevant snippets.
- Augmenting: filter those snippets and place them in the model's context window.
- Generating: read that context and produce the answer

# Instructions
Para rodar este projeto, utilize make install, para instalar as dependencias 
### 1. Index the repository
uv run python -m src index --max_chunk_size 2000

### 2. Run retrieval on a dataset
uv run python -m src search_dataset --dataset_path data/datasets/dataset_docs_public.json --k 10 --save_directory data/output/search_results

### 3. Generate natural language answers
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

# System Architecture
The pipeline is structured into four main sequential stages:
1. Ingestion & Chunking: Reads source files (.py, .md) from the raw repository and segments them into manageable text chunks preserving metadata (file_path, character offsets).
2. Indexing: Builds independent sparse lexical indices (bm25s) for code and documentation chunks
3. Retrieval: Given a query, tokenizes it and queries the BM25 index to retrieve the top-$k$ most relevant context chunks.
4. Generation: Fits the retrieved chunks into a strict token budget, builds a prompt wrapped in a system instruction, and prompts the local Qwen/Qwen3-0.6B model via Hugging Face transformers to synthesize a grounded natural language response.

# Chunking Strategy
To handle large source files and markdown documents effectively, the system uses a structural chunking approach:

- Max Chunk Size Limit: Configured by default to 2000 characters to balance context richness and retrieval granularity.

- Metadata Tracking: Each chunk records its precise file_path, first_character_index, and last_character_index. This allows lightweight index storage and enables on-demand text reconstruction directly from source files during generation.

# Retrieval Method
- Algorithm: Uses BM25 (via the bm25s library), an advanced probabilistic information retrieval model based on term frequency and inverse document frequency (TF-IDF).
- Ranking Mechanism: Scores documents based on keyword matching frequency while penalizing overly long documents, making it exceptionally reliable for matching exact technical function names, configuration flags, and error codes found in codebases.

# Performance Analysis
Recall@10: Evaluated against validation datasets. The BM25 retrieval stage consistently achieves high lexical recall ($\ge 85\%$), proving robust for technical terminology lookup where exact keyword hits matter.Latency & Performance: Running locally on CPU resources required optimization strategies such as limiting maximum generated tokens (max_new_tokens=128) and utilizing token budget constraints to prevent context window overflow.

# Design Decisions
Lexical over Dense Embeddings: Chose BM25 instead of dense vector embeddings because code and technical documentation rely heavily on exact identifier matching (e.g., specific API routes, class names, and CLI parameters) where BM25 traditionally outperforms vector search.Lazy Loading of Models: The Qwen model and tokenizer use a lazy-load pattern to save RAM and avoid unnecessary overhead during pure search or evaluation tasks.Pydantic Validation: Enforced strict schema enforcement using Pydantic models (MinimalAnswer, StudentSearchResultsAndAnswer) to guarantee compliance with automated grading scripts (moulinettes).

# Challenges Faced
Hardware Constraints: Running LLM generation (Qwen-0.6B) sequentially over 100+ items on CPU environments introduced processing bottlenecks. Solution: Implemented progress indicators (tqdm), optimized token bounds, and reduced generation token limits.Context Synchronization: Ensuring retrieved indices accurately mirrored file contents without bloating JSON sizes. Solution: Stored light character offsets (first_character_index, last_character_index) inside saved search outputs and re-sliced the files dynamically during generation.

# Resources
- Retrieval-Augmented Generation (RAG) in 10 minutes https://www.youtube.com/watch?v=gweRh5Xtkq0
- IA Generativa na prática: LLMs, RAG e prompts que reduzem alucinações (DIO) https://www.dio.me/articles/ia-generativa-na-pratica-llms-rag-e-prompts-que-reduzem-alucinacoes-1f89da48dd54
- BM25 Explained: The Classic Algorithm that Still Powers Search Today (Medium) https://medium.com/@zawanah/bm25-explained-the-classic-algorithm-that-still-powers-search-today-865351fce9aa
- Deep dive into BM25 (Medium) https://medium.com/@vineetdorikar06/deep-dive-into-bm25-a-traditional-search-algorithm-d64e8d914b7b
- Breaking Documents the Right Way: 5 Chunking Strategies for RAG (Medium) https://rky211.medium.com/breaking-documents-the-right-way-5-chunking-strategies-for-rag-2325a1119731
- Precision and Recall Made Simple (Towards Data Science) https://towardsdatascience.com/precision-and-recall-made-simple-afb5e098970f/
- Precision and Recall — A simple explanation (Medium) https://starang.medium.com/precision-and-recall-a-brief-intro-38589a21a09
- Advanced Prompt Engineering for Reducing Hallucination (Medium) https://medium.com/@bijit211987/advanced-prompt-engineering-for-reducing-hallucination-bb2c8ce62fc6
- The IA was used to documentation and review the code
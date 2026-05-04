# LangChain RAG 2 Tutorial

This project is a notebook-first tutorial for building a multi-document Retrieval-Augmented Generation (RAG) workflow with LangChain, OpenAI embeddings, ChromaDB, and PDF/text document loading.

It builds on a simple single-document RAG pattern and adds the pieces you need when a corpus contains many related files:

1. **Multi-document ingestion:** scan a data folder, load `.txt` transcripts and `.pdf` show notes, attach metadata, split the content, embed each chunk, and persist the vectors to Chroma.
2. **Metadata-aware querying:** load the persisted vector store, retrieve relevant chunks, format source metadata into the prompt, answer with citations, and debug retrieved context.

The sample corpus is a set of Security Now! episode transcripts and show notes under [`data/`](data/), spanning episodes `sn-1059` to `sn-1073`.

## Project Layout

```text
.
|-- data/
|   |-- sn-1059.txt                 # Episode transcript
|   |-- sn-1059-notes.pdf           # Episode show notes
|   |-- ...
|   |-- sn-1073.txt
|   `-- sn-1073-notes.pdf
|-- docs/
|   `-- RAG_Learning_Readmap.md     # Broader RAG learning roadmap
|-- notebooks/
|   |-- Rag2_Multi_Docs.ipynb       # Phase 1: load, chunk, embed, and store many docs
|   `-- Rag2_Query.ipynb            # Phase 2: retrieve, cite, answer, and debug
|-- src/
|   |-- rag1_add_docs.py            # Script version of ingestion flow
|   `-- rag1_query.py               # Script version of query flow
|-- pyproject.toml                  # Python dependencies
|-- justfile                        # Convenience commands
`-- README.md
```

## What You Will Build

The complete RAG flow looks like this:

```text
Text files + PDF files
        -> Documents with metadata
        -> Chunks
        -> Embeddings
        -> Chroma vector store

Question
        -> Retriever
        -> Metadata-aware context formatting
        -> Prompt + LLM
        -> Cited answer
```

The first notebook creates the vector database. The second notebook depends on that database already existing.

## Roadmap Mapping

This tutorial is the second hands-on checkpoint in the broader [`RAG Learning Roadmap`](docs/RAG_Learning_Readmap.md). It moves from a basic single-file RAG pipeline into multi-document retrieval, source metadata, citations, and early retrieval debugging.

| Roadmap phase | How this tutorial connects | Where to look |
| --- | --- | --- |
| Phase 1: Foundations | Reinforces documents, chunks, embeddings, vector stores, and similarity retrieval across a larger corpus. | [`Rag2_Multi_Docs.ipynb`](notebooks/Rag2_Multi_Docs.ipynb) |
| Phase 2: Basic RAG Pipeline | Implements the full end-to-end flow: load many documents, split, embed, store, retrieve, prompt, call LLM, and parse the answer. | [`Rag2_Multi_Docs.ipynb`](notebooks/Rag2_Multi_Docs.ipynb), [`Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb) |
| Phase 3: Retrieval Optimization | Adds practical tuning knobs: `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `RETRIEVAL_K`. The query notebook also inspects retrieved chunks. | Experiment config cells in both notebooks |
| Phase 4: Advanced Retrieval Techniques | Partially introduced with episode-aware retrieval. If a question names an episode like `sn-1069`, the notebook searches more candidates and filters by episode metadata. | `retrieve_docs()` in [`Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb) |
| Phase 5: Context Optimization | Partially introduced through metadata-rich context formatting. The chain sends source headers along with retrieved text so the model can cite evidence. | `format_docs()` in [`Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb) |
| Phase 6: Adaptive & Iterative RAG | Not implemented yet. The current system does one retrieval pass, with a simple conditional branch for explicit episode IDs. | Suggested follow-up |
| Phase 7: Agentic RAG | Not implemented yet. The model does not choose tools or decide whether to retrieve again. | Suggested follow-up |
| Phase 8: Workflow Orchestration | Not implemented yet. The flow is an LCEL chain, not a graph with state and branches. | Move this flow into LangGraph later |
| Phase 9: Evaluation & Debugging | Introduced through debug cells that print retrieved chunks, episode IDs, doc types, file names, and page numbers. | Debug section in [`Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb) |
| Phase 10: Production Considerations | Reinforced by separating ingestion from querying and preserving metadata for traceability. | See "Why Metadata Matters" below |

In roadmap terms, this repository currently teaches:

```text
Phase 1 foundations -> Phase 2 basic RAG -> Phase 3 retrieval tuning
    -> early Phase 4 metadata-aware retrieval -> Phase 9 debugging habits
```

Use this tutorial as the "multi-document RAG baseline" before adding reranking, hybrid search, query rewriting, evaluation datasets, or workflow orchestration.

## Why Metadata Matters

Single-document RAG can often get by with raw text chunks. Multi-document RAG needs stronger source tracking.

This project adds metadata during ingestion:

- `source`: original file path
- `file_name`: original file name
- `doc_type`: `transcript` for `.txt` files or `show_notes` for `.pdf` files
- `episode_id`: episode identifier such as `sn-1073`
- `episode_number`: numeric episode number such as `1073`
- `page`: PDF page number for show notes

That metadata improves the system in three ways:

- **Traceability:** answers can cite where information came from.
- **Debuggability:** retrieved chunks can be inspected by episode, document type, file, and page.
- **Retrieval control:** explicit episode questions can be handled by filtering candidate chunks to the requested episode.

In this tutorial, [`Rag2_Multi_Docs.ipynb`](notebooks/Rag2_Multi_Docs.ipynb) represents the ingestion job, and [`Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb) represents the query path.

## Prerequisites

- Python 3.12 or newer
- An OpenAI API key
- `uv` is recommended for environment setup

Install dependencies:

```bash
uv sync
```

If you prefer to use the `justfile`:

```bash
just setup
```

Create a local `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The project uses `python-dotenv`, so both notebooks and scripts call `load_dotenv()` to read this key.

## Tutorial Part 1: Add Multiple Documents

Open [`notebooks/Rag2_Multi_Docs.ipynb`](notebooks/Rag2_Multi_Docs.ipynb).

This notebook implements:

```text
Documents -> Chunk -> Embed -> Store
```

### 1. Choose the input folder and vector store location

For local development, the notebook uses:

```python
data_folder = "../data"
store_location = "../vector_db/chroma_db2"
```

`data_folder` points to the source corpus. `store_location` is where Chroma persists the generated vector database.

### 2. Load the environment

```python
from dotenv import load_dotenv
import os

load_dotenv()
```

This loads `OPENAI_API_KEY` from `.env`.

### 3. Set chunking experiment values

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
```

These values control how source documents are split:

- `CHUNK_SIZE=500`: each chunk aims to be about 500 characters.
- `CHUNK_OVERLAP=250`: adjacent chunks share 250 characters to preserve context across boundaries.

The larger overlap is useful for transcripts because important details can span several short speaker turns.

### 4. Build metadata for every file

```python
def build_base_metadata(file_path: Path, doc_type: str) -> dict:
    stem = file_path.stem.lower()
    match = re.search(r"(sn-\d+)", stem)
    episode_id = match.group(1) if match else None
    episode_number = int(episode_id.split("-")[1]) if episode_id else None

    metadata = {
        "source": str(file_path),
        "file_name": file_path.name,
        "doc_type": doc_type,
    }
    if episode_id is not None:
        metadata["episode_id"] = episode_id
    if episode_number is not None:
        metadata["episode_number"] = episode_number
    return metadata
```

This function extracts episode information from file names like `sn-1073.txt` or `sn-1073-notes.pdf`.

### 5. Load text and PDF documents

```python
def load_documents(data_dir: str) -> list[Document]:
    docs = []
    for file_path in sorted(Path(data_dir).rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            metadata = build_base_metadata(file_path, doc_type="transcript")
            docs.append(Document(page_content=text, metadata=metadata))
        elif suffix == ".pdf":
            base_metadata = build_base_metadata(file_path, doc_type="show_notes")
            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    metadata = {**base_metadata, "page": page_number}
                    docs.append(Document(page_content=text, metadata=metadata))

    return docs
```

This loader standardizes different file types into LangChain `Document` objects:

- `.txt` files become transcript documents.
- `.pdf` files are read page by page with `pypdf`.
- Empty PDF pages are skipped.
- Unsupported file types are ignored.

### 6. Split the documents into chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

splits = splitter.split_documents(docs)
```

RAG systems usually retrieve chunks, not whole files. Because `split_documents()` preserves document metadata, every chunk still knows its original source, episode, document type, and page.

### 7. Embed and persist the chunks

```python
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=store_location,
    collection_metadata={"hnsw:space": "cosine"},
)
```

This sends each chunk to OpenAI's embedding model, stores the resulting vectors in Chroma, and persists the database under `vector_db/chroma_db2`.

After this notebook runs successfully, the query notebook can load the vector store without re-embedding the corpus.

## Tutorial Part 2: Query Multiple Documents

Open [`notebooks/Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb).

This notebook implements:

```text
Query -> Retrieve -> Format context -> LLM -> Cited answer
```

Run the multi-document ingestion notebook first, because this notebook expects the vector database to already exist.

### 1. Point to the existing vector store

```python
store_location = "../vector_db/chroma_db2"

if not Path(store_location).exists():
    raise FileNotFoundError(f"Vector store not found: {store_location}")
```

This must match the `store_location` used in the ingestion notebook.

### 2. Load the environment

```python
from dotenv import load_dotenv
import os

load_dotenv()
```

The query notebook needs `OPENAI_API_KEY` for both embeddings and chat completion.

### 3. Set retrieval experiment values

```python
RETRIEVAL_K = 6
```

`RETRIEVAL_K` controls how many chunks are sent to the model. Higher values can improve recall, but they also add more context noise and token cost.

### 4. Load the vector store and create a retriever

```python
embedding = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory=store_location,
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
```

The retriever searches Chroma for the chunks most relevant to the question.

### 5. Create a citation-aware prompt

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a Security Now RAG assistant.

You are given context retrieved from a Security Now document database.
The corpus contains episode-based sources (for example sn-1072), document types
(transcript/show_notes), and optional page numbers.

Instructions:
- Use ONLY the provided context to answer the question.
- Do NOT use outside knowledge.
- If the answer is not in the context, say: "Not found in provided context."
- Do not make up information.
- For each key claim, cite episode/doc metadata when available.
- Prefer show_notes for concise factual recall and transcripts for detailed explanation.

Citation format:
- Include citations inline using this format: [episode_id | doc_type | page if available]
- Example: [sn-1072 | show_notes | p25]

Context:
{context}

Question:
{question}

Answer:
""")
```

This prompt makes the model behave like a grounded assistant over the retrieved Security Now corpus. It also tells the model how to cite retrieved metadata.

### 6. Resolve and format metadata

```python
def resolve_metadata(raw_metadata: dict) -> dict:
    source = raw_metadata.get("source", "unknown")
    page = raw_metadata.get("page")
    file_name = raw_metadata.get("file_name")
    episode_id = raw_metadata.get("episode_id")
    episode_number = raw_metadata.get("episode_number")
    doc_type = raw_metadata.get("doc_type")

    source_path = Path(source) if source and source != "unknown" else None

    if not file_name and source_path:
        file_name = source_path.name

    if not episode_id and source_path:
        match = re.search(r"(sn-\d+)", source_path.stem.lower())
        if match:
            episode_id = match.group(1)

    if not doc_type and source_path:
        suffix = source_path.suffix.lower()
        if suffix == ".txt":
            doc_type = "transcript"
        elif suffix == ".pdf":
            doc_type = "show_notes"

    return {
        "source": source,
        "page": page,
        "file_name": file_name or "unknown",
        "episode_id": episode_id or "unknown",
        "episode_number": episode_number if episode_number is not None else "unknown",
        "doc_type": doc_type or "unknown",
    }
```

`resolve_metadata()` makes the query notebook robust even if older chunks have less metadata than the latest ingestion notebook writes.

```python
def format_docs(docs):
    formatted = []
    for doc in docs:
        md = resolve_metadata(doc.metadata)
        location = f"{md['source']} (page {md['page']})" if md["page"] else md["source"]
        metadata_header = (
            f"Episode: {md['episode_id']} | Episode Number: {md['episode_number']} | "
            f"Doc Type: {md['doc_type']} | File: {md['file_name']} | Location: {location}"
        )
        formatted.append(f"{metadata_header}\n{doc.page_content}")
    return "\n\n".join(formatted)
```

`format_docs()` sends both the retrieved text and its metadata into the prompt, giving the model enough information to produce citations.

### 7. Add episode-aware retrieval

```python
def extract_episode_from_question(question: str) -> str | None:
    match = re.search(r"\b(sn-\d+)\b", question.lower())
    return match.group(1) if match else None

def retrieve_docs(question: str):
    episode_id = extract_episode_from_question(question)

    if episode_id:
        candidate_k = max(RETRIEVAL_K * 5, 20)
        candidates = vectorstore.similarity_search(question, k=candidate_k)

        episode_docs = []
        for d in candidates:
            md = resolve_metadata(d.metadata)
            source = str(md.get("source", "")).lower()
            meta_episode = str(md.get("episode_id", "")).lower()
            if meta_episode == episode_id or episode_id in source:
                episode_docs.append(d)

        if episode_docs:
            return episode_docs[:RETRIEVAL_K]

    return retriever.invoke(question)
```

This function handles two retrieval modes:

- If the question names an episode like `sn-1069`, search a wider candidate set and keep chunks from that episode.
- Otherwise, use normal semantic retrieval.

This is a small but useful step toward metadata filtering and query routing.

### 8. Compose the RAG chain with LCEL

```python
rag_chain = (
    {
        "context": RunnableLambda(retrieve_docs) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

This chain:

1. Receives a question.
2. Retrieves relevant chunks, with episode-aware behavior when possible.
3. Formats retrieved chunks with metadata headers.
4. Inserts context and question into the prompt.
5. Sends the prompt to the chat model.
6. Parses the model response as a string.

### 9. Ask several questions

```python
questions = [
    "What are the main topics across these documents?",
    "any microsoft security news?",
    "any CVE 10 security incidents?",
    "Can summarize episode sn-1069?",
    "Can summarize episode sn-1070?",
    "Summarize the most recent episode with CVE 10 security incidents."
]

for idx, question in enumerate(questions, start=1):
    response = rag_chain.invoke(question)
    print("=" * 60)
    print(f"Question {idx}: {question}")
    print("-" * 60)
    print(response)
    print()
```

You can replace these with your own questions about the corpus. Questions that include an explicit episode ID should benefit from the episode-aware retrieval path.

### 10. Debug retrieved chunks

The notebook includes a debug section:

```python
for idx, question in enumerate(questions, start=1):
    print("=" * 60)
    print(f"Debug for Question {idx}: {question}")
    print("=" * 60)

    docs = retrieve_docs(question)
    for i, d in enumerate(docs, start=1):
        md = resolve_metadata(d.metadata)
        location = f"{md['source']} (page {md['page']})" if md["page"] else md["source"]
        print(
            f"chunk {i} | episode={md['episode_id']} ({md['episode_number']}) | "
            f"type={md['doc_type']} | file={md['file_name']} | location={location}"
        )
        print(d.page_content)
        print("-" * 50)

    print()
```

Use this when an answer looks weak, incomplete, or poorly cited. It shows exactly which chunks were sent to the model as context.

## Running the Script Versions

The `src/` directory contains script versions of the notebook workflows.

Build the vector store:

```bash
uv run python src/rag1_add_docs.py
```

Query the vector store:

```bash
uv run python src/rag1_query.py
```

Important differences from the notebooks:

- The scripts currently use `vector_db/chroma_db1a`.
- The notebooks use `vector_db/chroma_db2`.
- The ingestion script loads `.txt` and `.pdf` files, but its metadata is simpler than the notebook metadata.

If you switch between notebooks and scripts, make sure both phases point to the same store location.

## Colab Notes

Both notebooks include commented cells for Google Colab:

- Mount Google Drive.
- Set document and Chroma paths under Drive.
- Load `OPENAI_API_KEY` from Colab user data.

For local development, keep the Colab cells commented and use the local path cells instead.

## Common Issues

### Missing API key

If you see an authentication error, confirm that `.env` exists in the project root and contains:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Vector store not found

Run [`notebooks/Rag2_Multi_Docs.ipynb`](notebooks/Rag2_Multi_Docs.ipynb) before [`notebooks/Rag2_Query.ipynb`](notebooks/Rag2_Query.ipynb).

Also confirm that both notebooks use the same `store_location`.

### Empty or irrelevant answers

Inspect the retrieved chunks with the debug section. If the retrieved chunks do not contain the information needed to answer the question, try:

- asking a more specific question
- increasing `RETRIEVAL_K`
- decreasing `RETRIEVAL_K` if the model is getting too much noisy context
- adjusting `CHUNK_SIZE` and `CHUNK_OVERLAP`
- asking with an explicit episode ID, such as `sn-1073`
- adding stronger metadata filtering

### Poor citations

Confirm that the ingestion notebook was run after the metadata fields were added. Older vector stores may contain chunks with only partial metadata.

If citations are still weak, inspect the formatted context and make sure each chunk includes `episode_id`, `doc_type`, and `page` where available.

### Slow ingestion or high embedding cost

This corpus contains both transcripts and PDFs, so ingestion can create many chunks. To reduce work while experimenting:

- ingest fewer files
- increase `CHUNK_SIZE`
- reduce `CHUNK_OVERLAP`
- delete and rebuild the vector store only when needed

## Next Steps

Once the multi-document flow works, useful experiments include:

- Rename the script files from `rag1_*` to `rag2_*` and align them with the notebook metadata.
- Add true Chroma metadata filters instead of post-filtering retrieved candidates.
- Add MMR retrieval and compare it with plain similarity search.
- Add source citations to the final answer in a stricter structured format.
- Create an evaluation set of questions with expected source episodes.
- Add reranking or contextual compression before the prompt.
- Wrap the query chain in a small CLI or web app.
- Move the flow into LangGraph for explicit retrieval, answer, and retry nodes.

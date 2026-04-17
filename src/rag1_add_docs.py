import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DATA_LOCATION = Path("data")
STORE_LOCATION = Path("vector_db/chroma_db1a")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"


def load_documents(data_location: Path) -> list[Document]:
    logging.info("Loading supported documents from %s", data_location)
    documents: list[Document] = []

    for file_path in sorted(data_location.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(page_content=text, metadata={"source": str(file_path)})
            )
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(file_path), "page": page_number},
                        )
                    )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    logging.info("Splitting %s document(s) into chunks", len(documents))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(splits: list[Document], store_location: Path) -> Chroma:
    logging.info("Ensuring vector store directory exists at %s", store_location)
    store_location.mkdir(parents=True, exist_ok=True)

    logging.info("Building embeddings with model %s", EMBEDDING_MODEL)
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    logging.info("Persisting %s chunk(s) to Chroma", len(splits))
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(store_location),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    load_dotenv()
    logging.info("Environment variables loaded")

    if not DATA_LOCATION.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_LOCATION}")

    documents = load_documents(DATA_LOCATION)
    if not documents:
        raise ValueError(f"No supported documents found in {DATA_LOCATION}")

    splits = split_documents(documents)
    build_vector_store(splits, STORE_LOCATION)

    logging.info("Loaded %s document/page item(s) from %s", len(documents), DATA_LOCATION)
    logging.info("Created %s chunk(s)", len(splits))
    logging.info("Persisted Chroma vector store to %s", STORE_LOCATION)


if __name__ == "__main__":
    main()

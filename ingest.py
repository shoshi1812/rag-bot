"""PREPARATION stage. Run once, and again whenever the documents change:

    python ingest.py
"""
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from models import get_embed_model


def main():
    # Step 1: load the documents (md, txt, pdf, docx)
    documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
    print(f"Loaded {len(documents)} documents")

    # Step 2: split into chunks
    #   a. by section headings (each section is a complete idea)
    #   b. a section that is too long is cut again, with overlap
    chunks = MarkdownNodeParser().get_nodes_from_documents(documents)
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.get_nodes_from_documents(chunks)
    print(f"Created {len(chunks)} chunks")

    # Step 3 + 4: turn each chunk into a vector and save it in Chroma.
    # The collection is rebuilt from scratch, so there are no duplicates.
    client = chromadb.PersistentClient(path=config.DB_DIR)
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(config.COLLECTION)

    storage = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=collection))
    VectorStoreIndex(chunks, storage_context=storage, embed_model=get_embed_model(), show_progress=True)
    print("Done. The vector database is saved in the 'storage' folder.")


if __name__ == "__main__":
    main()

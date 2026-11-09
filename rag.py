"""REAL-TIME stage: question -> (expansion) -> hybrid search -> prompt -> answer."""
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from keyword_search import KeywordRetriever
from models import get_embed_model, get_llm

# The prompt limits the model to the retrieved chunks (protection against hallucination)
ANSWER_PROMPT = """אתה עוזר שעונה על שאלות עובדים לפי נהלי הארגון.
ענה אך ורק על סמך הקטעים שלמטה. אם התשובה לא מופיעה בהם, כתוב: "לא נמצא מידע על כך בנהלים".
ציין בסוף את הסעיף שממנו נלקחה התשובה. ענה בעברית, בקצרה.

קטעים מהנהלים:
{context}

שאלה: {question}
תשובה:"""

# Used only when the question is short: the LLM writes alternative phrasings
EXPAND_PROMPT = """כתוב {num_queries} ניסוחים חלופיים בעברית לשאלת החיפוש הבאה, כל אחד בשורה נפרדת,
ללא מספור וללא הסברים. השתמש במונחים רשמיים.
שאלה: {query}
ניסוחים:
"""


class RagBot:
    def __init__(self):
        self.llm = get_llm()

        # Open the Chroma collection that ingest.py created
        client = chromadb.PersistentClient(path=config.DB_DIR)
        collection = client.get_collection(config.COLLECTION)  # error if ingest.py was not run

        # Semantic search: by meaning (same embedding model as in ingest.py!)
        index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=collection), embed_model=get_embed_model()
        )
        self.semantic = index.as_retriever(similarity_top_k=config.TOP_K * 2)

        # Keyword search: on the same chunks, read back from Chroma
        stored = collection.get(include=["documents", "metadatas"])
        chunks = [
            metadata_dict_to_node(meta, text=text)
            for text, meta in zip(stored["documents"], stored["metadatas"])
        ]
        self.keyword = KeywordRetriever(chunks, top_k=config.TOP_K * 2)

    def search(self, question):
        """Hybrid search. Returns the TOP_K best chunks and whether the question was expanded."""
        expand = len(question.split()) < config.SHORT_QUESTION_WORDS
        retriever = QueryFusionRetriever(
            [self.semantic, self.keyword],
            llm=self.llm,
            similarity_top_k=config.TOP_K,
            num_queries=2 if expand else 1,  # 1 = original question only, no LLM call
            query_gen_prompt=EXPAND_PROMPT,
            mode="reciprocal_rerank",  # merges the two result lists by rank
            use_async=False,
        )
        return retriever.retrieve(question), expand

    def ask(self, question):
        # 1. Search
        chunks, expanded = self.search(question)

        # 2. Build the prompt: instructions + chunks + question
        context = "\n\n---\n\n".join(c.node.get_content() for c in chunks)
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        # 3. The LLM writes the answer
        answer = self.llm.complete(prompt).text
        return answer, chunks, expanded

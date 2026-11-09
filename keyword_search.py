"""Keyword search (BM25): finds exact words such as clause numbers ("5.2"), names and codes.

Semantic search is weak at exact terms, so we combine both (hybrid search).
BM25 gives a high score to a chunk that contains the question's words, especially rare words.
"""
import math
import re
from collections import Counter

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

# A word = letters/digits. "5.2" stays one word, so clause numbers can be found.
WORD = re.compile(r"\w+(?:\.\w+)*")


def tokenize(text):
    return WORD.findall(text.lower())


class KeywordRetriever(BaseRetriever):
    def __init__(self, chunks, top_k, k1=1.5, b=0.75):
        super().__init__()
        self.chunks = chunks
        self.top_k = top_k
        self.k1 = k1  # how fast repeated words stop adding score
        self.b = b    # how much long chunks are penalized

        self.word_counts = [Counter(tokenize(c.get_content())) for c in chunks]
        self.lengths = [sum(counts.values()) for counts in self.word_counts]
        self.avg_length = sum(self.lengths) / len(chunks) if chunks else 0

        # Rare words are worth more (idf = inverse document frequency)
        n = len(chunks)
        doc_freq = Counter(word for counts in self.word_counts for word in counts)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in doc_freq.items()}

    def score(self, query_words, i):
        counts, length = self.word_counts[i], self.lengths[i]
        total = 0.0
        for word in query_words:
            tf = counts.get(word, 0)  # how many times the word appears in this chunk
            if tf:
                size_factor = 1 - self.b + self.b * length / self.avg_length
                total += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * size_factor)
        return total

    def _retrieve(self, query_bundle):
        query_words = set(tokenize(query_bundle.query_str))
        scores = [(self.score(query_words, i), i) for i in range(len(self.chunks))]
        best = sorted((s for s in scores if s[0] > 0), reverse=True)[: self.top_k]
        return [NodeWithScore(node=self.chunks[i], score=score) for score, i in best]

"""Evidence retrieval (E1). Each retriever ranks one resume's bullets against one job
requirement and returns [(bullet_idx, score), ...] best first.

- TfidfRetriever       TF-IDF vectors + cosine similarity (Topic 1 lab)
- BM25Retriever        TF-IDF with document-length normalisation (Topic 8), no dependencies
- EmbeddingRetriever   sentence-transformers bi-encoder + cosine (Topics 2 and 8)
- HybridRetriever      BM25 + embedding candidates, re-ordered by a cross-encoder re-ranker
                       (Topic 8 two-phase retrieval), or reciprocal-rank fusion if it can't load
"""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING

from .text import tokenize

if TYPE_CHECKING:
    import numpy as np


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in corpus]
        self.len = [len(d) for d in corpus]
        self.avgdl = sum(self.len) / max(len(corpus), 1) or 1.0
        df = Counter(t for d in self.tf for t in d)
        n = len(corpus)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.tf)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                    scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


def _top(scores, k, positive_only=False):
    order = sorted((i for i, score in enumerate(scores)
                    if not positive_only or score > 0), key=lambda i: -scores[i])
    return [(i, float(scores[i])) for i in order[:k]]


class TfidfRetriever:
    """Fitted per resume (the bullets are the corpus), as in the Topic 1 lab."""
    name = "tfidf"

    def rank(self, query: str, bullets: list[str], k: int):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        try:
            X = vec.fit_transform(bullets)
        except ValueError:          # empty vocabulary
            return []
        scores = cosine_similarity(vec.transform([query]), X)[0]
        return _top(list(scores), k, positive_only=True)


class BM25Retriever:
    name = "bm25"

    def rank(self, query: str, bullets: list[str], k: int):
        bm = BM25Okapi([tokenize(b) for b in bullets])
        return _top(bm.get_scores(tokenize(query)), k, positive_only=True)


_MODEL_CACHE: dict = {}


def _st_model(name):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _cross_encoder(name):
    key = ("ce", name)
    if key not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        _MODEL_CACHE[key] = CrossEncoder(name)
    return _MODEL_CACHE[key]


class EmbeddingRetriever:
    name = "embedding"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = _st_model(model_name)
        self.e5 = "e5" in model_name   # e5 models expect query:/passage: prefixes
        self._corpus_embeddings: OrderedDict[tuple[str, ...], "np.ndarray"] = OrderedDict()

    def rank(self, query: str, bullets: list[str], k: int):
        q = f"query: {query}" if self.e5 else query
        docs = [f"passage: {b}" for b in bullets] if self.e5 else bullets
        qv = self.model.encode([q], normalize_embeddings=True)[0]
        key = tuple(docs)
        if key in self._corpus_embeddings:
            dv = self._corpus_embeddings.pop(key)
            self._corpus_embeddings[key] = dv
        else:
            dv = self.model.encode(docs, normalize_embeddings=True)
            self._corpus_embeddings[key] = dv
            if len(self._corpus_embeddings) > 32:
                self._corpus_embeddings.popitem(last=False)
        return _top(list(dv @ qv), k)


class HybridRetriever:
    name = "hybrid"

    def __init__(self, embedding_model: str, reranker_model: str | None, candidates: int = 10):
        self.bm25 = BM25Retriever()
        self.emb = EmbeddingRetriever(embedding_model)
        self.candidates = candidates
        try:
            self.reranker = _cross_encoder(reranker_model) if reranker_model else None
        except Exception as e:  # missing package / no network: fall back to fusion
            print(f"[hybrid] reranker unavailable ({e!r}); using reciprocal-rank fusion")
            self.reranker = None

    def rank(self, query: str, bullets: list[str], k: int):
        a = self.bm25.rank(query, bullets, self.candidates)
        b = self.emb.rank(query, bullets, self.candidates)
        if self.reranker is not None:
            cand = sorted({i for i, _ in a} | {i for i, _ in b})
            scores = self.reranker.predict([(query, bullets[i]) for i in cand])
            return sorted(((i, float(s)) for i, s in zip(cand, scores)), key=lambda x: -x[1])[:k]
        fused = Counter()
        for ranking in (a, b):
            for r, (i, _) in enumerate(ranking):
                fused[i] += 1 / (60 + r)
        return [(i, s) for i, s in fused.most_common(k)]


def build_retriever(cfg: dict, method: str | None = None):
    r = cfg["retrieval"]
    method = method or r["method"]
    if method == "tfidf":
        return TfidfRetriever()
    if method == "bm25":
        return BM25Retriever()
    if method == "embedding":
        return EmbeddingRetriever(r["embedding_model"])
    if method == "hybrid":
        return HybridRetriever(r["embedding_model"], r.get("reranker_model"))
    raise ValueError(f"unknown retrieval method {method!r}")

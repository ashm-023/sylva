"""
Sylva-wild: the potion model's BERT WordPiece tokenizer in plain Python.

Same ids as the `tokenizers` library for this tokenizer.json (BertNormalizer with
lowercase and accent stripping, BertPreTokenizer, WordPiece, no special tokens
added), checked on catalogue texts by scripts/nature/wild/export_query_model.py.
Written out so the Vercel function does not ship `tokenizers` and the Hugging Face
packages it pulls in (~30 MB, against a 250 MB function limit).
"""

import json
import unicodedata
from pathlib import Path


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch) in ("Cc", "Cf", "Cn", "Co", "Cs")


def _is_whitespace(ch: str) -> bool:
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        model, norm = spec["model"], spec["normalizer"]
        if (model["type"], norm["type"], spec["pre_tokenizer"]["type"]) != (
                "WordPiece", "BertNormalizer", "BertPreTokenizer"):
            raise ValueError(f"{path}: not a BERT WordPiece tokenizer")
        if spec.get("post_processor") and spec["post_processor"].get("special_tokens"):
            raise ValueError(f"{path}: adds special tokens, which this reader does not")
        self.vocab: dict[str, int] = model["vocab"]
        self.unk_id = self.vocab[model["unk_token"]]
        self.prefix = model["continuing_subword_prefix"]
        self.max_chars = model["max_input_chars_per_word"]
        self.lowercase = norm["lowercase"]
        strip = norm.get("strip_accents")
        self.strip_accents = self.lowercase if strip is None else strip
        self.clean = norm["clean_text"]
        self.cjk = norm["handle_chinese_chars"]

    def normalise(self, text: str) -> str:
        out = []
        for ch in text:
            if self.clean:
                if ch == "\0" or ch == "�" or _is_control(ch):
                    continue
                if _is_whitespace(ch):
                    ch = " "
            if self.cjk and _is_cjk(ord(ch)):
                ch = f" {ch} "
            out.append(ch)
        text = "".join(out)
        if self.strip_accents:
            text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        if self.lowercase:
            text = text.lower()
        return text

    @staticmethod
    def pre_tokenise(text: str) -> list[str]:
        words, cur = [], []
        for ch in text:
            if ch.isspace():
                if cur:
                    words.append("".join(cur))
                    cur = []
            elif _is_punctuation(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(ch)
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))
        return words

    def word_ids(self, word: str) -> list[int]:
        if len(word) > self.max_chars:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    break
                end -= 1
            else:
                return [self.unk_id]
            start = end
        return ids

    def encode(self, text: str, max_tokens: int | None = None) -> list[int]:
        ids = [i for w in self.pre_tokenise(self.normalise(text)) for i in self.word_ids(w)]
        return ids[:max_tokens] if max_tokens else ids

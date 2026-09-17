"""CLIP BPE tokenizer in pure Python.

`transformers`/`tokenizers` have no riscv64 wheels, so we reimplement the exact CLIP
byte-pair encoding behind SD's text encoder. Only `json` and `re` are needed.

Two details that are easy to get wrong and silently ruin every image:

* sd-turbo sets ``pad_token`` to ``"!"``, whose id is **0** -- not the usual
  ``<|endoftext|>``/49407. Padding with the wrong id makes the text encoder attend to 77
  copies of EOS.
* ``"!"`` is registered as an *added token*, so HF maps it straight to id 0 and never
  runs BPE on it. Fed through BPE it would come out as ``"!</w>"`` = 256 instead.
"""
import json
import os
import re

# Mirrors the CLIP pattern. Python's `re` has no \p{L}, so:
#   [^\W\d_]+ == letters (unicode aware),  \d == digits,  [^\s\w]+ == punctuation
_PAT = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+",
    re.IGNORECASE | re.UNICODE,
)

BOS = 49406
EOS = 49407


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    return set(zip(word[:-1], word[1:]))


def _is_cjk(cp):
    """BERT BasicTokenizer's Chinese ranges. Deliberately excludes kana/hangul: HF only
    splits these, and adding more ranges changes the token stream."""
    return (
        (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)
    )


class ClipTokenizer:
    def __init__(self, vocab_path, merges_path, model_max_length=77, config_path=None):
        with open(vocab_path, "r", encoding="utf-8") as fh:
            self.encoder = json.load(fh)
        with open(merges_path, "r", encoding="utf-8") as fh:
            merges = fh.read().split("\n")
        merges = [tuple(m.split()) for m in merges[1:] if len(m.split()) == 2]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.cache = {}
        self.max_len = model_max_length

        # Special/added tokens short-circuit BPE. Defaults match a stock CLIP tokenizer;
        # the SD-Turbo config overrides pad_token with "!" (id 0).
        self.special = {"<|startoftext|>": BOS, "<|endoftext|>": EOS}
        self.pad_id = EOS
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            for tid, entry in (cfg.get("added_tokens_decoder") or {}).items():
                content = entry.get("content") if isinstance(entry, dict) else None
                if content is not None:
                    self.special[content] = int(tid)
            for key in ("pad_token", "bos_token", "eos_token", "unk_token"):
                tok = cfg.get(key)
                if not isinstance(tok, str) or tok in self.special:
                    continue
                # vocab stores word-final tokens as "<tok></w>"
                for cand in (tok, tok + "</w>"):
                    if cand in self.encoder:
                        self.special[tok] = self.encoder[cand]
                        break
            pad_tok = cfg.get("pad_token")
            if isinstance(pad_tok, str) and pad_tok in self.special:
                self.pad_id = self.special[pad_tok]
        # Longest added tokens first so the trie behaves greedily like HF's.
        toks = sorted(self.special, key=len, reverse=True)
        self._added_re = (
            re.compile("|".join(re.escape(t) for t in toks)) if toks else None
        )

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        self.cache[token] = " ".join(word)
        return self.cache[token]

    def _normalize(self, text):
        text = re.sub(r"\s+", " ", text.strip().lower())
        # HF's CLIPTokenizer falls back to BERT's BasicTokenizer when ftfy is absent,
        # which separates every CJK character so each becomes its own BPE token.
        out = []
        for ch in text:
            out.append(" %s " % ch if _is_cjk(ord(ch)) else ch)
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def _split_added(self, text):
        """Yield (chunk, is_added_token) pairs, matching added tokens greedily."""
        if not self._added_re:
            yield text, False
            return
        pos = 0
        for m in self._added_re.finditer(text):
            if m.start() > pos:
                yield text[pos:m.start()], False
            yield m.group(0), True
            pos = m.end()
        if pos < len(text):
            yield text[pos:], False

    def encode(self, text, max_len=None):
        """Return a list of token ids, padded to model_max_length."""
        max_len = max_len or self.max_len
        ids = []
        for chunk, is_added in self._split_added(self._normalize(text)):
            if is_added:
                ids.append(self.special[chunk])
                continue
            for token in _PAT.findall(chunk):
                if token in self.special:
                    ids.append(self.special[token])
                    continue
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                ids.extend(self.encoder[t] for t in self._bpe(token).split(" "))
        ids = [BOS] + ids + [EOS]
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def __call__(self, text):
        return self.encode(text)


def load(model_dir, model_max_length=77):
    return ClipTokenizer(
        os.path.join(model_dir, "vocab.json"),
        os.path.join(model_dir, "merges.txt"),
        model_max_length,
        os.path.join(model_dir, "tokenizer_config.json"),
    )

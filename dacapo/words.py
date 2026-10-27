"""Lyric words: group ACE-Step lyric tokens into words, normalise words, match lyric words to ASR words.

Pure Python (no ACE-Step import), so it runs in the main environment and in unit tests.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

# ASR output may spell a sung number as digits ("7 seagulls"); the lyrics spell it out.
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                "fifteen sixteen seventeen eighteen nineteen twenty").split()


@dataclass(frozen=True)
class Word:
    text: str     # surface form, e.g. "Carry"
    norm: str     # matching form, e.g. "carry"
    tokens: tuple  # indices into the pure lyric token list (ACE-Step's _extract_lyric_segment slice)


def _spell(m: re.Match) -> str:
    n = int(m.group())
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else m.group()


def norm_word(s: str) -> str:
    """Lower case, curly apostrophes folded, whole numbers 0-20 written out, keep a-z and inner apostrophes."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"\d+", _spell, s)
    return re.sub(r"[^a-z']", "", s).strip("'")


def tag_mask(tokens: list[str]) -> list[bool]:
    """True for tokens of a [...] structure tag. Same rule as ACE-Step's
    MusicLyricScorer._generate_token_type_mask: '[' opens, ']' closes, both ends belong to the tag."""
    out, inside = [], False
    for t in tokens:
        if "[" in t:
            inside = True
        is_tag = inside
        if "]" in t:
            inside, is_tag = False, True
        out.append(is_tag)
    return out


def group_words(tokens: list[str]) -> list[Word]:
    """Group decoded lyric tokens into words.

    A token starts a word if it begins with whitespace or follows the lyric start, a newline or a tag;
    otherwise it continues the current word ('Car' + 'ry'). Tag tokens and whitespace-only tokens
    belong to no word, and a token without letters (punctuation) never starts one.
    """
    groups: list[list[int]] = []
    boundary = True
    for i, (t, is_tag) in enumerate(zip(tokens, tag_mask(tokens))):
        if is_tag or t.strip() == "":
            boundary = True
            continue
        if (boundary or t[0].isspace()) and norm_word(t):
            groups.append([i])
        elif groups:
            groups[-1].append(i)
        boundary = t.endswith("\n")
    words = []
    for g in groups:
        text = "".join(tokens[j] for j in g).strip()
        words.append(Word(text=text, norm=norm_word(text), tokens=tuple(g)))
    return words


def lyric_words(lyrics: str) -> list[str]:
    """Normalised sung words of a lyric string: tag lines like [Verse] dropped."""
    return [w for w in (norm_word(x) for x in re.sub(r"\[[^\]]*\]", " ", lyrics).split()) if w]


def match_words(lyric: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """Order-preserving exact matches between normalised lyric words and ASR words
    (difflib matching blocks, autojunk off). Returns (lyric index, hyp index) pairs."""
    sm = difflib.SequenceMatcher(a=lyric, b=hyp, autojunk=False)
    return [(m.a + k, m.b + k) for m in sm.get_matching_blocks() for k in range(m.size)]

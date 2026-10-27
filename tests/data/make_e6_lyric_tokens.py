"""Write tests/data/e6_lyric_tokens.json: ACE-Step's lyric token ids and incremental decodes for the 16
read-out lyrics (Qwen3-Embedding tokenizer from the ACE-Step checkpoint, prompt from
ConditioningTextMixin._format_lyrics, slice from LyricAlignmentCommonMixin._extract_lyric_segment)."""
import json, sys
from pathlib import Path
draft, tok_dir, out, ace_repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, ace_repo)
sys.path.insert(0, draft); sys.path.insert(0, draft + "/experiments/e6_clock")
from transformers import AutoTokenizer
from acestep.core.scoring.dit_alignment import MusicStampsAligner
import readout
tok = AutoTokenizer.from_pretrained(tok_dir)
al = MusicStampsAligner(tok)
seen = {}
for it in readout.items():
    seen.setdefault(it["lyric_id"], it["lyrics"])
fx = {"tokenizer": "ace_ckpt/Qwen3-Embedding-0.6B", "language": "en", "lyrics": {}}
hdr = tok.encode("# Languages\nen\n\n# Lyric\n", add_special_tokens=False)
for lid, lyr in seen.items():
    ids = tok(f"# Languages\nen\n\n# Lyric\n{lyr}<|endoftext|>", padding="longest", truncation=True, max_length=2048,
              return_tensors="pt").input_ids[0].tolist()
    s, e = len(hdr), ids.index(151643)
    pure = ids[s:e]
    fx["lyrics"][lid] = {"text": lyr, "ids": ids, "start": s, "end": e, "tokens": al._decode_tokens_incrementally(pure)}
Path(out).write_text(json.dumps(fx, indent=1, ensure_ascii=False))
print(len(fx["lyrics"]), "lyrics written")

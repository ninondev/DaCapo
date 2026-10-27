"""E4 runner (ACE-Step environment): can a wordless guide fix the notes while the lyrics supply the words?

Arms per (item, seed), all with an a cappella caption and the item's lyrics:
  noguide            text2music, no audio condition (pitch should be ~chance)
  ctx_{g}_k{k}       cover-nofsq with the guide g as dense context for the first k of 8 steps,
                     then ACE-Step's own hand-off to text-only conditioning (audio_cover_strength=k/8)
  sde_{g}_n{x}       SDEdit: start from the noised guide latent at noise level 1-x, context silent
Swap mode (--mode swap) measures when things are decided: the context melody (or the lyrics)
is replaced by another item's from step s on, s = 1..7.

  python run_sing.py --device cuda --mode sweep --split all --seeds 1,2
  python run_sing.py --device cuda --mode swap  --seeds 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dacapo import ace as ace_lab

HERE = Path(__file__).parent
CAPTION = "a cappella, solo female vocal, no instruments, no drums, dry studio recording, clear lyrics"
KS = (1, 2, 3, 4, 6, 8)
NOISES = (0.10, 0.17, 0.25, 0.36)  # cover_noise_strength -> start t ~ .9 / .833 / .75 / .643 (shift-3 grid)


def arms(guides: list[str]) -> list[dict]:
    a = [{"name": "noguide", "task": "text2music"}]
    for g in guides:
        a += [{"name": f"ctx_{g}_k{k}", "task": "cover-nofsq", "guide": g, "cover_strength": k / 8} for k in KS]
        a += [{"name": f"sde_{g}_n{x:.2f}", "task": "cover-nofsq", "guide": g, "cover_strength": 0.0, "cover_noise": x}
              for x in NOISES]
    return a


def run_one(h, it: dict, arm: dict, seed: int, out: Path, **extra) -> dict:
    wav = out / f"{it['id']}__{arm['name']}__s{seed}.wav"
    if wav.exists():
        return {"skipped": True}
    r = ace_lab.generate(h, wav, CAPTION, it["duration"], seed, lyrics=it["lyrics"], bpm=int(it["melody"]["bpm"]),
                         keyscale=it["key"], task=arm["task"],
                         src=str(HERE / it["guides"][arm["guide"]]) if "guide" in arm else None,
                         cover_strength=arm.get("cover_strength", 1.0), cover_noise=arm.get("cover_noise", 0.0), **extra)
    return {"seconds": r["seconds"], "calls": r["n_denoiser_calls"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", choices=["sweep", "swap"], default="sweep")
    ap.add_argument("--split", default="all", help="dev | test | all")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--guides", default="clarinet,oohs")
    ap.add_argument("--arms", default="", help="comma list of arm names to keep (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    items = [it for it in man["items"] if args.split in ("all", it["split"])]
    if args.limit:
        items = items[: args.limit]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = HERE / f"outputs_{args.mode}"
    out.mkdir(exist_ok=True)
    log_path = out / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"runs": []}
    log["settings"] = {**vars(args), "caption": CAPTION, "ks": KS, "noises": NOISES, "shift": 3.0, "steps": 8}
    h = ace_lab.load(args.device)

    def record(entry):
        log["runs"].append(entry)
        log_path.write_text(json.dumps(log, indent=1))

    if args.mode == "sweep":
        keep = set(args.arms.split(",")) if args.arms else None
        for it in items:
            for seed in seeds:
                for arm in arms(args.guides.split(",")):
                    if keep and arm["name"] not in keep:
                        continue
                    r = run_one(h, it, arm, seed, out)
                    if not r.get("skipped"):
                        record({"id": it["id"], "arm": arm["name"], "seed": seed, **r})
                        print(f"{it['id']} {arm['name']:>20s} s{seed} {r['seconds']:6.1f}s", flush=True)
        return

    # swap mode: both endpoints of each test pair, then a switch at steps 1..7
    test = [it for it in man["items"] if it["split"] == "test"]
    pairs = [(test[k], test[(k + 1) % len(test)]) for k in range(0, len(test) - 1, 2)]
    if args.limit:
        pairs = pairs[: args.limit]
    for a, b in pairs:
        for seed in seeds:
            # melody swap: guide context (Voice Oohs, full strength) a -> b
            arm_a = {"name": "ctxfull_oohs", "task": "cover-nofsq", "guide": "oohs", "cover_strength": 1.0}
            wa = out / f"{a['id']}__{arm_a['name']}__s{seed}.wav"
            r = ace_lab.generate(h, wa, CAPTION, a["duration"], seed, lyrics=a["lyrics"], bpm=100, keyscale=a["key"],
                                 task="cover-nofsq", src=str(HERE / a["guides"]["oohs"]), capture=True)
            rb = ace_lab.generate(h, out / f"{b['id']}__melsrc_for_{a['id']}__s{seed}.wav", CAPTION, a["duration"], seed,
                                  lyrics=a["lyrics"], bpm=100, keyscale=a["key"], task="cover-nofsq",
                                  src=str(HERE / b["guides"]["oohs"]), capture=True)
            ctx_b = {"context_latents": rb["captured"]["context_latents"]}
            # guides give 372 or 373 context frames: swap_conditioning crops b's to a's or pads with silence latents
            fa, fb = r["captured"]["context_latents"].shape[1], ctx_b["context_latents"].shape[1]
            for s in range(1, 8):
                w = out / f"{a['id']}__melswap_to_{b['id']}_at{s}__s{seed}.wav"
                if not w.exists():
                    ace_lab.generate(h, w, CAPTION, a["duration"], seed, lyrics=a["lyrics"], bpm=100, keyscale=a["key"],
                                     task="cover-nofsq", src=str(HERE / a["guides"]["oohs"]),
                                     pre=ace_lab.swap_conditioning(s, ctx_b, silence_latent=h.silence_latent))
                record({"id": a["id"], "other": b["id"], "kind": "melody", "at": s, "seed": seed, "wav": w.name,
                        "frames": fa, "frames_other": fb})
            # lyrics swap: text2music a cappella, lyrics a -> b
            ace_lab.generate(h, out / f"{a['id']}__lyr_own__s{seed}.wav", CAPTION, a["duration"], seed,
                             lyrics=a["lyrics"], bpm=100, keyscale=a["key"])
            lb = ace_lab.generate(h, out / f"{a['id']}__lyrsrc_{b['id']}__s{seed}.wav", CAPTION, a["duration"], seed,
                                  lyrics=b["lyrics"], bpm=100, keyscale=a["key"], capture=True)
            enc_b = {k: v for k, v in lb["captured"].items() if k in ("encoder_hidden_states", "encoder_attention_mask")}
            for s in range(1, 8):
                w = out / f"{a['id']}__lyrswap_to_{b['id']}_at{s}__s{seed}.wav"
                if not w.exists():
                    ace_lab.generate(h, w, CAPTION, a["duration"], seed, lyrics=a["lyrics"], bpm=100, keyscale=a["key"],
                                     pre=ace_lab.swap_conditioning(s, enc_b))
                record({"id": a["id"], "other": b["id"], "kind": "lyrics", "at": s, "seed": seed, "wav": w.name})
            print(f"pair {a['id']}->{b['id']} s{seed} done", flush=True)


if __name__ == "__main__":
    main()

"""The listening analysis: the exact conditional randomization test, the side mapping, the sheet reader and the
Run A summaries, on synthetic keys and ratings (no real key is read here)."""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analyze_ratings as L  # noqa: E402

B2_HEADER = "pair,lyric,melody_right,words_clear,notes"


def brute_conditional(a, s):
    """Centred margin Dc = D - (2m - n) mean(a), over every side vector with the same m."""
    n, m = len(a), sum(x > 0 for x in s)

    def centred(sv):
        return sum(x * y for x, y in zip(a, sv)) - (2 * m - n) * sum(a) / n

    obs = centred(s)
    hits = total = 0
    for on_a in itertools.combinations(range(n), m):
        sp = [1 if i in on_a else -1 for i in range(n)]
        total += 1
        hits += abs(centred(sp)) >= abs(obs) - 1e-9
    return sum(x * y for x, y in zip(a, s)), hits / total


@pytest.mark.parametrize("seed", range(12))
def test_conditional_test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 11))
    a = [int(x) for x in rng.integers(-4, 5, size=n)]
    s = [int(x) for x in rng.choice([-1, 1], size=n)]
    res = L.signflip_conditional(a, s)
    d, p = brute_conditional(a, s)
    assert res["D"] == d and res["m"] == sum(x > 0 for x in s)
    assert math.isclose(res["p_two_sided"], p, rel_tol=1e-12)


def test_unconditional_reference_matches_brute_force():
    a, s = [2, -1, 3, 0, -2], [1, -1, 1, 1, -1]
    d = sum(x * y for x, y in zip(a, s))
    hits = sum(abs(sum(x * y for x, y in zip(a, sg))) >= abs(d) for sg in itertools.product((1, -1), repeat=5))
    assert math.isclose(L.signflip_conditional(a, s)["p_unconditional"], hits / 32)


@pytest.mark.parametrize("m", range(0, 25, 3))
@pytest.mark.parametrize("c", [1, -1, 7])
def test_position_preference_alone_never_gives_a_label(m, c):
    """Raters who answer the same side everywhere say nothing about the recipes; given the side split, p is 1."""
    s = [1] * m + [-1] * (24 - m)
    res = L.signflip_conditional([c] * 24, s)
    assert res["p_two_sided"] == 1.0 and res["D_centred"] == 0
    assert L.b2_label(res["z"], res["p_two_sided"]) == "no clear preference"


@pytest.mark.parametrize("m", [6, 9, 15, 18])
def test_effect_on_top_of_a_position_preference_is_found(m):
    """10 raters who like clip A, and the clip without the clock more: on pairs with S* on A, 6 A to 4 B (a = 2);
    on pairs with S* on B, 8 A to 2 B (a = 6). The uncentred margin can be 0; the centred one finds the effect."""
    s = [1] * m + [-1] * (24 - m)
    a = [2] * m + [6] * (24 - m)
    res = L.signflip_conditional(a, s)
    assert math.isclose(res["p_two_sided"], 1 / math.comb(24, m))
    assert L.b2_label(res["z"], res["p_two_sided"]) == "listeners favour no clock"
    mirror = L.signflip_conditional([6] * m + [2] * (24 - m), s)
    assert L.b2_label(mirror["z"], mirror["p_two_sided"]) == "listeners favour the clock"


def key(sides):
    """sides: per pair, the side that holds S* ('A' or 'B')."""
    pairs = {}
    for i, s in enumerate(sides, 1):
        o = "B" if s == "A" else "A"
        pairs[f"pair_{i:02d}"] = {"item": f"t{i:02d}", s: {"arm": "S*", "run_id": f"t{i:02d}_clock"},
                                  o: {"arm": "S*_b0", "run_id": f"t{i:02d}_none"}}
    return {"pairs": pairs}


def scores(n, wr=(0.6, 0.3), ctc=(0.2, 0.2), npa=(0.5, 0.4)):
    clips = []
    for i in range(1, n + 1):
        clips.append({"run_id": f"t{i:02d}_clock", "wr": wr[0], "wr_ctc": ctc[0], "npa_v": npa[0]})
        clips.append({"run_id": f"t{i:02d}_none", "wr": wr[1], "wr_ctc": ctc[1], "npa_v": npa[1]})
    return {"results": {"clips": clips}}


def sheet(tmp_path, name, answers, header=B2_HEADER, enc="utf-8", delim=",", ids=None, eol="\n"):
    lines = [header.replace(",", delim)]
    for i, (m, w) in enumerate(answers, 1):
        pid = ids[i - 1] if ids else f"pair_{i:02d}"
        lines.append(delim.join([pid, "some words", m, w, ""]))
    p = tmp_path / f"ratings_{name}.csv"
    p.write_bytes((eol.join(lines) + eol).encode(enc))
    return p


def load(paths, sides, **kw):
    raters, dropped, empty = L.load_raters(paths, kw.pop("exclude", ()), "pair", L.B2_QUESTIONS, L.B2_ANSWERS,
                                           {f"pair_{i:02d}" for i in range(1, len(sides) + 1)}, kw.pop("same_ok", ()))
    return raters, dropped, empty


def test_b2_maps_sides_to_recipes(tmp_path):
    sides = ["A", "B", "A", "B"]
    p = sheet(tmp_path, "r1", [("A", "B"), ("A", "B"), ("same", "neither"), ("", "x")])
    raters = load([p], sides)[0]
    res = L.analyse_b2(key(sides), raters, scores(4))
    m, w = res["questions"]["melody_right"], res["questions"]["words_clear"]
    assert m["counts"] == {"S*": 1, "S*_b0": 1, "same": 1, "neither": 0, "blank": 1, "invalid": 0}
    assert w["counts"] == {"S*": 1, "S*_b0": 1, "same": 0, "neither": 1, "blank": 0, "invalid": 1}
    assert res["per_pair"]["pair_01"]["melody_right"] == 1 and res["per_pair"]["pair_02"]["melody_right"] == -1
    assert res["per_pair"]["pair_01"]["words_clear"] == -1 and res["per_pair"]["pair_02"]["words_clear"] == 1
    assert w["test"]["D"] == 0 and m["test"]["D"] == 0 and w["test"]["m"] == 2
    assert res["raters_detail"]["r1"]["invalid_answers"] == {"x": 1}


@pytest.mark.parametrize("m", [3, 12, 21])
def test_b2_strong_preference_is_found(tmp_path, m):
    sides = ["A"] * m + ["B"] * (24 - m)
    paths = [sheet(tmp_path, n, [(s, s) for s in sides]) for n in ("r1", "r2")]
    paths[1].write_bytes(paths[1].read_bytes() + b"\n")       # two raters, two different files
    res = L.analyse_b2(key(sides), load(paths, sides)[0], scores(24))
    for q in L.B2_QUESTIONS:
        t = res["questions"][q]["test"]
        assert t["D"] == 48 and t["m"] == m and 0 < t["p_two_sided"] <= 2 / math.comb(24, m) + 1e-12
        assert res["questions"][q]["label"] == "listeners favour the clock"
        assert res["questions"][q]["balanced_share_for_clock"] == 1.0


def test_b2_machine_agreement(tmp_path):
    sides = ["A", "B"]
    sc = {"results": {"clips": [
        {"run_id": "t01_clock", "wr": 0.8, "wr_ctc": 0.1, "npa_v": 0.5},
        {"run_id": "t01_none", "wr": 0.2, "wr_ctc": 0.3, "npa_v": 0.5},
        {"run_id": "t02_clock", "wr": 0.6, "wr_ctc": 0.6, "npa_v": 0.9},
        {"run_id": "t02_none", "wr": 0.1, "wr_ctc": 0.1, "npa_v": 0.2}]}}
    p = sheet(tmp_path, "r", [("A", "A"), ("A", "A")])
    res = L.analyse_b2(key(sides), load([p], sides)[0], sc)
    w = res["questions"]["words_clear"]["agreement_with_machine"]
    assert w == {"whisper_wr": {"agree": 1, "n": 2, "share": 0.5}, "ctc_wr": {"agree": 0, "n": 2, "share": 0.0}}
    assert res["questions"]["melody_right"]["agreement_with_machine"] == {"npa_v": {"agree": 0, "n": 1, "share": 0.0}}


def test_key_and_scores_are_checked(tmp_path):
    k = key(["A"])
    p = sheet(tmp_path, "r", [("A", "A")])
    k["pairs"]["pair_01"]["B"]["arm"] = "S*_rand"
    with pytest.raises(ValueError):
        L.analyse_b2(k, load([p], ["A"])[0], scores(1))
    with pytest.raises(ValueError):
        L.analyse_b2(key(["A"]), load([p], ["A"])[0], {"results": {"clips": []}})


@pytest.mark.parametrize("enc,delim,eol", [("utf-8", ",", "\n"), ("utf-8-sig", ",", "\r\n"), ("gb18030", ",", "\n"),
                                           ("utf-8", ";", "\n"), ("utf-16", "\t", "\r\n"), ("utf-8", ",", "\r")])
def test_sheet_reader_formats(tmp_path, enc, delim, eol):
    p = sheet(tmp_path, "zh", [("都一样", " Ｂ "), ("都不行", "a段")], header="Pair,Lyric,Melody Right,Words-Clear,Notes",
              enc=enc, delim=delim, eol=eol)
    res = L.analyse_b2(key(["A", "A"]), load([p], ["A", "A"])[0], scores(2))
    m, w = res["questions"]["melody_right"]["counts"], res["questions"]["words_clear"]["counts"]
    assert m["same"] == 1 and m["neither"] == 1
    assert w["S*"] == 1 and w["S*_b0"] == 1


def test_ids_are_normalised_and_unknown_ids_refused(tmp_path):
    ids = ["Pair 1", "ｐａｉｒ＿０２", "03"]
    p = sheet(tmp_path, "r", [("A", "A")] * 3, ids=ids)
    raters = load([p], ["A"] * 3)[0]
    assert sorted(raters["r"]["answers"]) == ["pair_01", "pair_02", "pair_03"]
    bad = sheet(tmp_path, "s", [("A", "A")] * 2, ids=["pair_01", "pair_99"])
    with pytest.raises(ValueError):
        load([bad], ["A"] * 3)


def test_missing_rows_are_reported(tmp_path):
    p = sheet(tmp_path, "r", [("A", "A")] * 2)
    raters = load([p], ["A"] * 3)[0]
    assert raters["r"]["missing_ids"] == ["pair_03"]


def test_missing_column_is_refused_and_empty_sheet_left_out(tmp_path):
    p = sheet(tmp_path, "r", [("A", "A")], header="pair,lyric,melody,words,notes")
    with pytest.raises(ValueError):
        load([p], ["A"])
    q = sheet(tmp_path, "s", [("", "huh")])
    t = sheet(tmp_path, "t", [("A", "A")])
    raters, _, empty = load([q, t], ["A"])
    assert sorted(raters) == ["t"] and empty[0]["name"] == "s" and empty[0]["unreadable"] == {"huh": 1}


def test_row_with_answers_but_no_name_is_refused(tmp_path):
    p = tmp_path / "ratings_r.csv"
    p.write_text(B2_HEADER + "\npair_01,x,A,A,\n,x,B,B,\n")
    with pytest.raises(ValueError):
        load([p], ["A", "A"])


def test_duplicate_rows(tmp_path):
    same = sheet(tmp_path, "r", [("A", "B"), ("a", " b ")], ids=["pair_01", "pair_01"])
    assert load([same], ["A"])[0]["r"]["answers"]["pair_01"] == {"melody_right": "A", "words_clear": "B"}
    diff = sheet(tmp_path, "s", [("A", "B"), ("B", "B")], ids=["pair_01", "pair_01"])
    with pytest.raises(ValueError):
        load([diff], ["A"])


def test_extra_fields_and_blank_first_line(tmp_path):
    p = tmp_path / "ratings_r.csv"
    p.write_text("\n" + B2_HEADER + "\npair_01,words,A,B,clip A louder, B cleaner\n")
    raters = load([p], ["A"])[0]
    assert raters["r"]["answers"]["pair_01"] == {"melody_right": "A", "words_clear": "B"}


def test_rater_names_exclusion_and_duplicates(tmp_path):
    a = sheet(tmp_path, "amy", [("A", "A")])
    b = sheet(tmp_path, "bo", [("B", "B")])
    raters, dropped, _ = load([a, b], ["A"], exclude={"BO"})
    assert sorted(raters) == ["amy"] and list(dropped) == ["bo"] and dropped["bo"]["sha256"] == L.sha256(b)
    with pytest.raises(ValueError):
        load([a, b], ["A"], exclude={"cy"})
    copy = tmp_path / "ratings_amy (1).csv"
    copy.write_bytes(b.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        load([a, copy], ["A"])
    twin = tmp_path / "ratings_dan.csv"
    twin.write_bytes(a.read_bytes())
    with pytest.raises(ValueError):
        load([a, twin], ["A"])
    assert sorted(load([a, twin], ["A"], same_ok={"amy", "DAN"})[0]) == ["amy", "dan"]


def test_answer_tables():
    assert L.answer("不确定", L.B2_ANSWERS) == "blank"
    assert L.answer("No difference", L.B2_ANSWERS) == "same"
    assert L.answer("是的。", L.YES_NO) == "yes" and L.answer("×", L.YES_NO) == "no"
    assert L.answer("不清楚", L.YES_NO) == "invalid"
    assert L.answer("Ｙｅｓ", L.YES_NO) == "yes"


def test_perm_test_two_groups_matches_brute_force():
    a, b = [1.0, 0.5, 1.0, 0.0, 1.0], [0.0, 0.5, 0.0, 0.0, 1.0]
    res = L.perm_test_two_groups(a, b)
    pooled, obs = a + b, np.mean(a) - np.mean(b)
    hits = sum(abs(np.mean([pooled[i] for i in s]) - np.mean([pooled[i] for i in range(10) if i not in s])) >= abs(obs) - 1e-12
               for s in itertools.combinations(range(10), 5))
    assert res["n_perm"] == 252 and math.isclose(res["p_two_sided"], round(hits / 252, 6), rel_tol=1e-5)
    assert res["diff"] == round(obs, 6) and res["n_lit"] == 5 and res["n_ctl"] == 5


def test_earlier_units_follow_the_patterns():
    assert L.EARLIER["lit"] == {"A", "B"} and L.EARLIER["null"] == set()
    assert "T+2A" in L.EARLIER["rel+2"] and "T+4A" in L.EARLIER["seq"] and "T+5A" in L.EARLIER["ana"]
    assert set(L.EARLIER) == {"lit", "ctl", "rel+2", "rel+5", "rel-3", "seq", "ana", "null"}
    assert L.MOVED["seq"] == {"T+6A"} and L.MOVED["ana"] == {"T+5B"}


def runa_key():
    clips, n = {}, 0
    for cond in ("lit", "ctl", "null"):
        for d in range(2):
            n += 1
            clips[f"clip_{n:02d}"] = {"experiment": "e5", "id": f"m{d}", "cond": cond, "seed": f"d{d}",
                                      "phase_dir": "outputs_v2_main"}
    for i in range(2):
        n += 1
        clips[f"clip_{n:02d}"] = {"experiment": "e6", "id": f"item0{i}_r0", "set": "e4"}
    return {"clips": clips}


def runa_sheet(tmp_path, name, k, repeat_lit="yes"):
    rows = ["clip,vocals_present,words_intelligible,accompaniment,gap_repeats_earlier_motif,notes"]
    for c, v in sorted(k["clips"].items()):
        rep = repeat_lit if v.get("cond") == "lit" else "no" if v["experiment"] == "e5" else ""
        voc = "yes" if v["experiment"] == "e6" else "no"
        rows.append(f"{c},{voc},{voc},no,{rep},")
    p = tmp_path / f"ratings_{name}.csv"
    p.write_text("\n".join(rows) + "\n")
    return p


E5 = {"results": {"rows": [{"id": f"m{d}", "cond": c, "seed": f"d{d}", "choice": "B" if c == "lit" else "none"}
                           for c in ("lit", "ctl", "null") for d in range(2)]}}
E6 = {"results": {"rows": [{"id": "item00_r0", "vocal": True, "n_matched": 9, "n_words": 12},
                           {"id": "item01_r0", "vocal": True, "n_matched": 2, "n_words": 12}]}}


def load_a(paths, k, exclude=()):
    return L.load_raters(paths, exclude, "clip", L.RUNA_QUESTIONS, L.YES_NO, set(k["clips"]))[:2]


def test_runa_summaries(tmp_path):
    k = runa_key()
    raters, _ = load_a([runa_sheet(tmp_path, "r", k)], k)
    res = L.analyse_runa(k, raters, E5, E6)
    assert res["yes_rates"]["e5:lit"]["gap_repeats_earlier_motif"] == 1.0
    assert res["yes_rates"]["e5:ctl"]["gap_repeats_earlier_motif"] == 0.0
    assert "gap_repeats_earlier_motif" not in res["yes_rates"]["e6:all"]
    t = res["tests"]["repeat_lit_minus_ctl"]
    assert t["diff"] == 1.0 and t["n_lit"] == 2 and t["n_ctl"] == 2
    s = res["agreement"]["repeat_vs_tracker_strict"]
    assert s["all"]["agree"] == 6 and s["all"]["n"] == 6 and s["primary"]["n"] == 6
    assert res["agreement"]["e6_vocals_present"]["agree"] == 2
    assert res["agreement"]["e6_words_intelligible"] == {"n": 2, "agree": 1, "kappa": 0.0}


def test_runa_owner_sheet_is_tested_apart(tmp_path):
    k = runa_key()
    paths = [runa_sheet(tmp_path, "owner", k), runa_sheet(tmp_path, "friend", k, repeat_lit="no")]
    res = L.analyse_runa(k, load_a(paths, k)[0], E5, E6, owner="Owner")
    assert res["tests"]["repeat_lit_minus_ctl"]["diff"] == 0.5
    assert res["tests"]["repeat_lit_minus_ctl_without_owner"]["diff"] == 0.0
    with pytest.raises(ValueError):
        L.analyse_runa(k, load_a(paths, k)[0], E5, E6, owner="nobody")


def test_runa_inputs_are_checked(tmp_path):
    k = runa_key()
    raters, _ = load_a([runa_sheet(tmp_path, "r", k)], k)
    with pytest.raises(ValueError):
        L.analyse_runa(k, raters, {"results": {"rows": []}}, E6)
    with pytest.raises(ValueError):
        L.analyse_runa(k, raters, E5, {"results": {"rows": []}})


def test_tracker_ties_are_left_out(tmp_path):
    k = runa_key()
    e5 = json.loads(json.dumps(E5))
    e5["results"]["rows"][0]["choice"] = "tie"
    res = L.analyse_runa(k, load_a([runa_sheet(tmp_path, "r", k)], k)[0], e5, E6)
    assert res["agreement"]["repeat_vs_tracker_strict"]["all"]["n"] == 5


def test_cli_writes_provenance(tmp_path):
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps(key(["A", "B"])))
    sp = tmp_path / "scores.json"
    sp.write_text(json.dumps(scores(2)))
    r = sheet(tmp_path, "r1", [("A", "B"), ("B", "A")])
    out = tmp_path / "sub" / "out.json"
    assert L.main(["b2", "--key", str(kp), "--ratings", str(r), "--scores", str(sp), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["provenance"]["key_sha256"] == L.sha256(kp)
    assert res["provenance"]["ratings"]["r1"]["sha256"] == L.sha256(r)
    ex = tmp_path / "exclusions.json"
    ex.write_text(json.dumps([{"name": "r1", "reason": "heard the demo", "date": "2026-10-31"}]))
    with pytest.raises(ValueError):          # nobody left after the exclusion
        L.main(["b2", "--key", str(kp), "--ratings", str(r), "--scores", str(sp), "--exclude-file", str(ex),
                "--out", str(out)])
    with pytest.raises(SystemExit):
        L.main(["b2", "--key", str(kp), "--ratings", str(r), "--out", str(out)])


def test_check_mode_needs_no_key(tmp_path, capsys):
    tpl = tmp_path / "ratings.csv"
    tpl.write_text(B2_HEADER + "\n" + "".join(f"pair_{i:02d},x,,,\n" for i in range(1, 4)))
    good = sheet(tmp_path, "good", [("A", "same"), ("都一样", "B"), ("", "")])
    assert L.main(["check", "--template", str(tpl), "--ratings", str(good)]) == 0
    bad = sheet(tmp_path, "bad", [("A", "A")] * 2, ids=["pair_01", "pair_09"])
    assert L.main(["check", "--template", str(tpl), "--ratings", str(good), str(bad)]) == 1
    assert "STOP" in capsys.readouterr().out


def test_runa_owner_flag_is_required(tmp_path):
    k = runa_key()
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps(k))
    e5p, e6p = tmp_path / "e5.json", tmp_path / "e6.json"
    e5p.write_text(json.dumps(E5))
    e6p.write_text(json.dumps(E6))
    r = runa_sheet(tmp_path, "r", k)
    args = ["runa", "--key", str(kp), "--ratings", str(r), "--e5", str(e5p), "--e6", str(e6p), "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit):
        L.main(args)
    assert L.main(args + ["--owner", "none"]) == 0
    assert json.loads((tmp_path / "o.json").read_text())["owner_sheet"] == "none"

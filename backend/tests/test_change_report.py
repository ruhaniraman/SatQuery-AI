"""Evidence-built comparison: subtle-change pass, LoRA scan comparison, assembled answer, endpoint wiring."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.scan_compare import format_scan_change, summarize_scan_change  # noqa: E402
from agent_manager.scene_stats import comparability_warning, compute_scene_stats  # noqa: E402
from change_detection.cdvqa_engine import (  # noqa: E402
    ChangeDetectionEngine,
    _odd_kernel,
    describe_location as engine_location,
    find_subtle_regions,
)
from change_detection.change_report import (  # noqa: E402
    compare_report_enabled,
    _clip,
    clean_caption,
    compose_change_answer,
    compose_change_details,
    is_generic_question,
    lora_compare_enabled,
    match_scans_to_regions,
    region_items,
)
from change_detection.locations import describe_location  # noqa: E402
from geospatial_preprocessing.standard_alignment import align_standard_images, align_with_status  # noqa: E402
from test_endpoint_wiring import client, png_bytes, post  # noqa: E402,F401


def textured(seed, h=400, w=400):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(60, 160, (h // 8, w // 8, 3)).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)


def never(*_a, **_k):
    raise AssertionError("the model must not write the comparison in evidence mode")


def run(a, b, **kw):
    return ChangeDetectionEngine().detect(a, b, never, compose_report=True, **kw)


# ------------------------------------------------------------------ helpers / env

def test_describe_location_is_shared_and_still_importable_from_the_engine():
    assert engine_location is describe_location and describe_location(10, 10, 300, 300) == "upper-left"


def test_env_switches_default_on_and_can_be_turned_off(monkeypatch):
    for var, fn in (("CHANGE_EVIDENCE_REPORT", compare_report_enabled), ("LORA_COMPARE", lora_compare_enabled)):
        monkeypatch.delenv(var, raising=False)
        assert fn()
        monkeypatch.setenv(var, "off")
        assert not fn()


def test_generic_questions_are_recognised():
    assert is_generic_question("") and is_generic_question("what changed?") and is_generic_question("Any differences?")
    assert not is_generic_question("how many trucks are visible?")


# ------------------------------------------------------------------ LoRA scan comparison

def grid(*hot):
    g = np.full((4, 4), 0.1)
    for r, c in hot:
        g[r, c] = 0.9
    return g


def test_scan_change_reports_where_cells_flipped():
    s = summarize_scan_change("mining", grid((0, 3)), grid((0, 3), (3, 0), (3, 3)))
    assert (s["positive_before"], s["positive_after"], s["total"]) == (1, 3, 16)
    assert s["gained"] == ["lower-left (row 4, col 1)", "lower-right (row 4, col 4)"] and s["lost"] == []
    text = format_scan_change(s)
    assert "1 of 16 tiles positive before, 3 after" in text
    assert "Clearly newly positive (score up by 0.4 or more): lower-left (row 4, col 1), lower-right (row 4, col 4)" in text
    assert "Clearly dropped: none" in text and "smaller margin" not in text


def test_scan_change_lost_cells_and_noise_near_the_threshold():
    s = summarize_scan_change("agriculture", grid((1, 1)), grid())
    assert s["lost"] and s["gained"] == []
    before, after = np.full((4, 4), 0.48), np.full((4, 4), 0.52)      # crosses 0.5 but barely moved
    assert summarize_scan_change("mining", before, after)["gained"] == []


def test_cells_sharing_a_coarse_word_stay_distinguishable():
    # (row 3, col 2) and (row 3, col 3) are both 'centre' in 3x3 words
    s = summarize_scan_change('mining', grid((2, 1)), grid((2, 2)))
    assert s['lost'] == ['centre (row 3, col 2)'] and s['gained'] == ['centre (row 3, col 3)']


def test_a_scan_that_says_yes_almost_everywhere_is_flagged_as_saturated():
    many = [(r, c) for r in range(3) for c in range(3)]              # 9 of 16 already positive
    s = summarize_scan_change('mining', grid(*many), grid(*many, (3, 3)))
    assert s['saturated'] and 'separates the dates poorly' in format_scan_change(s)
    calm = summarize_scan_change('mining', grid((0, 0)), grid((0, 0), (3, 3)))
    assert not calm['saturated'] and 'separates' not in format_scan_change(calm)


def test_a_modest_score_movement_is_not_reported_as_a_flip():
    before, after = np.full((4, 4), 0.3), np.full((4, 4), 0.3)
    after[1, 1] = 0.75                                                # moved 0.45: counts
    assert summarize_scan_change('mining', before, after)['gained'] == ['centre (row 2, col 2)']
    after[1, 1] = 0.6                                                 # moved 0.3: below the 0.4 minimum
    assert summarize_scan_change('mining', before, after)['gained'] == []


def test_tiles_that_crossed_by_a_small_margin_are_counted_not_silently_dropped():
    before, after = np.full((4, 4), 0.3), np.full((4, 4), 0.3)
    after[0, 0], after[0, 1] = 0.9, 0.55                              # +0.6 (clear) and +0.25 (quiet)
    s = summarize_scan_change('deforestation', before, after)
    assert (s['crossed_up'], len(s['gained_cells'])) == (2, 1) and s['gained_cells'] == [(0, 0)]
    text = format_scan_change(s)
    assert '0 of 16 tiles positive before, 2 after' in text
    assert '1 more tile(s) crossed to positive by a smaller margin' in text
    assert s['grid'] == (4, 4)


def test_scan_change_rejects_mismatched_grids():
    with pytest.raises(ValueError):
        summarize_scan_change("mining", np.zeros((4, 4)), np.zeros((3, 3)))


# ------------------------------------------------------------------ the assembled answer

def _two_regions():
    return region_items([(10, 10, 80, 80, 0.06)], [(300, 300, 60, 60, 0.02)], (400, 400),
                        [("bare pit", "pit with pond"), None])


def test_the_answer_shows_only_warnings_changed_areas_and_before_after():
    text = compose_change_answer(_two_regions(), ["Image B could not be registered (x)."], "how many trucks?")
    assert text.startswith("WARNING: Image B could not be registered (x).")
    assert "Changed areas" in text and "Major change, upper-left (~6% of the scene)." in text
    assert "Before: bare pit" in text and "After: pit with pond" in text
    assert "Possible smaller change, lower-right (~2% of the scene)." in text
    for hidden in ("Measured", "scan", "LoRA", "single-image", "Comparison summary", "captured", "Assembled", "WEAK"):
        assert hidden not in text, hidden
    assert "Your question (how many trucks?) is not answered separately" in text


def test_each_area_carries_its_measured_shift_after_the_captions():
    items = region_items([(10, 10, 80, 80, 0.06)], [], (400, 400), [("bare pit", "pit with pond")],
                         ["green-dominant pixels 71% -> 22%"])
    lines = compose_change_answer(items).splitlines()
    assert lines[-3:] == ["    Before: bare pit", "    After: pit with pond",
                          "    Measured: green-dominant pixels 71% -> 22%."]
    no_caption = compose_change_answer(region_items([(10, 10, 80, 80, 0.06)], [], (400, 400), None, ["x -> y"]))
    assert no_caption.splitlines()[-1] == "    Measured: x -> y." and "Before:" not in no_caption
    assert "Measured" not in compose_change_answer(region_items([(10, 10, 80, 80, 0.06)], [], (400, 400)))


def test_a_subtle_area_becomes_likely_only_when_a_scan_flags_the_same_place():
    subtle = region_items([], [(110, 110, 80, 80, 0.06)], (400, 400))
    assert "- Possible smaller change, centre (~6% of the scene)." in compose_change_answer(subtle)
    backed =match_scans_to_regions(region_items([], [(110, 110, 80, 80, 0.06)], (400, 400)), (400, 400),
                                    [summarize_scan_change("deforestation", grid(), grid((1, 1)))])
    text = compose_change_answer(backed)
    assert "- Likely change, " in text and "Also flagged by the deforestation scan." in text
    assert "Possible smaller change" not in text
    lone = match_scans_to_regions(region_items([], [(110, 110, 80, 80, 0.06)], (400, 400)), (400, 400),
                                  [summarize_scan_change("deforestation", grid(), grid((3, 3)))])
    assert "Possible smaller change" in compose_change_answer(lone) and "Likely" not in compose_change_answer(lone)


def test_a_major_area_stays_major_even_when_a_scan_agrees():
    items = match_scans_to_regions(region_items([(110, 110, 80, 80, 0.06)], [], (400, 400)), (400, 400),
                                   [summarize_scan_change("mining", grid(), grid((1, 1)))])
    text = compose_change_answer(items)
    assert "- Major change, " in text and "Also flagged by the mining scan." in text and "Likely" not in text


def test_the_measured_line_is_caveated_only_when_a_warning_is_present():
    items = region_items([(10, 10, 80, 80, 0.06)], [], (400, 400), None, ["green-dominant pixels 0% -> 18%"])
    warned = compose_change_answer(items, ["Image B could not be registered."])
    assert "    Measured: green-dominant pixels 0% -> 18%. This may reflect the two images differing in source, not the ground." in warned
    plain = compose_change_answer(items)
    assert plain.splitlines()[-1] == "    Measured: green-dominant pixels 0% -> 18%." and "differing in source" not in plain


def test_details_use_the_same_labels():
    items = match_scans_to_regions(region_items([], [(110, 110, 80, 80, 0.06)], (400, 400)), (400, 400),
                                   [summarize_scan_change("deforestation", grid(), grid((1, 1)))])
    d = compose_change_details(items, "", [summarize_scan_change("deforestation", grid(), grid((1, 1)))])
    assert d["Flagged regions"][0].startswith("Likely change, ")
    assert "'Likely' means a specialist scan also flagged the same place" in d["Flagged regions"][-1]


def test_answer_without_regions_is_one_plain_line_or_an_honest_warning_line():
    assert compose_change_answer([]) == "Changed areas\n- No changed area was found."
    warned = compose_change_answer([], ["some warning"])
    assert warned.startswith("WARNING: some warning") and "NOT evidence that nothing changed" in warned


def test_a_generic_question_adds_no_note_to_the_answer():
    assert "Your question" not in compose_change_answer(_two_regions(), (), "what changed?")


def test_a_scan_tile_overlapping_a_region_is_reported_as_corroboration():
    items = region_items([(110, 110, 80, 80, 0.04)], [], (400, 400))          # tile (row 2, col 2) = x 100-200, y 100-200
    match_scans_to_regions(items, (400, 400), [summarize_scan_change("deforestation", grid(), grid((1, 1)))])
    assert items[0]["scans"] == ["deforestation"]
    assert "Also flagged by the deforestation scan." in compose_change_answer(items)


def test_scans_elsewhere_or_only_touching_a_corner_are_not_corroboration():
    items = region_items([(190, 190, 30, 30, 0.01)], [], (400, 400))          # only a 10x10 corner of tile (1, 1)
    s = [summarize_scan_change("mining", grid(), grid((1, 1))), summarize_scan_change("agriculture", grid(), grid((3, 3)))]
    match_scans_to_regions(items, (400, 400), s)
    assert items[0]["scans"] == [] and "Also flagged" not in compose_change_answer(items)


def test_several_scans_are_named_together():
    items = region_items([(110, 110, 80, 80, 0.04)], [], (400, 400))
    s = [summarize_scan_change(n, grid(), grid((1, 1))) for n in ("mining", "deforestation", "agriculture")]
    match_scans_to_regions(items, (400, 400), s)
    assert "Also flagged by the mining, deforestation and agriculture scans." in compose_change_answer(items)


def test_details_hold_everything_the_answer_leaves_out():
    items = match_scans_to_regions(_two_regions(), (400, 400), [summarize_scan_change("mining", grid(), grid((0, 0)))])
    summaries = [summarize_scan_change("mining", grid(), grid((0, 0)))]
    d = compose_change_details(items, "green up.", summaries, ("2019-01-10", "2024-01-10"), "optical", ["a warning"])
    assert list(d)[:2] == ["How this was produced", "Image order"]
    assert "The model did not write the conclusion" in d["How this was produced"][0]
    assert d["Image order"] == ["Before: captured 2019-01-10. After: captured 2024-01-10."]
    assert d["Measured overall shift"][0] == "Pixel counts, not materials: green up."
    assert "differing in source, zoom, haze or exposure" in d["Measured overall shift"][1]      # caveat: a warning exists
    assert d["Specialist scans on both dates"][0].startswith("LoRA adapter scores") and "WEAK evidence" in d["Specialist scans on both dates"][0]
    assert d["Specialist scans on both dates"][1].startswith("mining / extraction pits: 0 of 16 tiles positive before, 1 after")
    assert d["Agreement between scans and flagged regions"] == [
        "upper-left: the mining scan also turned positive on a tile overlapping this area."]
    assert any("subtler colour differences" in line for line in d["Flagged regions"])


def test_details_shift_has_no_caveat_without_warnings_and_dates_can_be_unknown():
    d = compose_change_details([], "green up.", [], None, "optical", [])
    assert d["Measured overall shift"] == ["Pixel counts, not materials: green up."]
    assert "Capture dates are unknown" in d["Image order"][0]
    assert "Flagged regions" not in d and "Specialist scans on both dates" not in d
    assert "SAR note" in compose_change_details([], "", [], None, "sar", [])


def test_details_have_no_agreement_section_when_nothing_was_flagged():
    d = compose_change_details([], "", [summarize_scan_change("mining", grid(), grid((3, 3)))], None, "optical", [])
    assert "Specialist scans on both dates" in d and "Agreement between scans and flagged regions" not in d


def test_details_say_when_no_scan_backs_the_flagged_regions():
    items = region_items([(10, 10, 80, 80, 0.06)], [], (400, 400))
    d = compose_change_details(items, "", [summarize_scan_change("mining", grid(), grid((3, 3)))], None, "optical", [])
    assert d["Agreement between scans and flagged regions"] == [
        "No scan flip overlaps a flagged region, so nothing independently backs the flagged areas."]


def test_pdf_prints_the_comparison_details_as_their_own_section():
    from pdf_report_generator import generate_pdf_report
    from pypdf import PdfReader
    trace = {"pipeline_id": "x", "execution_status": "completed", "telemetry": {"comparison_details": {
        "Measured overall shift": ["green up (1% to 9%)"],
        "Agreement between scans and flagged regions": ["centre: the deforestation scan also turned positive"]}}}
    pdf = generate_pdf_report(query="q", answer="Changed areas", agent_execution_trace=trace,
                              image_source=None, chat_history=[])
    text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
    assert "COMPARISON DETAILS" in text and "MEASURED OVERALL SHIFT" in text
    assert "deforestation scan also turned positive" in text
    assert text.count("green up") == 1                     # printed once, not again in the raw trace dump


# ------------------------------------------------------------------ caption cleaning

def test_the_stock_opener_list_numbering_and_absence_filler_are_removed_but_the_content_stays():
    raw = ("The image shows a satellite view of a landscape with various land cover types. Visible features include: "
           "1. Land Cover: The area appears to be a mix of vegetation and bare ground. "
           "2. Water: There is no visible water body in the image. "
           "3. Bare or Excavated Ground: The central part of the image shows a large, irregularly shaped area that "
           "appears to be bare or excavated ground.")
    assert clean_caption(raw) == ("The area appears to be a mix of vegetation and bare ground. The central part of the "
                                  "image shows a large, irregularly shaped area that appears to be bare or excavated ground.")


def test_only_the_absence_sentences_are_dropped_from_plain_prose():
    raw = ("The image shows a central area with land cover, water, and bare or excavated ground. "
           "There are no visible buildings or roads in the image.")
    assert clean_caption(raw) == "The image shows a central area with land cover, water, and bare or excavated ground."


def test_a_bare_numbered_list_becomes_one_sentence_and_clean_prose_is_untouched():
    assert clean_caption("1. Bare or excavated ground 2. Buildings 3. Roads") == "Noted: Bare or excavated ground, Buildings, Roads."
    prose = "Dense green canopy with a pale bare clearing in its middle. A track runs through the clearing."
    assert clean_caption(prose) == prose


def test_a_caption_that_is_only_absences_is_kept_rather_than_emptied():
    raw = "There is no visible water. Nothing visible here."
    assert clean_caption(raw) == raw
    assert clean_caption("") == "" and clean_caption(None) == ""


def test_a_longer_caption_is_not_cut_short_by_cleaning():
    long_prose = " ".join(f"Sentence number {i} describes the bare pale ground in some detail." for i in range(7))
    assert clean_caption(long_prose) == long_prose                       # 7 sentences survive intact
    assert len(_clip(long_prose, 500)) > 300                             # and a fuller caption is allowed on screen


def test_the_answer_shows_cleaned_captions():
    items = region_items([(10, 10, 80, 80, 0.06)], [], (400, 400),
                         [("1. Land Cover: Bare pale ground. 2. Water: There is no visible water.", "Bare pale ground.")])
    text = compose_change_answer(items)
    assert "    Before: Bare pale ground." in text and "1." not in text and "no visible water" not in text


# ------------------------------------------------------------------ caption trimming

def test_long_captions_are_cut_at_a_sentence_end_not_mid_word():
    text = 'Bare ground is visible. ' * 30
    out = _clip(text, 100)
    assert out.endswith('visible.') and len(out) <= 100 and '…' not in out


def test_a_list_number_is_not_mistaken_for_a_sentence_end():
    text = ('1. Land cover is mixed. 2. Water is visible. 3. Bare ground is seen. '
            '4. Roads run through the area and continue on.')
    out = _clip(text, text.index('4.') + 3)              # the cut lands right after '4. ' (the real failure)
    assert out == '1. Land cover is mixed. 2. Water is visible. 3. Bare ground is seen.'
    assert not out.rstrip().endswith(('4.', '3.', '2.', '1.'))


def test_a_caption_without_sentence_ends_is_cut_at_a_word_and_marked():
    out = _clip('word ' * 100, 50)
    assert out.endswith('…') and out[:-1].split()[-1] == 'word' and len(out) <= 51
    assert _clip('short one.', 100) == 'short one.'


# ------------------------------------------------------------------ subtle-change pass (through detect)

def test_local_subtle_difference_is_flagged_as_possible_not_major():
    a = textured(1)
    b = a.copy()
    b[100:260, 100:260] = np.clip(a[100:260, 100:260].astype(int) + 28, 0, 255).astype(np.uint8)
    res = run(a, b)
    assert res.major_regions_detected == 0 and res.subtle_regions_detected >= 1
    assert "Possible smaller change" in res.explanation
    yellow = (res.overlay_image == np.array([0, 255, 255], np.uint8)).all(axis=2)
    assert yellow.any()                                    # drawn on the evidence image


def _cored_scene(halo_rows=(120, 280), halo_cols=(120, 280)):
    """Strong core (+30) inside a fainter halo (+11): the halo is below the strict level but above the loose one."""
    a = textured(1)
    b = a.astype(int)
    b[halo_rows[0]:halo_rows[1], halo_cols[0]:halo_cols[1]] += 11
    b[150:250, 150:250] += 19                              # core 100 px, halo 160 px: area x2.6, inside the 3x limit
    return a, np.clip(b, 0, 255).astype(np.uint8)


def subtle_boxes(a, b):
    """The subtle pass's boxes (x, y, w, h, frac), measured directly (the drawn label text is ~150 px wide)."""
    h, w = a.shape[:2]
    scale = (h * w) ** 0.5 / 512
    return find_subtle_regions(a, b, None, None, _odd_kernel(21, max(scale, 1.0)), scale)


def test_a_region_grows_from_its_strong_core_to_the_whole_changed_patch():
    a, b = _cored_scene()
    boxes = subtle_boxes(a, b)
    assert len(boxes) == 1 and run(a, b).major_regions_detected == 0
    assert boxes[0][2] >= 140 and boxes[0][3] >= 140       # the 100 px core alone gives roughly 105


def test_growth_that_would_leak_across_the_scene_is_refused():
    a = textured(1)
    b = a.astype(int)
    b[0:120, :] += 11                                      # faint change across a third of the scene ...
    b[60:130, 60:130] += 19                                # ... touching one 70 px core
    boxes = subtle_boxes(a, np.clip(b, 0, 255).astype(np.uint8))
    assert len(boxes) == 1 and boxes[0][2] < 200           # only the core, not the 400 px wide faint band


def test_a_faint_change_with_no_strong_core_is_not_flagged_at_all():
    a = textured(1)
    b = a.astype(int)
    b[100:300, 100:300] += 11                              # above the loose level, below the strict one
    assert run(a, np.clip(b, 0, 255).astype(np.uint8)).subtle_regions_detected == 0


def test_overall_shift_between_dates_is_not_a_local_change():
    a = textured(1)
    b = np.clip(a.astype(int) + 30, 0, 255).astype(np.uint8)     # whole scene brighter (season / exposure)
    res = run(a, b)
    assert res.subtle_regions_detected == 0 and res.major_regions_detected == 0


def test_identical_images_and_sar_have_no_subtle_regions():
    a = textured(2)
    assert run(a, a.copy()).subtle_regions_detected == 0
    b = a.copy()
    b[100:260, 100:260] = np.clip(a[100:260, 100:260].astype(int) + 28, 0, 255).astype(np.uint8)
    assert run(a, b, modality="sar").subtle_regions_detected == 0


def test_major_regions_are_not_repeated_as_possible():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0
    res = run(a, b)
    assert res.major_regions_detected == 1 and res.subtle_regions_detected == 0


# ------------------------------------------------------------------ captions

def test_only_flagged_region_crops_are_captioned_never_the_whole_images():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0
    calls = []

    def notes(prompt, image, system=None):
        calls.append((prompt, image.shape, system))
        return "A pit with a pond. Water is visible."

    res = run(a, b, notes_fn=notes)
    assert res.major_regions_detected == 1 and len(calls) == 2         # one crop before, one after
    assert all(shape[0] < 400 and shape[1] < 400 for _, shape, _ in calls) and all(sys is None for _, _, sys in calls)
    assert "Before: A pit with a pond. Water is visible." in res.explanation
    assert "single-image look" not in res.explanation


def test_no_regions_means_no_model_calls_at_all():
    a = textured(2)
    calls = []
    res = run(a, a.copy(), notes_fn=lambda p, i, system=None: calls.append(p) or "x")
    assert calls == [] and "No changed area was found" in res.explanation


def test_detect_measures_each_flagged_area_before_and_after_from_the_pixels():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0                                # the area goes black
    res = run(a, b)
    assert res.major_regions_detected == 1
    measured = [l for l in res.explanation.splitlines() if l.strip().startswith('Measured:')]
    assert len(measured) == 1
    import re
    before_b, after_b = map(int, re.search(r'average brightness (\d+) -> (\d+)', measured[0]).groups())
    assert before_b > 60 and after_b < before_b / 4                  # the box is mostly the blacked-out area
    green_b, green_a = map(int, re.search(r'green-dominant pixels (\d+)% -> (\d+)%', measured[0]).groups())
    assert green_a < green_b
    assert 'pixel counts, not materials' in ' '.join(res.details['Flagged regions'])
    quiet = run(a, a.copy())
    assert 'Measured:' not in quiet.explanation


def test_failing_captions_still_produce_a_complete_answer():
    a = textured(1)
    b = a.copy()
    b[150:300, 200:350] = 0

    def boom(prompt, image, system=None):
        raise RuntimeError("out of memory")

    res = run(a, b, notes_fn=boom)
    assert "Major change" in res.explanation and "Before:" not in res.explanation


# ------------------------------------------------------------------ endpoint wiring

def two_pngs():
    img = png_bytes(np.full((64, 64, 3), 90, np.uint8))
    return [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))]


def test_endpoint_builds_the_answer_from_evidence_and_never_sends_the_stitched_pair(client, monkeypatch):
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "1")
    monkeypatch.setenv("DESCRIBE_FIRST", "1")
    scanned = []

    def scan_scores(img, adapter):
        scanned.append((adapter, img.shape))
        return grid((0, 0)) if len(scanned) % 2 else grid((0, 0), (2, 2))

    client.agent.scan_scores = scan_scores
    img = png_bytes(sharp(3, 160))                        # feature-rich, so alignment can verify it (no warning)
    r = post(client, [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Changed areas\n- No changed area was found."      # short and plain
    assert [c[0] for c in scanned] == ["mining", "mining", "deforestation", "deforestation", "agriculture", "agriculture"]
    assert client.vlm_calls == []                        # no flagged region -> no caption calls, and never the stitched pair
    tel = body["agent_execution_trace"]["telemetry"]
    assert tel["answer_source"].startswith("assembled by code")
    assert tel["lora_scans_compared"] == ["mining", "deforestation", "agriculture"]
    assert "EvidenceReport" in body["agent_execution_trace"]["nodes_traversed"]
    scans = tel["comparison_details"]["Specialist scans on both dates"]          # the rest lives in the details
    assert any("mining / extraction pits: 1 of 16 tiles positive before, 2 after" in line for line in scans)
    assert "Measured overall shift" in tel["comparison_details"] and "scan" not in body["answer"]


def test_endpoint_survives_a_failing_scan_and_skips_scans_for_sar(client, monkeypatch):
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "1")

    def flaky(img, adapter):
        if adapter == "deforestation":
            raise RuntimeError("no gpu")
        return grid()

    client.agent.scan_scores = flaky
    tel = post(client, two_pngs()).json()["agent_execution_trace"]["telemetry"]
    assert tel["lora_scans_compared"] == ["mining", "agriculture"]

    client.agent.scan_scores = lambda img, adapter: (_ for _ in ()).throw(AssertionError("SAR must not be scanned"))
    sar = post(client, two_pngs(), modality_a="sar", modality_b="sar")
    assert sar.status_code == 200 and sar.json()["agent_execution_trace"]["telemetry"]["lora_scans_compared"] == []


def test_endpoint_lora_compare_can_be_turned_off(client, monkeypatch):
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "1")
    monkeypatch.setenv("LORA_COMPARE", "0")
    client.agent.scan_scores = lambda img, adapter: (_ for _ in ()).throw(AssertionError("scans are off"))
    r = post(client, two_pngs())
    assert r.status_code == 200 and r.json()["agent_execution_trace"]["telemetry"]["lora_scans_compared"] == []


def test_legacy_mode_still_asks_the_model_about_the_stitched_pair(client):
    r = post(client, two_pngs())                            # fixture default: CHANGE_EVIDENCE_REPORT=0
    assert r.status_code == 200 and r.json()["answer"] == "VLM says things changed."
    assert client.vlm_calls[0][1].shape[1] == 128
    assert r.json()["agent_execution_trace"]["telemetry"]["answer_source"].startswith("written by the model")


# ------------------------------------------------------------------ cloud mask must not swallow pale ground

def _pale_speckled_scene():
    """~1.8% pale pixels in tiny specks 30 px apart (bare soil / roofs): before the opening step, dilating
    them blanked the whole scene (a real mining scene went from 1.7% bright to a 63% cloud mask)."""
    a = textured(1)
    for y in range(10, 395, 30):
        for x in range(10, 395, 30):
            a[y:y + 4, x:x + 4] = 235
    return a


def test_scattered_pale_specks_do_not_hide_a_real_change():
    a = _pale_speckled_scene()
    b = a.copy()
    b[150:300, 200:350] = 0                               # a genuine change on that scene
    assert run(a, b).major_regions_detected == 1


def test_a_real_cloud_patch_still_masks_change_beneath_it():
    a = textured(1)
    a[100:260, 100:260] = 245                             # one large bright patch = cloud in both dates
    b = a.copy()
    b[120:240, 120:240] = 0                               # 'change' only inside the cloud
    assert run(a, b).major_regions_detected == 0


# ------------------------------------------------------------------ is the pair comparable at all?

def sharp(seed=3, size=400):
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3)).astype(np.uint8)


def stats(edge_pct):
    return {"edge_pct": edge_pct}


def test_comparability_warning_needs_both_a_big_ratio_and_a_real_gap():
    assert comparability_warning(stats(4), stats(19)).startswith("The two images differ strongly in level of detail (4% vs 19%")
    assert "the second is much sharper" in comparability_warning(stats(4), stats(19))
    assert "the first is much sharper" in comparability_warning(stats(20), stats(4))
    assert comparability_warning(stats(30), stats(25)) == ""              # ordinary change
    assert comparability_warning(stats(1), stats(3)) == ""                # ratio 3x but only 2 points apart
    assert comparability_warning(stats(0), stats(6)) != ""                # flat vs busy
    assert comparability_warning(None, stats(9)) == "" and comparability_warning(stats(9), None) == ""


def test_real_looking_zoom_mismatch_is_detected_from_pixels():
    smooth, busy = textured(1), sharp()
    assert comparability_warning(compute_scene_stats(smooth), compute_scene_stats(busy))
    assert comparability_warning(compute_scene_stats(smooth), compute_scene_stats(smooth.copy())) == ""


def test_detect_puts_the_warning_first_and_refuses_to_call_no_change_evidence():
    res = run(textured(1), sharp())
    assert res.warnings and "differ strongly in level of detail" in res.warnings[0]
    assert "WARNING: The two images differ strongly" in res.explanation
    assert "NOT evidence that nothing changed" in res.explanation
    assert res.explanation.index("WARNING:") < res.explanation.index("Changed areas")


def test_a_comparable_pair_gets_no_warning_and_keeps_the_plain_no_change_line():
    a = textured(2)
    res = run(a, a.copy())
    assert res.warnings == [] and "WARNING" not in res.explanation
    assert "No changed area was found" in res.explanation


def test_extra_warnings_from_the_caller_are_shown_too():
    a = textured(2)
    res = run(a, a.copy(), extra_warnings=["Image B could not be registered (only 3 matching features)"])
    assert res.warnings == ["Image B could not be registered (only 3 matching features)"]
    assert "WARNING: Image B could not be registered" in res.explanation


def test_alignment_reports_why_it_only_resized():
    a = textured(1)
    same, valid, note = align_with_status(a, a.copy())
    assert note == "" and valid.all()
    _, _, note = align_with_status(sharp(3), sharp(4))                    # unrelated content
    assert note                                                            # some reason is given
    flat = np.full((200, 200, 3), 120, np.uint8)
    _, _, note = align_with_status(flat, flat.copy())                      # no features at all
    assert note == "no image features found"


def test_the_old_two_value_alignment_api_is_unchanged():
    a = textured(1)
    out = align_standard_images(a, a.copy())
    assert len(out) == 2 and out[0].shape == a.shape and out[1].dtype == bool


def test_endpoint_reports_a_detail_mismatch_and_an_alignment_fallback_in_the_answer_and_trace(client, monkeypatch):
    import main_api
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "1")
    monkeypatch.setenv("LORA_COMPARE", "0")
    monkeypatch.setattr(main_api, "align_with_status",
                        lambda a, b: (b, np.ones(b.shape[:2], bool), "only 3 matching features"))
    files = [("images", ("a.png", png_bytes(textured(1, 96, 96)), "image/png")),
             ("images", ("b.png", png_bytes(sharp(3, 96)), "image/png"))]
    r = post(client, files)
    assert r.status_code == 200, r.text
    body = r.json()
    warnings = body["agent_execution_trace"]["warnings"]
    assert any("could not be registered" in w and "only 3 matching features" in w for w in warnings)
    assert any("differ strongly in level of detail" in w for w in warnings)
    assert "WARNING: Image B could not be registered" in body["answer"]
    assert "NOT evidence that nothing changed" in body["answer"]


def test_endpoint_clean_pair_has_no_comparability_warnings(client, monkeypatch):
    monkeypatch.setenv("CHANGE_EVIDENCE_REPORT", "1")
    monkeypatch.setenv("LORA_COMPARE", "0")
    img = png_bytes(sharp(3, 160))                        # feature-rich, so alignment can verify it
    r = post(client, [("images", ("a.png", img, "image/png")), ("images", ("b.png", img, "image/png"))])
    assert r.status_code == 200, r.text
    assert r.json()["agent_execution_trace"]["warnings"] == []

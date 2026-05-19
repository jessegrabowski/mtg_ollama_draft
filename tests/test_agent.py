from mtg_drafting.agent import DraftAgent


def test_match_exact(cube):
    pack = cube[:5]
    assert DraftAgent._match("Card 002", pack) is pack[2]


def test_match_case_insensitive(cube):
    pack = cube[:5]
    assert DraftAgent._match("  card 003 ", pack) is pack[3]


def test_match_close_name(cube):
    pack = cube[:5]
    # A small typo still resolves to the intended card.
    assert DraftAgent._match("Card 04", pack) is pack[4]


def test_match_rejects_unrelated_name(cube):
    pack = cube[:5]
    assert DraftAgent._match("Lightning Bolt", pack) is None


def test_parse_valid_json():
    raw = '{"pick": "Sol Ring", "reasoning": "Fast mana.", "strategy": "Ramp."}'
    parsed = DraftAgent._parse(raw)
    assert parsed is not None
    assert parsed.pick == "Sol Ring"
    assert parsed.reasoning == "Fast mana."


def test_parse_salvages_truncated_reply():
    # JSON cut off mid-reasoning by the token limit; the leading pick is intact.
    raw = '{"pick": "Lightning Bolt", "reasoning": "Efficient removal that also'
    parsed = DraftAgent._parse(raw)
    assert parsed is not None
    assert parsed.pick == "Lightning Bolt"
    assert parsed.strategy == ""


def test_parse_returns_none_when_pick_absent():
    assert DraftAgent._parse('{"reasoning": "incomplete reply with no pick yet') is None

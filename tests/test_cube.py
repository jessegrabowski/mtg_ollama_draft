from mtg_drafting.cube import _parse_line, load_cube
from mtg_drafting.scryfall import ScryfallIndex


def test_parse_line_extracts_quantity_comment_and_set_tag():
    assert _parse_line("Sol Ring") == (1, "Sol Ring")
    assert _parse_line("1 Sol Ring") == (1, "Sol Ring")
    assert _parse_line("4x Lightning Bolt") == (4, "Lightning Bolt")
    assert _parse_line("Brainstorm (MH2) 123") == (1, "Brainstorm")
    assert _parse_line("3 Brainstorm (MH2) 123") == (3, "Brainstorm")
    assert _parse_line("  Counterspell  # staple") == (1, "Counterspell")
    assert _parse_line("# whole line comment") == (0, "")
    assert _parse_line("") == (0, "")


def test_load_cube_resolves_and_collects_unresolved(tmp_path, cube):
    index = ScryfallIndex(cube)
    listing = "\n".join(
        [
            "# my cube",
            "Card 000",
            "1 Card 001",
            "Card 002 (XYZ) 7",
            "Not A Real Card",
            "",
        ]
    )
    path = tmp_path / "cube.txt"
    path.write_text(listing)

    cards, unresolved = load_cube(path, index)
    assert [c.name for c in cards] == ["Card 000", "Card 001", "Card 002"]
    assert unresolved == ["Not A Real Card"]


def test_load_cube_keeps_duplicate_names(tmp_path, cube):
    index = ScryfallIndex(cube)
    path = tmp_path / "cube.txt"
    path.write_text("Card 010\ncard 010\nCard 011\n")

    cards, unresolved = load_cube(path, index)
    assert [c.name for c in cards] == ["Card 010", "Card 010", "Card 011"]
    assert unresolved == []


def test_load_cube_expands_copy_counts(tmp_path, cube):
    index = ScryfallIndex(cube)
    path = tmp_path / "cube.txt"
    path.write_text("3 Card 005\n2x Card 006\n")

    cards, unresolved = load_cube(path, index)
    assert [c.name for c in cards] == [
        "Card 005",
        "Card 005",
        "Card 005",
        "Card 006",
        "Card 006",
    ]
    assert unresolved == []

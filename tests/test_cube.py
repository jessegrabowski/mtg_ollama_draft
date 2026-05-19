import pytest

from mtg_drafting.cube import _parse_line, load_cube
from mtg_drafting.scryfall import ScryfallIndex


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Sol Ring", (1, "Sol Ring")),
        ("1 Sol Ring", (1, "Sol Ring")),
        ("4x Lightning Bolt", (4, "Lightning Bolt")),
        ("Brainstorm (MH2) 123", (1, "Brainstorm")),
        ("3 Brainstorm (MH2) 123", (3, "Brainstorm")),
        ("  Counterspell  # staple", (1, "Counterspell")),
        ("# whole line comment", (0, "")),
        ("", (0, "")),
    ],
)
def test_parse_line(line, expected):
    assert _parse_line(line) == expected


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

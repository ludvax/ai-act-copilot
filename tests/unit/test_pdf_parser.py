from ai_act_copilot.ingestion.parsers.pdf import _lines, split_sections


def test_table_of_contents_lines_are_dropped() -> None:
    # TOC entries otherwise become sections AND steal the real heading numbers.
    page = "\n".join(
        [
            "3.1. Rationale and objectives ......................................... 12",
            "3. Rationale and objectives",
            "The prohibition aims to protect vulnerable persons.",
        ]
    )

    sections = split_sections(_lines([page]), fallback_heading="Guidance")

    assert [section.number for section in sections] == ["3"]
    assert sections[0].heading == "Rationale and objectives"


def test_footnote_markers_are_not_headings() -> None:
    lines = ["1. Purpose", "207 https://example.invalid/reference", "Body text follows."]

    sections = split_sections(lines, fallback_heading="Guidance")

    assert [section.number for section in sections] == ["1"]


def test_multi_level_heading_without_trailing_dot_is_accepted() -> None:
    sections = split_sections(["3.2 Conditions", "Body."], fallback_heading="Guidance")

    assert [(section.number, section.heading) for section in sections] == [("3.2", "Conditions")]


def test_page_numbers_are_dropped() -> None:
    assert _lines(["12\nReal content here.\n"]) == ["Real content here."]


def test_groups_lines_under_their_heading() -> None:
    lines = [
        "1. Purpose",
        "This document explains the rules.",
        "2. Scope",
        "It applies to providers.",
    ]

    sections = split_sections(lines, fallback_heading="Guidance")

    assert [(section.number, section.heading) for section in sections] == [
        ("1", "Purpose"),
        ("2", "Scope"),
    ]
    assert sections[0].lines == ["This document explains the rules."]


def test_repeated_heading_numbers_get_unique_ids() -> None:
    # Annexes restart numbering, which previously collided in the provisions table.
    lines = ["1. Purpose", "First.", "2. Scope", "Second.", "1. Annex heading", "Third."]

    sections = split_sections(lines, fallback_heading="Guidance")

    numbers = [section.number for section in sections]
    assert numbers == ["1", "2", "1-2"]
    assert len(numbers) == len(set(numbers))


def test_numbered_sentences_are_not_treated_as_headings() -> None:
    lines = [
        "1. Purpose",
        "3. the provider shall ensure that the system is documented and kept up to date;",
    ]

    sections = split_sections(lines, fallback_heading="Guidance")

    assert len(sections) == 1
    assert sections[0].lines[0].startswith("3. the provider")


def test_text_before_any_heading_falls_back_to_one_section() -> None:
    sections = split_sections(["Introductory text without a heading."], fallback_heading="Guide")

    assert [(section.number, section.heading) for section in sections] == [("1", "Guide")]


def test_empty_document_yields_no_sections() -> None:
    assert split_sections([], fallback_heading="Guide") == []

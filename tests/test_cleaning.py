from rag.cleaning import clean_text


def test_clean_text_normalizes_whitespace() -> None:
    messy_text = (
        "  PostgreSQL    supports indexes.  \r\n"
        "\r\n\r\n\r\n"
        "Transactions   are reliable.  "
    )

    result = clean_text(messy_text)

    assert result == "PostgreSQL supports indexes.\n\nTransactions are reliable."


def test_clean_text_normalizes_line_endings() -> None:
    messy_text = "Line one\r\nLine two\rLine three\nLine four"

    result = clean_text(messy_text)

    assert result == "Line one\nLine two\nLine three\nLine four"


def test_clean_text_normalizes_unicode() -> None:
    text = "Cafe\u0301 is open."

    result = clean_text(text)

    assert result == "Caf\u00e9 is open."


def test_clean_text_does_not_change_normal_text() -> None:
    text = "PostgreSQL supports SQL."

    result = clean_text(text)

    assert result == text


def test_clean_text_removes_utf8_bom() -> None:
    text = "\ufeffRedis is an in-memory data store."

    result = clean_text(text)

    assert result == "Redis is an in-memory data store."

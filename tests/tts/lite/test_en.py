from spokestack.tts.lite import en


def test_ascii_conversion():
    assert en.clean("ko\u017eu\u0161\u010dek") == "kozuscek"


def test_lowercase():
    assert en.clean("HELLO") == "hello"


def test_number_expansion():
    # cardinals
    assert en.clean("1") == "one"

    # ordinals
    assert en.clean("2nd") == "second"

    # commas
    assert en.clean("3,000") == "three thousand"

    # years
    assert en.clean("2000") == "two thousand"
    assert en.clean("2005") == "two thousand five"
    assert en.clean("1900") == "nineteen hundred"
    assert en.clean("1906") == "nineteen oh six"
    assert en.clean("2017") == "twenty seventeen"


def test_currency_expansion():
    assert en.clean("$0") == "zero dollars"
    assert en.clean("$4") == "four dollars"
    assert en.clean("$5.67") == "five dollars, sixty-seven cents"
    assert en.clean("$.08") == "eight cents"


def test_decimal_expansion():
    assert en.clean("9.0") == "nine point zero"


def test_abbrev_expansion():
    assert en.clean("Mr.") == "mister"


def test_whitespace_collapse():
    assert en.clean("two  spaces") == "two spaces"

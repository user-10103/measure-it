"""Unit conversions for the (imperial) roof report."""

M2_TO_SQFT = 10.7639104167
M_TO_FT = 3.280839895


def m2_to_sqft(x: float) -> float:
    """Square metres -> square feet."""
    return x * M2_TO_SQFT


def m_to_ft(x: float) -> float:
    """Metres -> feet."""
    return x * M_TO_FT


def m_to_ftin(x: float) -> str:
    """Metres -> 'Xft Yin' (feet + whole inches), e.g. ' 508ft 8in'."""
    total_ft = x * M_TO_FT
    ft = int(total_ft)
    inches = int(round((total_ft - ft) * 12))
    if inches >= 12:                # carry
        ft += 1
        inches -= 12
    return f"{ft}ft {inches}in"


def ft_to_ftin(total_ft: float) -> str:
    """Feet (decimal) -> 'Xft Yin'."""
    return m_to_ftin(total_ft / M_TO_FT)


def sqft_to_squares(x: float) -> float:
    """Square feet -> roofing squares (100 sqft), rounded to 1 dp."""
    return round(x / 100.0, 1)

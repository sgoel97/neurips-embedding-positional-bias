import pandas as pd


def highlight_max(s: pd.Series):
    """
    Highlights the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ["font-weight: bold" if v else "" for v in is_max]


def style_df(
    df: pd.DataFrame,
    title: str | None = None,
    sortby: str | None = None,
    ascending: bool = False,
    precision: int = 2,
    apply_highlight_max: bool = True,
) -> pd.DataFrame:
    if sortby:
        df = df.sort_values(by=sortby, ascending=ascending)

    styled_df = df.style.format(precision=precision)

    if apply_highlight_max:
        styled_df = styled_df.apply(highlight_max)

    if title:
        styled_df = styled_df.set_caption(title)
        styled_df = styled_df.set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [("caption-side", "top"), ("padding", "10px")],
                }
            ]
        )

    return styled_df

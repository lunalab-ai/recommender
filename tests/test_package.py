from __future__ import annotations

import pandas as pd
import pytest

from luna_recsys.baselines import popularity_ranking


def test_popularity_ranking_orders_by_count_then_mean() -> None:
    ratings = pd.DataFrame(
        {
            "item_id": ["B", "A", "A", "B", "C"],
            "rating": [5.0, 4.0, 5.0, 3.0, 5.0],
        }
    )
    result = popularity_ranking(ratings)
    assert result["item_id"].tolist() == ["A", "B", "C"]
    assert result.loc[0, "rating_count"] == 2
    assert result.loc[0, "mean_rating"] == pytest.approx(4.5)


def test_popularity_ranking_validates_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        popularity_ranking([{"item_id": "A"}])

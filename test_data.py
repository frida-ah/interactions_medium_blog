import pandas as pd
import datetime
from main import select_features


def test_interactions():
    pdf = pd.DataFrame(
        {
            "date": [
                datetime(2018, 1, 1),
                datetime(2018, 1, 2),
                datetime(2018, 1, 3),
                datetime(2018, 1, 4),
                datetime(2018, 1, 5),
                datetime(2018, 1, 6),
                datetime(2018, 1, 7),
                datetime(2018, 1, 8),
            ],
            "searches": [58, 100, 58, 100, 58, 100, 58, 100],
            "total_rainfall_num": [3, 4, 3, 4, 3, 4, 3, 4],
            "avg_sunshine_num": [2, 5, 2, 5, 2, 5, 2, 5],
            "avg_temperature_num": [23, 15, 23, 15, 23, 15, 23, 15],
            "holiday_cat": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    a = select_features(pdf)
    print("a:", a.columns)

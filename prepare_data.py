from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq
from workalendar.europe import NetherlandsWithSchoolHolidays as NL
import pandas as pd
import numpy as np
import datetime


def create_date_filter(duration_filter, end_date=None, date_format="%Y-%m-%d"):
    # Reformat Date
    print("Duration filter Google Trends", duration_filter)

    if end_date is not None:
        end_date = datetime.datetime.strptime(end_date, date_format)
        end_date = end_date.date()

    # If no end date is defined for the filter use today as end date
    if end_date is None:
        end_date = datetime.date.today()

    print("End Date time filter Google Trends", end_date)
    print("Duration filter Google Trends", duration_filter)

    # Deduct start date of the date filter from the end date & the defined duration
    start_date = end_date + relativedelta(months=-duration_filter)
    time_filter = str(start_date) + " " + str(end_date)
    print("Time filter for Google Trends: ", time_filter)
    return time_filter


def download_google_trends(
    gt_date_filter,
    list_keywords=None,
    long_format=True,
    country_code="NL",
    language="en-US",
    search_category=71,
    gttz=360,
    gt_timeout=(10, 25),
    gt_retries=2,
    gt_backoff_factor=0.1,
):
    # Send error message if the list with keywords is not supplied
    if list_keywords is None:
        print("Supplied list of keywords is empty")
        print("Please make sure to supply a list of keywords")

    # Download the search volumes for the supplied keywords
    if list_keywords is not None:
        # Configure quasi-API
        pytrends = TrendReq(
            hl=language,
            tz=gttz,
            timeout=gt_timeout,
            retries=gt_retries,
            backoff_factor=gt_backoff_factor,
        )

        # Set the right configuration for request query Google trends. Where needed make sure the request is global
        if country_code != "WORLD":
            print(list_keywords, search_category, gt_date_filter)
            pytrends.build_payload(
                kw_list=list_keywords,
                cat=search_category,
                timeframe=gt_date_filter,
                geo=country_code,
            )
        if country_code == "WORLD":
            pytrends.build_payload(kw_list=list_keywords, cat=search_category, timeframe=gt_date_filter)

        # Download time-series
        pdf_temp_google_trends_results = None
        pdf_temp_google_trends_results = pytrends.interest_over_time()

        if pdf_temp_google_trends_results is not None and len(pdf_temp_google_trends_results.index) > 0:
            # Reset the index of dataframe
            pdf_temp_google_trends_results.reset_index(inplace=True)
            pdf_temp_google_trends_results = pdf_temp_google_trends_results.reset_index(drop=True)
            # Recast date field into string
            pdf_temp_google_trends_results["date"] = pdf_temp_google_trends_results["date"].astype(str).str[:10]
            # Reshape the dataframe into long format (if configured)
            if long_format is True:
                pdf_temp_google_trends_results_long = pd.melt(
                    pdf_temp_google_trends_results,
                    id_vars=["date", "isPartial"],
                    value_vars=list_keywords,
                    var_name=None,
                    value_name="value",
                    col_level=None,
                )
                pdf_temp_google_trends_results_final = pdf_temp_google_trends_results_long.rename(columns={"variable": "keyword", "value": "interest"})

            # Keep the dataframe into wide format (if configured).
            if long_format is False:
                pdf_temp_google_trends_results_final = pdf_temp_google_trends_results
        # Create the right Index for the dataframe
        pdf_temp_google_trends_results_final = pdf_temp_google_trends_results_final.set_index("date")
    return pdf_temp_google_trends_results_final


def create_test_data(keyword):
    duration_filter = 36
    date_filter = create_date_filter(duration_filter, "2020-12-31")
    google_trends_pdf = download_google_trends(date_filter, list_keywords=[keyword])
    google_trends_pdf = google_trends_pdf.sort_index()
    timeseries_test = google_trends_pdf[["interest"]]
    timeseries_test = timeseries_test.rename(columns={"interest": "searches"})
    timeseries_test = timeseries_test.reset_index(drop=False)
    timeseries_test["date"] = timeseries_test["date"].astype("datetime64[ns]")
    return timeseries_test


def get_school_holidays():
    years = [2018, 2019, 2020]
    regions = ["north", "middle", "south"]

    holiday_dict = {}

    for year in years:
        for region in regions:
            calendar = NL(region=region)

            # Get a list of holidays
            holiday_list = calendar.holidays(year)

            # Make a dictionary with a list of holidays for each date entry
            holiday_region_dict = {}
            for h in holiday_list:
                holiday_region_dict.setdefault(h[0], []).append(h[1])

            holiday_dict.update(holiday_region_dict)

    pdf_holidays = pd.DataFrame.from_dict(holiday_dict, orient="index", columns=["Holiday1", "Holiday2"])
    pdf_holidays = pdf_holidays.drop_duplicates()
    pdf_holidays.index.names = ["date"]
    pdf_holidays = pdf_holidays.reset_index(drop=False)

    pdf_holidays = pdf_holidays.sort_values(by=["date"], ascending=True)
    pdf_holidays = pdf_holidays.set_index("date")
    pdf_holidays.index = pd.DatetimeIndex(pdf_holidays.index)
    pdf_holidays = pdf_holidays.resample("W").agg({"Holiday1": "count"})
    pdf_holidays = pdf_holidays.rename(columns={"Holiday1": "holiday_cat"})
    pdf_holidays = pdf_holidays.reset_index(drop=False)
    return pdf_holidays


def set_columns_type(df, list_columns, data_type):
    df[list_columns] = df[list_columns].astype(data_type)
    return df


def get_weather_data():
    date_col = "date"
    weather_data_file = "data/meteo-data/parquet_new"
    pdf_weather = pd.read_parquet(weather_data_file)
    pdf_weather = pdf_weather.loc[pdf_weather.loc[:, "station_identifier"].str.contains("NLE", regex=True)]
    pdf_weather = pdf_weather.pivot_table("observation_value", ["date"], "observation_type")
    pdf_weather.reset_index(drop=False, inplace=True)
    pdf_weather = pdf_weather.drop(["TMAX", "TMIN"], axis=1)
    pdf_weather.reset_index(inplace=True)
    pdf_weather = set_columns_type(pdf_weather, ["date"], "datetime64[ns]")
    pdf_weather = pdf_weather.set_index(date_col)
    pdf_weather = pdf_weather.resample("W").agg({"PRCP": np.sum, "SNWD": np.mean, "TAVG": np.mean})
    pdf_weather.reset_index(inplace=True)
    pdf_weather = pdf_weather.rename(
        columns={
            "PRCP": "total_rainfall_num",
            "SNWD": "avg_sunshine_num",
            "TAVG": "avg_temperature_num",
        }
    )
    return pdf_weather


def prepare_input_data():
    # time search: 2018 - 2020, aggregation on week level
    pdf_searches = create_test_data("ijs")
    # time holidays: 2018 - 2020, aggregation on week level
    pdf_holidays = get_school_holidays()
    # time weather: 2018-2020, aggregation on week level
    pdf_weather = get_weather_data()

    pdf = pd.merge(
        pd.merge(pdf_searches, pdf_holidays, how="left", on=["date"]),
        pdf_weather,
        on="date",
    )
    pdf = pdf.fillna(0)
    return pdf

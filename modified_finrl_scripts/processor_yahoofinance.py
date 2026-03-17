"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
import time
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stockstats import StockDataFrame as Sdf
from webdriver_manager.chrome import ChromeDriverManager

### Added by aymeric75 for scrap_data function


class YahooFinanceProcessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self):
        pass

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    ######## ADDED BY aymeric75 ###################

    def date_to_unix(self, date_str) -> int:
        """Convert a date string in yyyy-mm-dd format to Unix timestamp."""
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        # Base URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver Setup
        options = Options()
        options.add_argument("--headless")  # Headless for performance
        options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # Navigate to the URL
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Wait for redirection and page load

        # Handle potential popup
        try:
            RejectAll = driver.find_element(
                By.XPATH, '//button[@class="btn secondary reject-all"]'
            )
            action = ActionChains(driver)
            action.click(on_element=RejectAll)
            action.perform()
            time.sleep(5)

        except Exception as e:
            print("Popup not found or handled:", e)

        # Parse the page for the table
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            raise Exception("No table found after handling redirection and popup.")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        headers[4] = "Close"
        headers[5] = "Adj Close"
        headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]
        # , 'tic', 'day'

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) == len(headers):  # Only add rows with correct column count
                rows.append(cells)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert columns to appropriate data types
        def safe_convert(value, dtype):
            try:
                return dtype(value.replace(",", ""))
            except ValueError:
                return value

        df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
        df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
        df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
        df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
        df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
        df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

        # Add 'tic' column
        df["tic"] = stock_name

        # Add 'day' column
        start_date = datetime.datetime.fromtimestamp(period1)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - start_date).dt.days
        df = df[df["day"] >= 0]  # Exclude rows with days before the start date

        # Reverse the DataFrame rows
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """Fetch and combine stock data for multiple stock names."""
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"Processing {stock_name} ({i + 1}/{total_stocks})... {(i + 1) / total_stocks * 100:.2f}% complete."
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error fetching data for {stock_name}: {e}")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(by=["day", "tick"]).reset_index(drop=True)

        return combined_df

    ######## END ADDED BY aymeric75 ###################

    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        yahoo_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
        if time_interval in yahoo_intervals:
            return time_interval
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        proxy: str | dict = None,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval
        
        # Set proxy via new config API if provided (deprecated parameter support)
        if proxy is not None:
            import yfinance as yf_config
            yf_config.set_config(proxy=proxy)

        # Download and save the data in a pandas DataFrame
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        data_df = pd.DataFrame()
        
        # Check if this is intraday data (needs day-by-day download due to yfinance limits)
        is_intraday = time_interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']
        
        for tic in ticker_list:
            if is_intraday:
                # For intraday data: download day-by-day (yfinance limits 1m data to 7 days max)
                delta = timedelta(days=1)
                current_tic_start_date = start_date_ts
                while current_tic_start_date <= end_date_ts:
                    temp_df = yf.download(
                        tic,
                        start=current_tic_start_date,
                        end=current_tic_start_date + delta,
                        interval=self.time_interval,
                        progress=False,  # Suppress progress bar for day-by-day
                    )
                    if temp_df.columns.nlevels != 1:
                        temp_df.columns = temp_df.columns.droplevel(1)
                    temp_df["tic"] = tic
                    data_df = pd.concat([data_df, temp_df])
                    current_tic_start_date += delta
            else:
                # For daily/weekly/monthly data: download entire range at once (fast!)
                temp_df = yf.download(
                    tic,
                    start=start_date_ts,
                    end=end_date_ts + timedelta(days=1),  # +1 to include end_date
                    interval=self.time_interval,
                    progress=False,
                )
                if temp_df.columns.nlevels != 1:
                    temp_df.columns = temp_df.columns.droplevel(1)
                temp_df["tic"] = tic
                data_df = pd.concat([data_df, temp_df])

        data_df = data_df.reset_index()
        # Drop "Adj Close" if present (not present when auto_adjust=True, which is new default)
        if "Adj Close" in data_df.columns:
            data_df = data_df.drop(columns=["Adj Close"])
        
        # convert the column names to match processor_alpaca.py as far as poss
        # Expected columns after reset_index: Date/Datetime, Open, High, Low, Close, Volume, tic
        data_df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        # Normalize timestamps to date strings for merging
        df['_date_key'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
        
        # Get all trading days in the date range (don't filter aggressively)
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        
        # Determine if daily or intraday based on time_interval
        daily_patterns = ['1d', '1D', '1Day', 'day', '1wk', '1W', '1Week', '1mo', '1M', '1Month']
        is_daily = self.time_interval in daily_patterns
        
        # produce full timestamp index
        if is_daily:
            # Convert to tz-aware Timestamps to match downloaded data format
            times = [pd.Timestamp(day).tz_localize(NY) for day in trading_days]
        elif self.time_interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(390):  # 390 minutes in trading day
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                f"Data clean at given time interval '{self.time_interval}' is not supported for YahooFinance data."
            )

        # Create a date key for the times index
        times_dates = [pd.Timestamp(t).strftime('%Y-%m-%d') if isinstance(t, str) else t.strftime('%Y-%m-%d') for t in times]
        
        # create a new dataframe with full timestamp series using vectorized merge
        new_df = pd.DataFrame()
        for tic in tic_list:
            # Create template DataFrame with all target timestamps
            tmp_df = pd.DataFrame({
                'timestamp': times,
                '_date_key': times_dates
            })
            
            # Get this ticker's data
            tic_df = df[df.tic == tic][['_date_key', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Merge on date key
            tmp_df = tmp_df.merge(tic_df, on='_date_key', how='left')
            tmp_df = tmp_df.drop(columns=['_date_key'])
            
            # Handle NaN values - fill first row with first valid data
            if pd.isna(tmp_df.iloc[0]['close']):
                first_valid_idx = tmp_df['close'].first_valid_index()
                if first_valid_idx is not None:
                    first_valid_close = tmp_df.loc[first_valid_idx, 'close']
                    tmp_df.iloc[0, tmp_df.columns.get_indexer(['open', 'high', 'low', 'close', 'volume'])] = [
                        first_valid_close, first_valid_close, first_valid_close, first_valid_close, 0.0
                    ]
                else:
                    print(f"Warning: Missing data for ticker {tic}. All prices are NaN. Filling with 0.")
                    tmp_df.iloc[0, tmp_df.columns.get_indexer(['open', 'high', 'low', 'close', 'volume'])] = [0.0] * 5

            # Forward-fill remaining NaN values
            tmp_df[['open', 'high', 'low', 'close']] = tmp_df[['open', 'high', 'low', 'close']].ffill()
            tmp_df['volume'] = tmp_df['volume'].fillna(0.0)
            
            tmp_df = tmp_df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            tmp_df['tic'] = tic
            new_df = pd.concat([new_df, tmp_df])
        
        # Clean up the temporary column from source df
        df = df.drop(columns=['_date_key'])

        # reset index and rename columns
        new_df = new_df.reset_index(drop=True)
        
        # Ensure timestamp is datetime
        # The index comes from 'times' which may be strings (trading days) or Timestamps
        if len(new_df) > 0:
            # Convert to datetime, handling both string and Timestamp inputs
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], utc=True).dt.tz_convert('America/New_York')
            
            # Add 'date' column for compatibility with data_split() and other FinRL functions
            new_df['date'] = new_df['timestamp'].dt.strftime('%Y-%m-%d')
        else:
            new_df['date'] = []

        #        print("Data clean all finished!")

        return new_df

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        vix = cleaned_vix[["timestamp", "close"]]
        vix = vix.rename(columns={"close": "VIXY"})

        df = data.copy()
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb

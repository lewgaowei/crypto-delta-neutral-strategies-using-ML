import requests
import json
import time
from datetime import datetime, timedelta
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import asyncio
from binance.client import Client
import traceback
from pytz import timezone  # Ensure this import is present

class API_BINANCE_MARKET_DATA():
    
    def __init__(self):
        self.API_KEY = "UuSexlTldLhkGCYY3QxJDlstoItOpUdtbstsHxXGVwHkqGhwo6v9drrzxII5qIal"
        self.SECRET_KEY = "uvkvrLeYcjr947jGrpBkIMjmijX4aZVVXDIMcD8VSNOxbbrIpxIkZtdaCJ93Q44x"

        # Smartproxy configuration
        username = 'sp4itr1kd8'
        password = '~6ejeidL2Ej0mOgo7O'
        proxy = f"http://{username}:{password}@dc.smartproxy.com:10001"
        
        # Create a session with the Smartproxy settings
        proxies = {
            'http': proxy,
            'https': proxy
        }
        
        # Initialize the Spot client
        # self.Client = Client(api_key=self.API_KEY, api_secret=self.SECRET_KEY, requests_params={'proxies': proxies})
        self.Client = Client(api_key=self.API_KEY, api_secret=self.SECRET_KEY)

        # self.current_directory = os.getcwd()
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.download_path_historical_price = self.current_directory + "/raw_historical_price"
        if not os.path.exists(self.download_path_historical_price):
            os.makedirs(self.download_path_historical_price)

        # Log if the client is signed in
        try:
            account_info = self.Client.get_account()
            print("Binance Spot client signed in successfully.")
        except Exception as e:
            print(f"Binance Spot client sign-in failed: {e}")
            
            
    def get_server_time(self):
        """Fetch the server time from Binance"""
        try:
            server_time = self.Client.get_server_time()
            print(server_time)
            return datetime.fromtimestamp(server_time['serverTime'] / 1000, timezone('UTC'))
        except Exception as e:
            print(f"Error fetching server time: {e}")
            return datetime.now(timezone('UTC'))  # Fallback to local time if there's an error

    def clear_historical_data(self):
        """Remove all files in the historical price data directory"""
        try:
            # Get list of all files in directory
            files = os.listdir(self.download_path_historical_price)
            
            # Remove each file
            for file in files:
                file_path = os.path.join(self.download_path_historical_price, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        # print('remove')
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
                    
            print(f"Successfully cleared all files from {self.download_path_historical_price}")
            
        except Exception as e:
            print(f"Error clearing historical data: {str(e)}")
            traceback.print_exc()
            

    # def get_historical_prices(self, symbol):
        
    #     # Determine the starting point from the existing data
    #     latest_time = self.find_latest_time_interval(symbol)
    #     # If there is no existing data, use a default start date
    #     if latest_time is None:
    #         latest_time = datetime(2024, 11, 25, 0, 0)
        
    #     # Generate the time intervals from the latest timestamp      
    #     time_interval_list = self.get_time_intervals(latest_time)
        
    #     df_list = []
       
    #     for time_interval in time_interval_list:
    #         index = time_interval[0]
    #         time_from_str = time_interval[1]
    #         time_to_str = time_interval[2]
            
            
    #         time_from_int = int(pd.to_datetime(time_from_str).timestamp() * 1000)
    #         time_to_int = int(pd.to_datetime(time_to_str).timestamp() * 1000)


    #         print(time_from_str, time_from_int)
    #         # Fetch historical data
    #         # klines = self.Client.klines(symbol, '1m', time_from_str, time_to_str)
    #         klines = self.Client.klines(symbol, "1m", startTime=time_from_int, endTime=time_to_int)
    #         # print(klines)
    #         # sleep(1000)
            
    #         # Convert the data to a pandas DataFrame
    #         df = pd.DataFrame(klines, columns=[
    #             'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    #             'close_time', 'quote_asset_volume', 'number_of_trades', 
    #             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    #         ])
            
    #         df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    #         df['index_interval'] = index
    #         df_list.append(df)
        
    #     if df_list:
    #         print(f"PROCESSING | SAVING EXCEL FILE | {symbol}")
    #         final_df = pd.concat(df_list, ignore_index=True)
            
    #         # Check if the file exists before reading
    #         excel_file_path = f'{self.download_path_historical_price}/{symbol}_binance_historical_data.xlsx'
    #         temp_excel_file_path = f'{self.download_path_historical_price}/{symbol}_binance_historical_data_temp.xlsx'
            
    #         if os.path.exists(excel_file_path):
    #             existing_df = pd.read_excel(excel_file_path)
    #         else:
    #             # Create an empty DataFrame if the file does not exist
    #             existing_df = pd.DataFrame(columns=final_df.columns)
            
    #         updated_df = pd.concat([existing_df, final_df], ignore_index=True)
            
    #         # Save to a temporary file first
    #         updated_df.to_excel(temp_excel_file_path, index=False)
            
    #         # Verify the temporary file is not corrupted
    #         try:
    #             pd.read_excel(temp_excel_file_path)
    #             # If successful, replace the original file
    #             os.replace(temp_excel_file_path, excel_file_path)
    #         except Exception as e:
    #             print(f"Error verifying the temporary Excel file: {e}")
    #             # Optionally, handle the error (e.g., log it, notify the user, etc.)
        
    #     print(f"SUCCESS | GET HISTORICAL PRICE | {symbol}")
    
    def find_latest_time_interval(self, symbol):
        excel_file_path = f'{self.download_path_historical_price}/{symbol}_binance_historical_data.xlsx'
        if os.path.exists(excel_file_path):
            df_existing = pd.read_excel(excel_file_path)
            latest_time = df_existing['datetime'].max()
            return latest_time
        else:
            return None
    
    def get_time_intervals(self, start_time, timeframe="1m"):
        # Calculate interval based on timeframe to fetch ~500-1000 candles per request
        # This optimizes API calls while staying within Binance's 1500 candle limit
        timeframe_intervals = {
            "1m": 500,           # 500 minutes = ~8 hours
            "5m": 2500,          # 2500 minutes = ~2 days
            "15m": 7500,         # 7500 minutes = ~5 days
            "1h": 30000,         # 30000 minutes = ~21 days
            "4h": 120000,        # 120000 minutes = ~83 days
            "1d": 720000,        # 720000 minutes = ~500 days
            "1M": 21600000       # 21600000 minutes = ~500 months (~41 years)
        }

        interval_minutes = timeframe_intervals.get(timeframe, 500)  # Default to 500 if unknown
        now = datetime.now(timezone('UTC'))  # Use UTC time
        # Convert to UTC

        intervals = []
        count = 1
        # Localize the start_time to UTC
        time_from = start_time.replace(tzinfo=timezone('UTC')) + timedelta(minutes=1)
        
        # print(f"Start time (SGT): {time_from}")  # Debug: Check start time
        # print(time_from, now)
        
        while time_from < now:
            time_to = time_from + timedelta(minutes=interval_minutes)
            if time_to > now:
                time_to = now

            time_from_str = time_from.strftime('%Y-%m-%d %H:%M:%S')
            time_to_str = time_to.strftime('%Y-%m-%d %H:%M:%S')
            # print(f"Interval {count}: {time_from_str} to {time_to_str}")  # Debug: Check each interval
            intervals.append((count, time_from_str, time_to_str))
            time_from = time_to
            count += 1
        
        return intervals



    def get_historical_prices_json(self, symbol, timeframe="1m", start_date=None):
        # Determine the starting point from the existing data
        print(timeframe)
        latest_time = self.find_latest_time_interval_json(symbol, timeframe)
        if latest_time is None:
            # Use provided start_date or default to BTC spot trading launch (August 2017)
            if start_date is None:
                start_date = datetime(2017, 8, 1, 0, 0)
            latest_time = start_date
        
        # Generate the time intervals from the latest timestamp
        time_interval_list = self.get_time_intervals(latest_time, timeframe)
        df_list = []
        
        for time_interval in time_interval_list:
            index = time_interval[0]
            time_from_str = time_interval[1]
            time_to_str = time_interval[2]
            
            time_from_int = int(pd.to_datetime(time_from_str).timestamp() * 1000)
            time_to_int = int(pd.to_datetime(time_to_str).timestamp() * 1000)
            
            print(time_from_str, time_to_str)
            # Retry logic
            while True:
                try:
                    # Fetch historical data using the specified timeframe (Spot API uses get_klines)
                    klines = self.Client.get_klines(symbol=symbol, interval=timeframe, startTime=time_from_int, endTime=time_to_int)
                    break  # Exit loop if successful
                except Exception as e:
                    if "Too many requests" in str(e):
                        print(f"Rate limit exceeded for {symbol} ({timeframe}). Waiting before retrying...")
                        time.sleep(60)  # Wait for 60 seconds before retrying
                    else:
                        raise  # Re-raise the exception if it's not a rate limit error
            
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['index_interval'] = index
            df_list.append(df)
        
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            
            # Include timeframe in the filename
            json_file_path = f'{self.download_path_historical_price}/{symbol}_{timeframe}_binance_spot_historical_data.json'
            temp_json_file_path = f'{self.download_path_historical_price}/{symbol}_{timeframe}_binance_spot_historical_data_temp.json'
            
            if os.path.exists(json_file_path):
                try:
                    existing_df = pd.read_json(json_file_path, orient='records', lines=True)
                except ValueError as e:
                    print(f"Error reading JSON file: {e}")
                    existing_df = pd.DataFrame(columns=final_df.columns)
            else:
                existing_df = pd.DataFrame(columns=final_df.columns)
            
            updated_df = pd.concat([existing_df, final_df], ignore_index=True)
            
            # Save to a temporary file first
            updated_df.to_json(temp_json_file_path, orient='records', lines=True)
            
            # Verify the temporary file is not corrupted
            try:
                pd.read_json(temp_json_file_path, orient='records', lines=True)
                os.replace(temp_json_file_path, json_file_path)
            except Exception as e:
                print(f"Error verifying the temporary JSON file for {symbol} ({timeframe}): {e}")
        
        print(f"SUCCESS | GET HISTORICAL PRICE | {symbol} ({timeframe})")

    def find_latest_time_interval_json(self, symbol, timeframe="1m"):
        json_file_path = f'{self.download_path_historical_price}/{symbol}_{timeframe}_binance_spot_historical_data.json'
        if os.path.exists(json_file_path):
            try:
                df_existing = pd.read_json(json_file_path, orient='records', lines=True)

                # Check if dataframe is empty or doesn't have datetime column
                if df_existing.empty or 'datetime' not in df_existing.columns:
                    return None

                # Convert datetime column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df_existing['datetime']):
                    df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])

                latest_time = df_existing['datetime'].max()
                return latest_time
            except Exception as e:
                print(f"Error reading existing data for {symbol} ({timeframe}): {e}")
                return None
        else:
            return None

    def get_multi_timeframe_data(self, symbol, timeframe="1M", start_date=None):
        """Fetch historical data for a symbol with specified timeframe and start date"""
        try:
            print(f"Fetching {timeframe} data for {symbol}...")
            self.get_historical_prices_json(symbol, timeframe, start_date)

            return True
        except Exception as e:
            print(f"Error fetching multi-timeframe data for {symbol}: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":

    # ========== CONFIGURATION - CHANGE THESE VALUES ==========
    START_DATE = datetime(2020, 1, 1, 0, 0)  # Start date for historical data
    TIMEFRAME = "5m"  # Timeframe: "1M" (monthly), "1d" (daily), "1h" (hourly), "5m" (5-min), "1m" (1-min)
    # =========================================================

    # ticker_list = ['1000000MOGUSDT', '1000CHEEMSUSDT', '1000WHYUSDT', 'ADAUSDT', 'AMBUSDT', 'ATOMUSDT', 'BANUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'CETUSUSDT', 'DASHUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHUSDT', 'FIOUSDT', 'FTMUSDT', 'GLMUSDT', 'GRASSUSDT', 'HIFIUSDT', 'HIPPOUSDT', 'IOTAUSDT', 'KEYUSDT', 'LINKUSDT', 'LTCUSDT', 'MANAUSDT', 'OMGUSDT', 'ONTUSDT', 'POWRUSDT', 'REEFUSDT', 'RENUSDT', 'RPLUSDT', 'SANDUSDT', 'SANTOSUSDT', 'SEIUSDT', 'SSVUSDT', 'SWELLUSDT', 'TRXUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'ZECUSDT']
    # ticker_list = ['BTCUSDT']
    # print(len(ticker_list))
    ticker_list = ["BTCUSDT"]
    # ticker_list = ["ETHUSDT"]
    ticker_list = ["TNSRUSDT", "HYPEUSDT", "XRPUSDT", "SOLUSDT", "TRONUSDT", "DOGEUSDT", "ADAUSDT"]
    
    # ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "TAOUSDT", "SUPERUSDT", "LSKUSDT", "RESOLVUSDT", "ORCAUSDT", "TURBOUSDT", "LTCUSDT", "BANANAUSDT", "SKLUSDT", "ACEUSDT", "AWEUSDT", "MLNUSDT", "OMUSDT"]
    # API_BINANCE_MARKET_DATA().clear_historical_data()
    # ticker_list = ["TRXUSDT"]
    # Use ThreadPoolExecutor to parallelize the process
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(API_BINANCE_MARKET_DATA().get_multi_timeframe_data, ticker, TIMEFRAME, START_DATE): ticker for ticker in ticker_list}
        
        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()  # This will raise an exception if the function raised
                print(f"Completed fetching data for {ticker} (all timeframes)")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                # Add more detailed logging
                traceback.print_exc()
    
    



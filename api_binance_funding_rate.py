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
from binance.um_futures import UMFutures
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
        
        # Initialize the client with the session
        # self.Client = UMFutures(key=self.API_KEY, secret=self.SECRET_KEY, proxies=proxies)
        self.Client = UMFutures(key=self.API_KEY, secret=self.SECRET_KEY)
   
        # self.current_directory = os.getcwd()
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.download_path_historical_price = self.current_directory + "/raw_historical_price"
        if not os.path.exists(self.download_path_historical_price):
            os.makedirs(self.download_path_historical_price)

        # Base URL for Binance Futures API
        self.base_url = "https://fapi.binance.com"

        # Log if the client is signed in
        try:
            account_info = self.Client.account()
            print("Binance client signed in successfully.")
        except Exception as e:
            print(f"Binance client sign-in failed: {e}")

        # Create directory for funding rate data
        self.download_path_funding_rate = self.current_directory + "/raw_funding_rate"
        if not os.path.exists(self.download_path_funding_rate):
            os.makedirs(self.download_path_funding_rate)
            
            
    def _make_request(self, endpoint, params=None, method='GET'):
        """
        General method to make REST API requests to Binance

        Args:
            endpoint (str): API endpoint path (e.g., '/fapi/v1/fundingRate')
            params (dict): Query parameters
            method (str): HTTP method (GET, POST, etc.)

        Returns:
            dict or list: API response data
        """
        url = self.base_url + endpoint

        try:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {e}")
            traceback.print_exc()
            return None

    def get_server_time(self):
        """Fetch the server time from Binance"""
        try:
            server_time = self.Client.time()
            print(server_time)
            return datetime.fromtimestamp(server_time['serverTime'] / 1000, timezone('UTC'))
        except Exception as e:
            print(f"Error fetching server time: {e}")
            return datetime.now(timezone('UTC'))  # Fallback to local time if there's an error

    def get_funding_rate_history(self, symbol=None, start_time=None, end_time=None, limit=1000):
        """
        Get funding rate history for a single symbol

        Args:
            symbol (str, optional): Trading pair (e.g., 'BTCUSDT'). If None, returns recent data for all symbols
            start_time (datetime or int, optional): Start time as datetime object or timestamp in ms
            end_time (datetime or int, optional): End time as datetime object or timestamp in ms
            limit (int): Number of records to fetch (1-1000, default 1000)

        Returns:
            list: List of funding rate records, each containing:
                - symbol: Trading pair
                - fundingRate: Funding rate percentage
                - fundingTime: Timestamp in milliseconds
                - markPrice: Mark price at funding time
        """
        endpoint = '/fapi/v1/fundingRate'
        params = {}

        if symbol:
            params['symbol'] = symbol

        if start_time:
            if isinstance(start_time, datetime):
                params['startTime'] = int(start_time.timestamp() * 1000)
            else:
                params['startTime'] = start_time

        if end_time:
            if isinstance(end_time, datetime):
                params['endTime'] = int(end_time.timestamp() * 1000)
            else:
                params['endTime'] = end_time

        if limit and 1 <= limit <= 1000:
            params['limit'] = limit

        print(f"Fetching funding rate history for {symbol or 'all symbols'}...")
        data = self._make_request(endpoint, params)

        if data:
            print(f"Successfully fetched {len(data)} funding rate records")

        return data

    def get_funding_rate_history_batch(self, symbols, start_time=None, end_time=None, limit=1000):
        """
        Get funding rate history for multiple symbols

        Args:
            symbols (list): List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            start_time (datetime or int, optional): Start time
            end_time (datetime or int, optional): End time
            limit (int): Number of records to fetch per symbol (1-1000, default 1000)

        Returns:
            dict: Dictionary with symbols as keys and funding rate data as values
        """
        results = {}

        for symbol in symbols:
            try:
                data = self.get_funding_rate_history(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                results[symbol] = data
                sleep(0.1)  # Small delay to respect rate limits
            except Exception as e:
                print(f"Error fetching funding rate for {symbol}: {e}")
                results[symbol] = None

        return results

    def get_funding_rate_history_range(self, symbol, start_time, end_time, limit=1000):
        """
        Get complete funding rate history for a symbol over a date range
        Handles pagination automatically if the range exceeds the limit

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT')
            start_time (datetime): Start date
            end_time (datetime): End date
            limit (int): Number of records per request (max 1000)

        Returns:
            list: Complete list of all funding rate records in the range
        """
        all_data = []
        current_start = start_time

        if isinstance(start_time, datetime):
            current_start_ms = int(start_time.timestamp() * 1000)
        else:
            current_start_ms = start_time

        if isinstance(end_time, datetime):
            end_time_ms = int(end_time.timestamp() * 1000)
        else:
            end_time_ms = end_time

        print(f"Fetching complete funding rate history for {symbol} from {start_time} to {end_time}")

        while current_start_ms < end_time_ms:
            data = self.get_funding_rate_history(
                symbol=symbol,
                start_time=current_start_ms,
                end_time=end_time_ms,
                limit=limit
            )

            if not data or len(data) == 0:
                break

            all_data.extend(data)

            if len(data) < limit:
                break

            current_start_ms = data[-1]['fundingTime'] + 1
            sleep(0.1)

        print(f"Total records fetched for {symbol}: {len(all_data)}")
        return all_data

    def save_funding_rate_to_csv(self, data, symbol, filename=None, start_time=None, end_time=None):
        """
        Save funding rate data to CSV file

        Args:
            data (list): Funding rate data
            symbol (str): Trading pair symbol
            filename (str, optional): Custom filename. If None, auto-generates based on symbol and date range
            start_time (datetime, optional): Start date of the data range for filename generation
            end_time (datetime, optional): End date of the data range for filename generation
        """
        if not data:
            print(f"No data to save for {symbol}")
            return

        df = pd.DataFrame(data)

        df['fundingDateTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['formattedFundingDateTime'] = df['fundingDateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')

        if 'markPrice' in df.columns:
            df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')

        # Drop the original fundingTime column (raw milliseconds) as it's redundant
        df = df.drop(columns=['fundingTime'])

        if not filename:
            if start_time and end_time:
                # Use date range for filename
                if isinstance(start_time, datetime):
                    start_str = start_time.strftime("%Y%m%d")
                else:
                    start_str = datetime.fromtimestamp(start_time / 1000).strftime("%Y%m%d")

                if isinstance(end_time, datetime):
                    end_str = end_time.strftime("%Y%m%d")
                else:
                    end_str = datetime.fromtimestamp(end_time / 1000).strftime("%Y%m%d")

                filename = f"{symbol}_funding_rate_{start_str}_{end_str}.csv"
            else:
                # Fallback to current timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_funding_rate_{timestamp}.csv"

        filepath = os.path.join(self.download_path_funding_rate, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")

        return filepath

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

    def clear_funding_rate_data(self):
        """Remove all files in the funding rate data directory"""
        try:
            files = os.listdir(self.download_path_funding_rate)

            for file in files:
                file_path = os.path.join(self.download_path_funding_rate, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")

            print(f"Successfully cleared all files from {self.download_path_funding_rate}")

        except Exception as e:
            print(f"Error clearing funding rate data: {str(e)}")
            traceback.print_exc()

    def process_funding_rate_for_symbol(self, symbol, start_time, end_time):
        """
        Process and save funding rate data for a single symbol
        Used for parallel processing with ThreadPoolExecutor

        Args:
            symbol (str): Trading pair
            start_time (datetime): Start date
            end_time (datetime): End date
        """
        try:
            # Generate expected filename based on date range
            if isinstance(start_time, datetime):
                start_str = start_time.strftime("%Y%m%d")
            else:
                start_str = datetime.fromtimestamp(start_time / 1000).strftime("%Y%m%d")

            if isinstance(end_time, datetime):
                end_str = end_time.strftime("%Y%m%d")
            else:
                end_str = datetime.fromtimestamp(end_time / 1000).strftime("%Y%m%d")

            filename = f"{symbol}_funding_rate_{start_str}_{end_str}.csv"
            filepath = os.path.join(self.download_path_funding_rate, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                print(f"[SKIP] File already exists for {symbol}: {filename}")
                return True

            print(f"Processing {symbol}...")
            data = self.get_funding_rate_history_range(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

            if data:
                self.save_funding_rate_to_csv(data, symbol, start_time=start_time, end_time=end_time)
                return True
            else:
                print(f"No data retrieved for {symbol}")
                return False

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            traceback.print_exc()
            return False
            

if __name__ == "__main__":

    # ========== CONFIGURATION - CHANGE THESE VALUES ==========
    # Configuration for funding rate data collection
    START_DATE = datetime(2020, 1, 1, 0, 0)  # Start date for historical data
    END_DATE = datetime(2025, 11, 30, 23, 59)  # End date for historical data

    # List of trading pairs to fetch
    ticker_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "TAOUSDT", "SUPERUSDT", "LSKUSDT", "RESOLVUSDT", "ORCAUSDT", "TURBOUSDT", "LTCUSDT", "BANANAUSDT", "SKLUSDT", "ACEUSDT", "AWEUSDT", "MLNUSDT", "OMUSDT"]
    # =========================================================

    # Initialize API client
    api = API_BINANCE_MARKET_DATA()


    # ========== EXAMPLE 4: Parallel processing for multiple symbols ==========
    print("\n" + "="*60)
    print("EXAMPLE 4: Fetch and save funding rate data for multiple symbols (parallel)")
    print("="*60)

    # Clear previous funding rate data (optional)
    # api.clear_funding_rate_data()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                api.process_funding_rate_for_symbol,
                ticker,
                START_DATE,
                END_DATE
            ): ticker for ticker in ticker_list
        }

        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"[SUCCESS] Completed fetching funding rate data for {ticker}")
                else:
                    print(f"[FAILED] Failed to fetch funding rate data for {ticker}")
            except Exception as e:
                print(f"[ERROR] Error fetching funding rate data for {ticker}: {e}")
                traceback.print_exc()

    print("\n" + "="*60)
    print("All funding rate data collection completed!")
    print("="*60)
    
    



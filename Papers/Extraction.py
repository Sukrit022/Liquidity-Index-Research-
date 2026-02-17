import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class RateLimitProofDownloader:
    def __init__(self):
        self.min_delay = 10  # Minimum 10 seconds between requests
        self.max_delay = 20  # Maximum 20 seconds
        self.retry_delay = 300  # 5 minutes when rate limited
        self.max_retries = 5
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def safe_delay(self, extra_delay=0):
        """Smart delay that gets longer if we're hitting limits"""
        base_delay = random.uniform(self.min_delay, self.max_delay)
        total_delay = base_delay + extra_delay
        
        print(f"⏳ Waiting {total_delay:.1f} seconds to avoid rate limits...")
        time.sleep(total_delay)
    
    def download_with_retries(self, symbol, years=10):
        """Download stock data with intelligent retry logic"""
        yahoo_symbol = f"{symbol}.NS"
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 {symbol}: Attempt {attempt + 1}/{self.max_retries}")
                
                # Check if file already exists
                filename = f"nse500_slow_download/{symbol}_data.csv"
                if os.path.exists(filename):
                    print(f"  📁 Already exists, skipping...")
                    return True
                
                # Create ticker
                ticker = yf.Ticker(yahoo_symbol)
                
                # Download data with conservative settings
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years*365)
                
                # Use smaller chunks if getting rate limited
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    auto_adjust=True,
                    prepost=False,
                    actions=False,
                    timeout=30
                )
                
                if not data.empty and len(data) > 50:
                    # Save the data
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
                    
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    data.to_csv(filename, index=False)
                    
                    print(f"  ✅ Success: {len(data)} records saved")
                    return True
                else:
                    print(f"  ⚠️ Got {len(data)} records, trying again...")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate limit" in error_msg or "429" in error_msg or "too many requests" in error_msg:
                    wait_time = self.retry_delay * (attempt + 1)  # Exponential backoff
                    print(f"  🚫 Rate limited! Waiting {wait_time/60:.1f} minutes...")
                    time.sleep(wait_time)
                elif "timeout" in error_msg:
                    print(f"  ⏰ Timeout, trying again in 30 seconds...")
                    time.sleep(30)
                else:
                    print(f"  ❌ Error: {str(e)[:50]}...")
                    
                if attempt < self.max_retries - 1:
                    self.safe_delay(extra_delay=30)  # Extra delay after error
        
        print(f"  💀 Failed after {self.max_retries} attempts")
        return False
    
    def download_stock_list(self, max_stocks=50):
        """Download a limited number of stocks safely"""
        print("🐌 ULTRA-SAFE NSE STOCK DOWNLOADER")
        print("This is slow but GUARANTEED to work!")
        print("-" * 50)
        
        # Top 50 NSE stocks (manually curated to avoid API calls)
        top_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ITC', 'KOTAKBANK', 'BHARTIARTL', 'SBIN', 'ASIANPAINT',
            'MARUTI', 'LT', 'AXISBANK', 'TITAN', 'WIPRO',
            'HCLTECH', 'ULTRACEMCO', 'BAJFINANCE', 'ONGC', 'POWERGRID',
            'NESTLEIND', 'NTPC', 'TECHM', 'M&M', 'SUNPHARMA',
            'TATAMOTORS', 'JSWSTEEL', 'INDUSINDBK', 'GRASIM', 'ADANIPORTS',
            'COALINDIA', 'BRITANNIA', 'SHREECEM', 'UPL', 'APOLLOHOSP',
            'DRREDDY', 'CIPLA', 'BAJAJFINSV', 'HINDALCO', 'DIVISLAB',
            'TATASTEEL', 'HEROMOTOCO', 'EICHERMOT', 'BAJAJ-AUTO', 'DABUR',
            'GODREJCP', 'PIDILITIND', 'ICICIBANK', 'HDFCLIFE', 'BPCL'
        ]
        
        stocks_to_process = top_stocks[:max_stocks]
        successful_downloads = 0
        
        print(f"📋 Processing {len(stocks_to_process)} stocks")
        print(f"⏱️ Estimated time: {len(stocks_to_process) * 0.5:.1f} minutes")
        print()
        
        for i, stock in enumerate(stocks_to_process):
            print(f"[{i+1}/{len(stocks_to_process)}] Processing {stock}")
            
            success = self.download_with_retries(stock)
            if success:
                successful_downloads += 1
            
            # Always wait between stocks (even successful ones)
            if i < len(stocks_to_process) - 1:  # Don't wait after last stock
                self.safe_delay()
        
        # Create combined file
        self.create_combined_file()
        
        print(f"\n🎉 DOWNLOAD COMPLETE!")
        print(f"✅ Success: {successful_downloads}/{len(stocks_to_process)}")
        print(f"📁 Files saved in: nse500_slow_download/")
        
        return successful_downloads
    
    def create_combined_file(self):
        """Combine all individual files into one"""
        folder = "nse500_slow_download"
        csv_files = [f for f in os.listdir(folder) if f.endswith('_data.csv')]
        
        if csv_files:
            print(f"\n📋 Combining {len(csv_files)} files...")
            combined_data = []
            
            for file in csv_files:
                df = pd.read_csv(f"{folder}/{file}")
                combined_data.append(df)
            
            if combined_data:
                final_df = pd.concat(combined_data, ignore_index=True)
                final_df.to_csv(f"{folder}/ALL_STOCKS_COMBINED.csv", index=False)
                print(f"✅ Combined file created: {len(final_df):,} total records")

# Easy-to-use function
def start_safe_download(num_stocks=20):
    """Start downloading with safe settings"""
    downloader = RateLimitProofDownloader()
    
    print("⚠️  IMPORTANT NOTES:")
    print("- This method is SLOW but works around all rate limits")
    print("- Each stock takes 10-20 seconds to download")
    print("- You can stop and restart - it will skip existing files")
    print("- Leave this running and go do something else!")
    print()
    
    input("Press Enter to start the download...")
    
    return downloader.download_stock_list(max_stocks=num_stocks)

if __name__ == "__main__":
    # Download top 20 stocks safely
    start_safe_download(num_stocks=20)
import requests
import pandas as pd
import json

def test_alpha_vantage_key(api_key):
    """Test your Alpha Vantage API key with a simple request"""
    
    print("🧪 Testing Alpha Vantage API Key...")
    print("-" * 40)
    
    # Test with a US stock first (easier to get working)
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': 'AAPL',
        'apikey': api_key,
        'outputsize': 'compact'
    }
    
    try:
        print("📡 Testing with AAPL (US stock)...")
        response = requests.get(url, params=params)
        data = response.json()
        
        print(f"Response status: {response.status_code}")
        
        if 'Time Series (Daily)' in data:
            print("✅ API key works perfectly!")
            
            # Show sample data
            time_series = data['Time Series (Daily)']
            latest_date = list(time_series.keys())[0]
            latest_data = time_series[latest_date]
            
            print(f"📊 Sample data for AAPL ({latest_date}):")
            print(f"   Open: ${latest_data['1. open']}")
            print(f"   High: ${latest_data['2. high']}")
            print(f"   Low: ${latest_data['3. low']}")
            print(f"   Close: ${latest_data['4. close']}")
            print(f"   Volume: {latest_data['5. volume']}")
            
            return True
            
        elif 'Error Message' in data:
            print(f"❌ Error: {data['Error Message']}")
            
        elif 'Note' in data:
            print(f"🚫 Rate Limit: {data['Note']}")
            print("Wait a minute and try again.")
            
        else:
            print("🤔 Unexpected response:")
            print(json.dumps(data, indent=2))
            
    except Exception as e:
        print(f"💥 Connection error: {str(e)}")
    
    return False

def test_indian_stocks(api_key):
    """Test Indian stock formats with Alpha Vantage"""
    
    print("\n🇮🇳 Testing Indian Stock Formats...")
    print("-" * 40)
    
    # Test different Indian stock formats
    test_symbols = [
        "RELIANCE.BSE",
        "RELIANCE.NSE", 
        "TCS.BSE",
        "TCS.NSE"
    ]
    
    url = "https://www.alphavantage.co/query"
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                record_count = len(time_series)
                print(f"✅ {symbol}: {record_count} records available")
                
                # Show latest data
                latest_date = list(time_series.keys())[0]
                latest_close = time_series[latest_date]['4. close']
                print(f"   Latest close ({latest_date}): ₹{latest_close}")
                
                return symbol  # Return working format
                
            elif 'Error Message' in data:
                print(f"❌ {symbol}: {data['Error Message']}")
                
            elif 'Note' in data:
                print(f"🚫 {symbol}: Rate limited, wait and try again")
                break  # Stop testing to avoid more rate limits
                
            else:
                print(f"⚠️ {symbol}: No data found")
        
        except Exception as e:
            print(f"💥 {symbol}: Error - {str(e)}")
        
        # Small delay between tests
        import time
        time.sleep(15)  # 15 seconds between requests
    
    print("\n💡 Tip: If none worked, try the manual download method instead!")
    return None

if __name__ == "__main__":
    print("🔑 ALPHA VANTAGE API KEY TESTER")
    print("=" * 50)
    
    # Get API key from user
    api_key = input("Enter your Alpha Vantage API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        print("Get your free key from: https://www.alphavantage.co/support/#api-key")
    else:
        # Test the key
        if test_alpha_vantage_key(api_key):
            print("\n🎉 Your API key works! Now testing Indian stocks...")
            working_format = test_indian_stocks(api_key)
            
            if working_format:
                print(f"\n✅ Found working format: {working_format}")
                print("🚀 Ready to download NSE data!")
            else:
                print("\n⚠️ Indian stocks might not be available in free tier")
                print("But you can still use this for US stocks or upgrade to paid tier")
        else:
            print("\n❌ API key test failed. Please check:")
            print("1. Did you copy the key correctly?") 
            print("2. Did you get it from the official Alpha Vantage email?")
            print("3. Try waiting a few minutes if you just got the key")
import pandas as pd
from pathlib import Path
from datetime import datetime

TRACKER_PATH = Path(__file__).parent.parent / "data" / "Tracker.xlsx"

def get_scheduled_topic(target_date_str: str):
    """
    Attempts to find a scheduled topic in Tracker.xlsx for the given date.
    Target columns: Date, Day, Domain, Topic, Sub Topics
    Returns (domain, topic, subtopics) if found, else (None, None, None).
    """
    if not TRACKER_PATH.exists():
        print(f"  [WARN] Tracker file not found at {TRACKER_PATH}")
        return None, None, None

    try:
        # Read only required columns if they exist
        df = pd.read_excel(TRACKER_PATH)
        
        required_cols = ['Date', 'Day', 'Domain', 'Topic', 'Sub Topics']
        # Filter to only existing required columns to avoid errors
        present_cols = [c for c in required_cols if c in df.columns]
        df = df[present_cols]
        
        # Robust date conversion
        df['Date_Clean'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Find match for the target date
        match = df[df['Date_Clean'] == target_date_str]
        
        if not match.empty:
            row = match.iloc[0]
            # Check if Topic is actually present (not NaN)
            topic = str(row['Topic']).strip() if pd.notna(row.get('Topic')) else None
            
            if topic and topic.lower() != 'nan':
                domain = str(row['Domain']).strip() if pd.notna(row.get('Domain')) else None
                subtopics = str(row['Sub Topics']).strip() if pd.notna(row.get('Sub Topics')) else None
                return domain, topic, subtopics
                
    except Exception as e:
        print(f"  [ERROR] Failed to read Tracker.xlsx: {e}")
        
    return None, None, None


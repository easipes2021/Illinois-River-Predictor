import dataretrieval.nwis as nwis
import pandas as pd

def update_hwy59_rating():
    site_id = '07195430'
    print(f"Fetching official USGS rating table for {site_id}...")
    
    try:
        # Pull the rating data
        rating_data, metadata = nwis.get_ratings(site_no=site_id)
        
        # USGS headers can vary. We'll look for the standard 'INDEP'/'DEP' 
        # or the expanded 'shift' columns if they exist.
        if 'INDEP' in rating_data.columns and 'DEP' in rating_data.columns:
            clean_df = rating_data[['INDEP', 'DEP']].copy()
        else:
            # Fallback: find columns by position if names changed
            # Usually column 0 is Height, column 1 is Discharge
            clean_df = rating_data.iloc[:, [0, 1]].copy()
            clean_df.columns = ['INDEP', 'DEP']

        # Save to CSV
        clean_df.to_csv('hwy59_official_rating.csv', index=False)
        print(f"Success! Official rating table saved with {len(clean_df)} points.")
        
        # Show a sample of the 'truth'
        print("\nSample of the Rating Table:")
        print(clean_df.head(3))
        
    except Exception as e:
        print(f"Error pulling rating table: {e}")
        print("Note: If the API is down, we will use the fallback formula in merge_data.py.")

if __name__ == "__main__":
    update_hwy59_rating()
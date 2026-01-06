"""
Simple and Intuitive Data Processing Pipeline
Handles: Loading → Cleaning → Feature Engineering → Scaling → Saving

Usage:
    python pipeline.py input.xlsx output.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import sys

class SimpleDataPipeline:
    """All-in-one data preprocessing pipeline"""
    
    def __init__(self, input_file, output_file="processed_data.csv"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.scaler = StandardScaler()
        
    def run(self):
        """Run complete pipeline"""
        print("="*60)
        print("DATA PROCESSING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.scale_features()
        self.save_data()
        
        print("\n✅ Pipeline Complete!")
        print(f"Output saved to: {self.output_file}")
        print(f"Final shape: {self.data.shape}")
        return self.data
    
    # ===== STEP 1: LOAD DATA =====
    def load_data(self):
        """Load data from file"""
        print("\n[1/5] Loading data...")
        
        if self.input_file.endswith('.xlsx'):
            self.data = pd.read_excel(self.input_file)
        elif self.input_file.endswith('.csv'):
            self.data = pd.read_csv(self.input_file)
        else:
            raise ValueError("File must be .xlsx or .csv")
        
        print(f"  Loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
    
    # ===== STEP 2: CLEAN DATA =====
    def clean_data(self):
        """Remove outliers and bad records"""
        print("\n[2/5] Cleaning data...")
        original_count = len(self.data)
        
        # Remove extreme outliers (adjust thresholds as needed)
        if 'sqft_lot' in self.data.columns:
            self.data = self.data[self.data['sqft_lot'] <= 213000]
        
        if 'sqft_living' in self.data.columns:
            self.data = self.data[self.data['sqft_living'] >= 500]
        
        if 'bedrooms' in self.data.columns:
            self.data = self.data[self.data['bedrooms'] >= 1]
        
        if 'bathrooms' in self.data.columns:
            self.data = self.data[self.data['bathrooms'] >= 0.5]
        
        if 'price' in self.data.columns:
            self.data = self.data[self.data['price'] >= 100000]
        
        removed = original_count - len(self.data)
        print(f"  Removed {removed} outliers ({removed/original_count*100:.2f}%)")
    
    # ===== STEP 3: ENGINEER FEATURES =====
    def engineer_features(self):
        """Create useful new features"""
        print("\n[3/5] Engineering features...")
        features_added = 0
        
        # Parse dates if present
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%dT%H%M%S')
            self.data['year'] = self.data['date'].dt.year
            self.data['month'] = self.data['date'].dt.month
            self.data['season'] = self.data['month'].apply(self._get_season)
            self.data.drop('date', axis=1, inplace=True)
            features_added += 3
        
        # Renovation features
        if 'yr_renovated' in self.data.columns:
            self.data['was_renovated'] = (self.data['yr_renovated'] > 0).astype(int)
            features_added += 1
        
        # Property age
        if 'yr_built' in self.data.columns and 'year' in self.data.columns:
            self.data['property_age'] = self.data['year'] - self.data['yr_built']
            features_added += 1
        
        # Room ratios
        if 'bedrooms' in self.data.columns and 'bathrooms' in self.data.columns:
            self.data['total_rooms'] = self.data['bedrooms'] + self.data['bathrooms']
            features_added += 1
        
        # Square footage ratios
        if 'sqft_living' in self.data.columns and 'sqft_lot' in self.data.columns:
            self.data['sqft_per_lot'] = self.data['sqft_living'] / (self.data['sqft_lot'] + 1)
            features_added += 1
        
        # Neighborhood comparisons
        if 'sqft_living' in self.data.columns and 'sqft_living15' in self.data.columns:
            self.data['above_neighborhood'] = (
                self.data['sqft_living'] > self.data['sqft_living15']
            ).astype(int)
            features_added += 1
        
        # Quality score
        if 'grade' in self.data.columns and 'condition' in self.data.columns:
            self.data['quality_score'] = self.data['grade'] * self.data['condition']
            features_added += 1
        
        # Log transforms for skewed features
        if 'price' in self.data.columns:
            self.data['price_log'] = np.log1p(self.data['price'])
            features_added += 1
        
        if 'sqft_living' in self.data.columns:
            self.data['sqft_living_log'] = np.log1p(self.data['sqft_living'])
            features_added += 1
        
        # ===== NEW REQUESTED FEATURES =====
        
        # days_on_market: Calculate days between listing and sale date
        # Note: If you have a listing_date column, use that instead
        if 'year' in self.data.columns and 'month' in self.data.columns:
            # Create sale date from year and month (using 15th as mid-month estimate)
            sale_date = pd.to_datetime(self.data[['year', 'month']].assign(day=15))
            # Estimate listing date as 30 days before sale (adjust threshold as needed)
            # You can modify this to use actual listing_date if available
            listing_date = sale_date - pd.Timedelta(days=30)
            self.data['days_on_market'] = (sale_date - listing_date).dt.days
            features_added += 1
        elif 'year' in self.data.columns:
            # Fallback: use average days on market (30 days)
            self.data['days_on_market'] = 30
            features_added += 1
        
        # years_since_renovation: Years from renovation year to current/sale year
        if 'yr_renovated' in self.data.columns:
            if 'year' in self.data.columns:
                # Use sale year if available
                self.data['years_since_renovation'] = self.data['year'] - self.data['yr_renovated']
                # Set to 0 for properties never renovated
                self.data.loc[self.data['yr_renovated'] == 0, 'years_since_renovation'] = 0
                features_added += 1
            else:
                # Use current year as fallback
                current_year = pd.Timestamp.now().year
                self.data['years_since_renovation'] = current_year - self.data['yr_renovated']
                self.data.loc[self.data['yr_renovated'] == 0, 'years_since_renovation'] = 0
                features_added += 1
        
        # sqft_lot_log: Log transform of sqft_lot
        if 'sqft_lot' in self.data.columns:
            self.data['sqft_lot_log'] = np.log1p(self.data['sqft_lot'])
            features_added += 1
        
        # bedrooms_per_bathroom: Ratio of bedrooms to bathrooms
        if 'bedrooms' in self.data.columns and 'bathrooms' in self.data.columns:
            # Avoid division by zero
            self.data['bedrooms_per_bathroom'] = self.data['bedrooms'] / (self.data['bathrooms'] + 0.01)
            features_added += 1
        
        # quality_grade_condition: Sum of grade and condition
        if 'grade' in self.data.columns and 'condition' in self.data.columns:
            # Add grade and condition
            self.data['quality_grade_condition'] = self.data['grade'] + self.data['condition']
            features_added += 1
        
        print(f"  Created {features_added} new features")
    
    # ===== STEP 4: SCALE FEATURES =====
    def scale_features(self):
        """Standardize numeric features"""
        print("\n[4/5] Scaling features...")
        
        # Select numeric columns (exclude ID and price)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['id', 'price'] if 'price' in self.data.columns else ['id']
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Scale
        self.data[scale_cols] = self.scaler.fit_transform(self.data[scale_cols])
        
        print(f"  Scaled {len(scale_cols)} columns")
    
    # ===== STEP 5: SAVE DATA =====
    def save_data(self):
        """Save processed data"""
        print("\n[5/5] Saving processed data...")
        
        # Save CSV
        self.data.to_csv(self.output_file, index=False)
        
        # Save scaler for future use
        scaler_file = self.output_file.replace('.csv', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_file)
        
        print(f"  Data saved: {self.output_file}")
        print(f"  Scaler saved: {scaler_file}")
    
    # ===== HELPER FUNCTIONS =====
    @staticmethod
    def _get_season(month):
        """Convert month to season (0-3)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Command line usage
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py input_file.xlsx [output_file.csv]")
        print("Example: python pipeline.py train.xlsx train_processed.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "processed_data.csv"
    
    # Run pipeline
    pipeline = SimpleDataPipeline(input_file, output_file)
    data = pipeline.run()
    
    # Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    if 'price' in data.columns:
        print(f"Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
    print("="*60)
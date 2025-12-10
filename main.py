import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from fastapi.staticfiles import StaticFiles
warnings.filterwarnings('ignore')

# Non-interactive backend
matplotlib.use('Agg')

app = FastAPI()

# ---------------------------------------
# 1. ENABLE CORS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Global Config
# ---------------------------------------
model = None
encoders = {}
feature_columns = []
numeric_columns = []
categorical_columns = []
MODEL_FILE = "lgbm_random_model.joblib"
ENCODERS_FILE = "encoders.joblib"
CSV_FILE = "home_price_original.csv"

# ---------------------------------------
# Helper Functions
# ---------------------------------------
def clean_numeric_column(df, column_name):
    """Clean a numeric column by converting to float and handling errors"""
    if column_name in df.columns:
        # Replace 'Bilinmiyor' with NaN
        df[column_name] = df[column_name].replace('Bilinmiyor', np.nan)
        # Replace empty strings with NaN
        df[column_name] = df[column_name].replace('', np.nan)
        # Convert to numeric, coercing errors to NaN
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        # Fill NaN with median
        median_val = df[column_name].median()
        if pd.isna(median_val):
            median_val = 0
        df[column_name] = df[column_name].fillna(median_val)
    return df

# ---------------------------------------
# 2. STARTUP: Smart Model Loading
# ---------------------------------------
@app.on_event("startup")
def startup_event():
    global model, encoders, feature_columns, numeric_columns, categorical_columns
    
    print("🚀 Starting up RealEstate AI Backend...")
    
    # --- A. Prepare Data ---
    if not os.path.exists(CSV_FILE):
        print(f"❌ CRITICAL: CSV file '{CSV_FILE}' not found.")
        return

    print(f"📊 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    print(f"📈 Initial data shape: {df.shape}")
    print(f"📊 Initial columns: {df.columns.tolist()}")
    
    # 1. Clean Data - replace empty strings and NaN with "Bilinmiyor" for categorical
    df = df.fillna("Bilinmiyor")
    df = df.replace("", "Bilinmiyor")
    
    # 2. Define column types based on your CSV
    numeric_columns = [
        'Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 
        'Banyo_Sayısı', 'Binanın_Kat_Sayısı'
    ]
    
    categorical_columns = [
        'Şehir', 'Bulunduğu_Kat', 'Isıtma_Tipi', 'Binanın_Yaşı',
        'Eşya_Durumu', 'Kullanım_Durumu', 'Yatırıma_Uygunluk',
        'Tapu_Durumu', 'Takas'
    ]
    
    # 3. Clean numeric columns
    print("🧹 Cleaning numeric columns...")
    for col in numeric_columns:
        if col in df.columns:
            df = clean_numeric_column(df, col)
            print(f"   - {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, median={df[col].median():.1f}")
    
    # 4. Clean Fiyat (target variable)
    if 'Fiyat' in df.columns:
        df = clean_numeric_column(df, 'Fiyat')
        print(f"   - Fiyat: min={df['Fiyat'].min():.1f}, max={df['Fiyat'].max():.1f}, median={df['Fiyat'].median():.1f}")
    
    # 5. Encode categorical variables
    print("🔠 Encoding categorical variables...")
    for col in categorical_columns:
        if col in df.columns:
            # Ensure 'Bilinmiyor' is always in the categories
            unique_vals = list(df[col].unique())
            if 'Bilinmiyor' not in unique_vals:
                unique_vals.append('Bilinmiyor')
            
            le = LabelEncoder()
            le.fit(unique_vals)
            df[col] = le.transform(df[col])
            encoders[col] = le
            print(f"   - {col}: {len(unique_vals)} unique values")
    
    # 6. Define feature columns (all columns except 'Fiyat')
    feature_columns = [col for col in df.columns if col != 'Fiyat']
    
    # 7. Ensure all columns are numeric
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any remaining NaN values
    df = df.dropna(subset=feature_columns + ['Fiyat'])
    
    # 8. Prepare X and y
    X = df[feature_columns]
    y = df['Fiyat']
    
    print(f"✅ Final data shape: {df.shape}")
    print(f"📋 Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"💰 Price range: {y.min():.0f} - {y.max():.0f} TRY")
    
    # 9. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"📚 Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # --- B. Load or Train Model ---
    need_retrain = False
    
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODERS_FILE):
        print(f"🔄 Attempting to load existing model and encoders...")
        try:
            model = joblib.load(MODEL_FILE)
            loaded_encoders = joblib.load(ENCODERS_FILE)
            
            # Test the model
            test_input = X_test.iloc[:1]
            test_pred = model.predict(test_input)
            print(f"✅ Existing model works! Test prediction: {test_pred[0]:.0f}")
            
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            print("🔄 Training new model...")
            need_retrain = True
    else:
        print("⚠️ No existing model/encoders found.")
        need_retrain = True

    # --- C. Train if needed ---
    if need_retrain:
        print("🚀 Training new LightGBM model...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"📊 Model trained! Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Save model and encoders
        joblib.dump(model, MODEL_FILE)
        joblib.dump(encoders, ENCODERS_FILE)
        print(f"💾 Model and encoders saved.")

    # --- D. Generate Visualizations ---
    print("🎨 Generating visualizations...")
    os.makedirs("visualizations", exist_ok=True)
    
    # Predict for charts
    y_pred = model.predict(X_test)

    # 1. Feature Importance (Top 10 features)
    plt.figure(figsize=(12, 8))
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        # Create more readable feature names
        feature_names = {
            'Net_Metrekare': 'Net Area',
            'Brüt_Metrekare': 'Gross Area',
            'Oda_Sayısı': 'Room Count',
            'Banyo_Sayısı': 'Bathroom Count',
            'Binanın_Kat_Sayısı': 'Building Floors',
            'Şehir': 'City',
            'Bulunduğu_Kat': 'Floor Level',
            'Isıtma_Tipi': 'Heating Type',
            'Binanın_Yaşı': 'Building Age',
            'Eşya_Durumu': 'Furnished',
            'Kullanım_Durumu': 'Usage Status',
            'Yatırıma_Uygunluk': 'Investment Suit.',
            'Tapu_Durumu': 'Title Deed',
            'Takas': 'Swap Available'
        }
        
        feature_imp['feature_name'] = feature_imp['feature'].map(
            lambda x: feature_names.get(x, x)
        )
        
        sns.barplot(x='importance', y='feature_name', data=feature_imp, palette='viridis')
        plt.title("Top 10 Feature Importance", fontsize=16, fontweight='bold')
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "Feature importance not available", ha='center', va='center')
    plt.savefig('visualizations/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 2. Prediction vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#10b981', s=20)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("Actual Price (TRY)", fontsize=12)
    plt.ylabel("Predicted Price (TRY)", fontsize=12)
    plt.title("Actual vs Predicted Price", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/prediction_vs_actual.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 3. Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#ef4444', bins=30)
    plt.title("Residual Error Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Residual (Actual - Predicted) TRY", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/residuals.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 4. Correlation (Top features only)
    plt.figure(figsize=(12, 10))
    # Select top 8 features plus price for correlation matrix
    if 'feature_imp' in locals():
        top_features = feature_imp['feature'].head(8).tolist()
    else:
        top_features = feature_columns[:8]
    
    corr_df = df[['Fiyat'] + top_features]
    corr_matrix = corr_df.corr()
    
    # Create readable feature names for correlation matrix
    feature_names_corr = {
        'Net_Metrekare': 'Net Area',
        'Brüt_Metrekare': 'Gross Area',
        'Oda_Sayısı': 'Room Count',
        'Banyo_Sayısı': 'Bathroom',
        'Binanın_Kat_Sayısı': 'Building Floors',
        'Şehir': 'City',
        'Bulunduğu_Kat': 'Floor',
        'Isıtma_Tipi': 'Heating',
        'Binanın_Yaşı': 'Age',
        'Eşya_Durumu': 'Furnished',
        'Kullanım_Durumu': 'Usage',
        'Yatırıma_Uygunluk': 'Invest.',
        'Tapu_Durumu': 'Title',
        'Takas': 'Swap',
        'Fiyat': 'Price'
    }
    
    # Rename columns for display
    display_columns = [feature_names_corr.get(col, col) for col in corr_matrix.columns]
    corr_matrix_display = pd.DataFrame(corr_matrix.values, 
                                       columns=display_columns, 
                                       index=display_columns)
    
    mask = np.triu(np.ones_like(corr_matrix_display, dtype=bool))
    sns.heatmap(corr_matrix_display, annot=True, cmap='coolwarm', fmt=".2f", 
                center=0, mask=mask, linewidths=0.5)
    plt.title("Correlation Matrix (Top Features)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("✅ Visualizations updated.")
    print("🎯 Backend ready to accept predictions!")

# ---------------------------------------
# 3. MOUNT STATIC FILES
# ---------------------------------------
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# ---------------------------------------
# Input Schema & Endpoint
# ---------------------------------------
class HouseFeatures(BaseModel):
    # All fields from your CSV with default values
    Net_Metrekare: float = Field(None, description="Net area in square meters")
    Brüt_Metrekare: float = Field(None, description="Gross area in square meters")
    Oda_Sayısı: float = Field(None, description="Number of rooms")
    Bulunduğu_Kat: str = Field(None, description="Floor level")
    Eşya_Durumu: str = Field(None, description="Furnished status")
    Binanın_Yaşı: str = Field(None, description="Building age")
    Isıtma_Tipi: str = Field(None, description="Heating type")
    Şehir: str = Field(None, description="City")
    Binanın_Kat_Sayısı: float = Field(None, description="Total floors in building")
    Kullanım_Durumu: str = Field(None, description="Usage status")
    Yatırıma_Uygunluk: str = Field(None, description="Investment suitability")
    Takas: str = Field(None, description="Swap available")
    Tapu_Durumu: str = Field(None, description="Title deed status")
    Banyo_Sayısı: float = Field(None, description="Number of bathrooms")

@app.post("/predict")
def predict_price(data: HouseFeatures):
    if not model or not encoders:
        raise HTTPException(status_code=500, detail="Model or encoders not loaded.")
    
    try:
        # Prepare input dictionary
        input_dict = {}
        
        # Process numeric columns
        for col in numeric_columns:
            value = getattr(data, col, None)
            if value is None or value == "":
                # Use median from training data or 0
                input_dict[col] = 0.0
            else:
                try:
                    input_dict[col] = float(value)
                except:
                    input_dict[col] = 0.0
        
        # Process categorical columns
        for col in categorical_columns:
            value = getattr(data, col, None)
            if value is None or value == "":
                input_dict[col] = "Bilinmiyor"
            else:
                input_dict[col] = str(value)
        
        # Create DataFrame row
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        for col in categorical_columns:
            if col in input_df.columns and col in encoders:
                try:
                    # Check if value exists in encoder
                    value = input_df[col].iloc[0]
                    if value in encoders[col].classes_:
                        input_df[col] = encoders[col].transform([value])[0]
                    else:
                        # Use "Bilinmiyor" encoding
                        input_df[col] = encoders[col].transform(["Bilinmiyor"])[0]
                except:
                    # Fallback to "Bilinmiyor"
                    input_df[col] = encoders[col].transform(["Bilinmiyor"])[0]
        
        # Ensure all feature columns are present and in correct order
        for col in feature_columns:
            if col not in input_df.columns:
                if col in numeric_columns:
                    input_df[col] = 0.0
                else:
                    input_df[col] = encoders[col].transform(["Bilinmiyor"])[0]
        
        input_df = input_df[feature_columns]
        
        # Ensure all values are numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Fill any remaining NaN
        input_df = input_df.fillna(0)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {
            "predicted_price": float(prediction[0]),
            "status": "success",
            "features_used": len(feature_columns),
            "currency": "TRY",
            "confidence": "high"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}\n{error_details}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/cities")
def get_cities():
    """Return list of all Turkish cities"""
    turkish_cities = [
        "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", 
        "Antalya", "Ardahan", "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", 
        "Bayburt", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", 
        "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Düzce", "Edirne", "Elazığ", 
        "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", 
        "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul", "İzmir", "Kahramanmaraş", 
        "Karabük", "Karaman", "Kars", "Kastamonu", "Kayseri", "Kırıkkale", "Kırklareli", 
        "Kırşehir", "Kilis", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", 
        "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Osmaniye", 
        "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Şanlıurfa", 
        "Şırnak", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", 
        "Yozgat", "Zonguldak"
    ]
    return {"cities": turkish_cities}

@app.get("/features")
def get_features():
    """Return available feature columns"""
    return {
        "features": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns
    }

@app.get("/stats")
def get_stats():
    """Return dataset statistics"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return {
            "total_samples": len(df),
            "columns": list(df.columns),
            "price_stats": {
                "min": float(df['Fiyat'].min()),
                "max": float(df['Fiyat'].max()),
                "mean": float(df['Fiyat'].mean()),
                "median": float(df['Fiyat'].median())
            } if 'Fiyat' in df.columns else {}
        }
    return {"error": "Dataset not found"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="site")    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
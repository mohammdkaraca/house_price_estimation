# visualization_script.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_visualize_data(filename):
    """
    Load cleaned real estate data and create visualization graphs
    """
    # Load the cleaned data
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        print(f"✓ Loaded data from {filename}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print(os.listdir(r"C:\Users\moham\OneDrive\Desktop\VS_CODE_FILES\Home_prices_estimation"))


        print("Please run the cleaning script first to create home_price_clean.csv")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Create a folder for saving graphs
    output_folder = "visualizations"
    Path(output_folder).mkdir(exist_ok=True)
    
    # Create figure 1: Main price-area relationships
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Real Estate Analysis: Price and Area Relationships', 
                  fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Price vs Net Area (scatter with dots)
    ax1 = axes1[0, 0]
    ax1.scatter(df["Net_Metrekare"], df["Fiyat"], 
                alpha=0.6, s=15, color='blue', edgecolor='none')
    ax1.set_xlabel('Net Metrekare (m²)', fontsize=11)
    ax1.set_ylabel('Fiyat (TL)', fontsize=11)
    ax1.set_title('Price vs Net Area', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')  # Disable scientific notation
    
    # 2. Price per square meter vs Net Area
    ax2 = axes1[0, 1]
    if "Net_Metrekare" in df.columns and "Fiyat" in df.columns:
        price_per_sqm = df["Fiyat"] / df["Net_Metrekare"]
        ax2.scatter(df["Net_Metrekare"], price_per_sqm, 
                    alpha=0.6, s=15, color='green', edgecolor='none')
        ax2.set_xlabel('Net Metrekare (m²)', fontsize=11)
        ax2.set_ylabel('Price per m² (TL/m²)', fontsize=11)
        ax2.set_title('Price per m² vs Net Area', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    
    # 3. Net vs Gross Area comparison
    ax3 = axes1[1, 0]
    if "Net_Metrekare" in df.columns and "Brüt_Metrekare" in df.columns:
        ax3.scatter(df["Net_Metrekare"], df["Brüt_Metrekare"], 
                    alpha=0.6, s=15, color='purple', edgecolor='none')
        
        # Add 45-degree line (Net = Gross)
        max_val = max(df["Net_Metrekare"].max(), df["Brüt_Metrekare"].max())
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Net = Brüt')
        
        ax3.set_xlabel('Net Metrekare (m²)', fontsize=11)
        ax3.set_ylabel('Brüt Metrekare (m²)', fontsize=11)
        ax3.set_title('Net vs Brüt Metrekare Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_aspect('equal', adjustable='box')
    
    # 4. Price vs Building Age
    ax4 = axes1[1, 1]
    if "Binanın_Yaşı" in df.columns:
        ax4.scatter(df["Binanın_Yaşı"], df["Fiyat"], 
                    alpha=0.6, s=15, color='orange', edgecolor='none')
        ax4.set_xlabel('Binanın Yaşı (Years)', fontsize=11)
        ax4.set_ylabel('Fiyat (TL)', fontsize=11)
        ax4.set_title('Price vs Building Age', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    fig1.savefig(f'{output_folder}/price_area_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/price_area_analysis.png")
    
    # Create figure 2: Room and distribution analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Real Estate Analysis: Rooms and Distributions', 
                  fontsize=16, fontweight='bold', y=1.02)
    
    # 5. Price vs Room Count (dot plot with jitter)
    ax5 = axes2[0, 0]
    if "Oda_Sayısı" in df.columns:
        # Add small random jitter to x-values for better visualization
        jitter = np.random.normal(0, 0.05, size=len(df))
        ax5.scatter(df["Oda_Sayısı"] + jitter, df["Fiyat"], 
                    alpha=0.5, s=20, color='red', edgecolor='none')
        ax5.set_xlabel('Oda Sayısı', fontsize=11)
        ax5.set_ylabel('Fiyat (TL)', fontsize=11)
        ax5.set_title('Price vs Room Count (with jitter)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.ticklabel_format(style='plain', axis='y')
    
    # 6. Average price by room count
    ax6 = axes2[0, 1]
    if "Oda_Sayısı" in df.columns and "Fiyat" in df.columns:
        room_price_stats = df.groupby('Oda_Sayısı')['Fiyat'].agg(['mean', 'median', 'count']).reset_index()
        room_price_stats = room_price_stats[room_price_stats['count'] >= 5]  # Only show with enough data
        
        ax6.scatter(room_price_stats["Oda_Sayısı"], room_price_stats["mean"], 
                    s=room_price_stats["count"]/10,  # Size based on count
                    alpha=0.7, color='darkblue', edgecolor='black', linewidth=1)
        ax6.set_xlabel('Oda Sayısı', fontsize=11)
        ax6.set_ylabel('Average Fiyat (TL)', fontsize=11)
        ax6.set_title('Average Price by Room Count (size = sample size)', 
                     fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    # 7. Building floor count distribution
    ax7 = axes2[1, 0]
    if "Binanın_Kat_Sayısı" in df.columns:
        floor_counts = df["Binanın_Kat_Sayısı"].value_counts().sort_index()
        ax7.scatter(floor_counts.index, floor_counts.values, 
                    s=50, alpha=0.7, color='teal', edgecolor='black')
        ax7.set_xlabel('Binanın Kat Sayısı', fontsize=11)
        ax7.set_ylabel('Number of Properties', fontsize=11)
        ax7.set_title('Distribution of Building Floor Counts', 
                     fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    
    # 8. Bathroom vs Room count
    ax8 = axes2[1, 1]
    if "Banyo_Sayısı" in df.columns and "Oda_Sayısı" in df.columns:
        # Create a heatmap-style dot plot
        ax8.scatter(df["Oda_Sayısı"], df["Banyo_Sayısı"], 
                    alpha=0.4, s=15, color='magenta', edgecolor='none')
        ax8.set_xlabel('Oda Sayısı', fontsize=11)
        ax8.set_ylabel('Banyo Sayısı', fontsize=11)
        ax8.set_title('Bathroom Count vs Room Count', 
                     fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(f'{output_folder}/rooms_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_folder}/rooms_distribution_analysis.png")
    
    # Create figure 3: Interactive correlation matrix (as scatter plot matrix)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        # Take top 6 numeric columns for readability
        numeric_cols = numeric_cols[:6]
        
        fig3, axes3 = plt.subplots(len(numeric_cols), len(numeric_cols), 
                                   figsize=(18, 18))
        fig3.suptitle('Pairwise Scatter Plot Matrix', fontsize=18, fontweight='bold', y=1.02)
        
        for i, col_i in enumerate(numeric_cols):
            for j, col_j in enumerate(numeric_cols):
                ax = axes3[i, j]
                
                if i == j:
                    # Diagonal: Show histogram
                    ax.hist(df[col_i].dropna(), bins=30, alpha=0.7, color='skyblue')
                    ax.set_title(f'Distribution of {col_i}', fontsize=10)
                else:
                    # Off-diagonal: Show scatter plot
                    ax.scatter(df[col_j], df[col_i], 
                               alpha=0.3, s=10, color='gray', edgecolor='none')
                
                # Set labels only on edges
                if i == len(numeric_cols) - 1:
                    ax.set_xlabel(col_j, fontsize=9)
                if j == 0:
                    ax.set_ylabel(col_i, fontsize=9)
                
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        fig3.savefig(f'{output_folder}/scatter_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_folder}/scatter_matrix.png")
    
    # Show basic statistics
    print("\n" + "="*50)
    print("DATA SUMMARY STATISTICS")
    print("="*50)
    print(f"Total properties: {len(df)}")
    
    if "Fiyat" in df.columns:
        print(f"\nPrice Statistics (Fiyat):")
        print(f"  Mean: {df['Fiyat'].mean():,.0f} TL")
        print(f"  Median: {df['Fiyat'].median():,.0f} TL")
        print(f"  Min: {df['Fiyat'].min():,.0f} TL")
        print(f"  Max: {df['Fiyat'].max():,.0f} TL")
        print(f"  Std Dev: {df['Fiyat'].std():,.0f} TL")
    
    if "Net_Metrekare" in df.columns and "Fiyat" in df.columns:
        price_per_sqm = df["Fiyat"] / df["Net_Metrekare"]
        print(f"\nPrice per m² Statistics:")
        print(f"  Mean: {price_per_sqm.mean():,.0f} TL/m²")
        print(f"  Median: {price_per_sqm.median():,.0f} TL/m²")
        print(f"  Min: {price_per_sqm.min():,.0f} TL/m²")
        print(f"  Max: {price_per_sqm.max():,.0f} TL/m²")
    
    print("\nVisualizations have been saved in the 'visualizations' folder.")
    print("="*50)
    
    # Show plots
    plt.show()
    
    return df

# Run the visualization
if __name__ == "__main__":
    # You can change the filename if you saved it differently
    df = load_and_visualize_data("home_price_cleaned_OHE.csv")
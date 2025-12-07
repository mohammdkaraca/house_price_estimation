import pandas as pd
import numpy as np

def review_and_optionally_delete(df, condition, description):


    outliers = df[condition]

    if outliers.empty:
        print(f"\nNo outliers found for: {description}")
        return df
    
    print(f"\n==============================")
    print(f"OUTLIERS FOUND: {description}")
    print(f"Count: {len(outliers)}")
    print(outliers.head(10))  # show first 10 outliers
    print("==============================")

    choice = input("Delete these rows? (y/n): ").strip().lower()
    
    if choice == "y":
        print("Deleting rows...\n")
        df = df[~condition]
    else:
        print("Keeping all rows.\n")

    return df

# Make a working copy
clean_df = df.copy()


condition = ~clean_df["Net_Metrekare"].between(10, 2000)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Unrealistic Net_Metrekare (<10 or >2000)")


condition = ~clean_df["Brüt_Metrekare"].between(20, 4000)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Unrealistic Brüt_Metrekare (<20 or >4000)")


condition = clean_df["Net_Metrekare"] > clean_df["Brüt_Metrekare"]
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Net_Metrekare greater than Brüt_Metrekare")


condition = ~clean_df["Oda_Sayısı"].between(0.5, 15)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Unrealistic Oda_Sayısı (not between 0.5 and 15)")

condition = ~clean_df["Banyo_Sayısı"].between(0, 10)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Unrealistic Banyo_Sayısı (not between 0 and 10)")

condition = clean_df["Banyo_Sayısı"] > clean_df["Oda_Sayısı"] + 2
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Banyo_Sayısı > Oda_Sayısı + 2")

# Negative or too small price
condition = clean_df["Fiyat"] <= 300000
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Fiyat ≤ 300000 TL (incorrect low price)")

condition = clean_df["Fiyat"] >= 500_000_000
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Fiyat ≥ 500M TL (unrealistic high price)")

clean_df["Binanın_Yaşı"] = pd.to_numeric(clean_df["Binanın_Yaşı"], errors="coerce")
condition = ~clean_df["Binanın_Yaşı"].between(0, 150)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Binanın_Yaşı not between 0 and 150")

condition = ~clean_df["Binanın_Kat_Sayısı"].between(1, 200)
clean_df = review_and_optionally_delete(clean_df, condition,
                                        "Unrealistic Binanın_Kat_Sayısı (not between 1 and 200)")

output_filename = "home_price_clean.csv"
clean_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n✓ Cleaned data saved to: {output_filename}")
print(f"✓ File contains {len(clean_df)} rows and {len(clean_df.columns)} columns")


print("\n" + "="*50)
print("CLEANING COMPLETE - SUMMARY")
print("="*50)
print(f"Original dataset rows: {len(df)}")
print(f"Cleaned dataset rows:  {len(clean_df)}")
print(f"Total rows removed:    {len(df) - len(clean_df)}")
print(f"Removal percentage:    {(len(df) - len(clean_df)) / len(df) * 100:.2f}%")
print("="*50)

# Optional: Show first few rows of cleaned data
if input("\nShow first 5 rows of cleaned data? (y/n): ").lower() == "y":
    print("\nFirst 5 rows of cleaned data:")
    print(clean_df.head())
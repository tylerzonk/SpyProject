import pandas as pd
print("Pandas version:", pd.__version__)
def clean_data():
    # Read the raw data
    df = pd.read_csv("/app/sample_data.csv")

    # Convert the value column to numeric, force-converting strings to numbers
    df["value"] = pd.to_numeric(df["value"])

    # Save the cleaned data
    df.to_csv("/app/data_cleaned.csv", index=False)

    print("Data cleaned successfully!")
    print("Original data types:", df.dtypes)

if __name__ == "__main__":
    print("Cleaning data...")
    clean_data()
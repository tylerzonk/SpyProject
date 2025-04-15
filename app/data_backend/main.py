from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/data")
def read_data():
    # Read the CSV file
    df = pd.read_csv("sample_data.csv")
    # Convert DataFrame to a list of dictionaries
    return df.to_dict(orient="records")
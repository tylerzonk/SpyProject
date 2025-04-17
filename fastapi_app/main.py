from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from functions.spy_updater import run_full_sync, fetch_yf_current

# --- Async background fetcher ---
async def recurring_fetch():
    while True:
        try:
            print("Fetching current 1-min candle...")
            fetch_yf_current()
        except Exception as e:
            print(f"[ERROR] fetch_yf_current failed: {e}")
        await asyncio.sleep(60)

# --- Lifespan setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Running full sync on startup...")

    # Run blocking sync code in a thread to avoid blocking event loop
    await asyncio.to_thread(run_full_sync)

    # Start recurring async task
    task = asyncio.create_task(recurring_fetch())

    yield  # App runs here

    # Handle shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Recurring fetch task cancelled.")

# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "SPY API is running"}

import asyncio
from functions.spy_updater import run_full_sync, fetch_yf_current

def start_background_tasks():
    # Run full sync once at startup
    print("Running full sync on startup...")
    run_full_sync()

    # Start periodic 1-min data fetch in background
    asyncio.create_task(recurring_fetch())

async def recurring_fetch():
    while True:
        try:
            print("Fetching current 1-min candle...")
            fetch_yf_current()
        except Exception as e:
            print(f"[ERROR] fetch_yf_current failed: {e}")
        await asyncio.sleep(60)

from fastapi import FastAPI
from spy_updater_runner import start_background_tasks

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    start_background_tasks()
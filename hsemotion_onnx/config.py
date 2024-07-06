import os

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "enet_b0_8_best_vgaf")
    MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    SKIP_FRAME = int(os.getenv("SKIP_FRAME", 1))
    TIMEOUT = int(os.getenv("TIMEOUT", 999999))
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 5))

config = Config()

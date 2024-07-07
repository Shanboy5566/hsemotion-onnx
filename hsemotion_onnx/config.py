import os

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "enet_b0_8_best_vgaf")
    MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    SKIP_FRAME = int(os.getenv("SKIP_FRAME", 1))
    TIMEOUT = int(os.getenv("TIMEOUT", 999999))
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 5))
    BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 10))

config = Config()

# frame: 0.03s
# windows: 0.167s 
# buffer: 0.03 * (5 + 10 - 1) = 0.42s
# [X X X X X] X X X X X 
# X [X X X X X] X X X X
# X X [X X X X X] X X X

# 1/fps * (window_size + buffer_size - 1)
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

BATTLE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [{extra[category]}] | {message}"
)

SP_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | [selfplay] | {message}"

DEFAULT_LEVEL = "INFO"

METRICS_CACHE_SIZE = 100

SAMPLE_INTERVAL = 10
SAMPLES_PER_WRITE = 6
NVIDIA_SMI_TIMEOUT = 5

TRAINING_DETAIL_LOG_INTERVAL = 1
TRAINING_STATUS_SAVE_INTERVAL = 50

RECENT_METRICS_WINDOW = 10
HISTORICAL_METRICS_WINDOW = 100

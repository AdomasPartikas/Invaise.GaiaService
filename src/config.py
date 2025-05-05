import os
from datetime import datetime

# Model parameters
MODEL_NAME = "GAIA"
MODEL_VERSION = "0.1.0"
MODEL_DESCRIPTION = "Ensemble model combining Apollo and Ignis predictions"

# API endpoints
ASPNET_URL = os.getenv("BUSINESS_DOMAIN_URL", "http://localhost:5116")

# Model prediction endpoints - updated to match actual API controller routes
PREDICTION_ENDPOINT = ASPNET_URL + "/api/ModelPrediction"
PORTFOLIO_ENDPOINT = ASPNET_URL + "/api/Portfolio/{portfolio_id}"
BATCH_PREDICTION_ENDPOINT = PREDICTION_ENDPOINT + "/batch"
HISTORICAL_PREDICTION_ENDPOINT = PREDICTION_ENDPOINT + "/{symbol}/{modelSource}/history"
REFRESH_PREDICTION_ENDPOINT = PREDICTION_ENDPOINT + "/{symbol}/refresh"
ALL_PREDICTIONS_ENDPOINT = PREDICTION_ENDPOINT + "/{symbol}/all"

# Apollo and Ignis-specific endpoints updated to match the controller route pattern
# These should be used with /symbol/Apollo or /symbol/Ignis appended
APOLLO_ENDPOINT = PREDICTION_ENDPOINT
IGNIS_ENDPOINT = PREDICTION_ENDPOINT

# File paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model weights (how much we trust each model)
DEFAULT_APOLLO_WEIGHT = 0.6  # Higher weight for historical data
DEFAULT_IGNIS_WEIGHT = 0.4   # Lower weight for real-time data (more volatile)

# Portfolio optimization parameters
RISK_TOLERANCE_PARAMETER = 0.5  # Default risk tolerance (0-1), can be overridden by user preferences
MAX_POSITION_SIZE = 0.2         # Maximum position size as a fraction of portfolio

# Heat score thresholds
HEAT_SCORE_THRESHOLD_HIGH = 0.7   # High confidence threshold
HEAT_SCORE_THRESHOLD_MEDIUM = 0.5 # Medium confidence threshold
HEAT_SCORE_THRESHOLD_LOW = 0.3    # Low confidence threshold

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(DATA_DIR, f"{MODEL_NAME.lower()}_log_{datetime.now().strftime('%Y%m%d')}.log") 

GAIA_API_PORT = 8001


# auth
GAIA_SERVICE_ACCOUNT_ID = os.getenv("GAIA_SERVICE_ACCOUNT_ID", "gaia_service_account")
GAIA_SERVICE_ACCOUNT_KEY = os.getenv("GAIA_SERVICE_ACCOUNT_KEY", "gaia_service_account_key")
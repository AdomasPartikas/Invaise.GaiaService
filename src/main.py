import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import sys
import requests

import config
from gaia import GaiaModel
from data_fetcher import DataFetcher
from logger_config import setup_logging

# Set up logging for all modules
setup_logging()
logger = logging.getLogger("gaia.main")

# Initialize FastAPI app
app = FastAPI(
    title="Gaia API",
    description="API for Gaia ensemble model combining Apollo and Ignis predictions.",
    version=config.MODEL_VERSION,
)

# Initialize Gaia model and data fetcher
data_fetcher = DataFetcher()
gaia_model = GaiaModel(data_fetcher)

# Define request and response models
class PredictionRequest(BaseModel):
    symbol: str
    portfolio_id: str

class OptimizationRequest(BaseModel):
    portfolio_id: str

class WeightAdjustRequest(BaseModel):
    apollo_weight: float
    ignis_weight: float

class HeatData(BaseModel):
    heat_score: float
    confidence: float
    direction: str
    timestamp: str
    source: str
    explanation: str
    apollo_contribution: float
    ignis_contribution: float
    prediction_id: int
    model_version: str
    prediction_target: str
    current_price: float
    predicted_price: float

class RecommendationData(BaseModel):
    symbol: str
    action: str
    currentQuantity: float
    targetQuantity: float
    currentWeight: float
    targetWeight: float
    explanation: str

class OptimizationResponse(BaseModel):
    id: str
    portfolioId: str
    userId: str
    timestamp: str
    explanation: str
    confidence: float
    riskTolerance: float
    isApplied: bool
    modelVersion: str
    sharpeRatio: float
    meanReturn: float
    variance: float
    expectedReturn: float
    projectedSharpeRatio: float = 0.0
    projectedMeanReturn: float = 0.0
    projectedVariance: float = 0.0
    projectedExpectedReturn: float = 0.0
    recommendations: List[RecommendationData]
    symbolsProcessed: List[str]
    portfolioStrategy: str

class PredictionResponse(BaseModel):
    symbol: str
    combined_heat: HeatData

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: str

# Track startup time for uptime calculation
startup_time = datetime.now()

# API routes
@app.get("/health", response_model=HealthResponse)
async def health():
    """Check the health status of the API"""
    uptime = datetime.now() - startup_time
    return {
        "status": "ok", 
        "version": config.MODEL_VERSION,
        "uptime": str(uptime)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get combined predictions from Apollo and Ignis models for a symbol"""
    symbol = request.symbol
    portfolio_id = request.portfolio_id
    logger.info(f"Prediction request received for symbol: {symbol}")
    
    # Fetch predictions from Apollo and Ignis
    apollo_data, ignis_data = data_fetcher.fetch_all_predictions(symbol)
    
    if not apollo_data and not ignis_data:
        raise HTTPException(status_code=404, detail=f"Could not get predictions for {symbol}")
    
    # Get portfolio strategy if portfolio_id is provided
    portfolio_strategy = None
    if portfolio_id:
        portfolio_data = data_fetcher.fetch_portfolio_data(portfolio_id)
        if portfolio_data:
            portfolio_strategy = portfolio_data.get("strategyDescription", "Balanced")
            logger.info(f"Using portfolio strategy: {portfolio_strategy} for prediction")
    
    # Combine predictions with portfolio strategy
    combined_heat = gaia_model.combine_heat_scores(apollo_data, ignis_data, portfolio_strategy)
    
    logger.info(f"Combined heat result for {symbol}: {combined_heat}")
    
    return {
        "symbol": symbol,
        "combined_heat": combined_heat
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """Optimize portfolio based on predictions and portfolio data"""
    portfolio_id = request.portfolio_id
    
    # If symbols not provided, fetch them from the portfolio stock endpoint
    try:
        # Fetch portfolio stocks from API
        endpoint = f"{config.ASPNET_URL}/api/PortfolioStock/portfolio/{portfolio_id}"
        
        # Get auth headers if available
        headers = {}
        if hasattr(data_fetcher, 'auth_service'):
            headers = data_fetcher.auth_service.get_auth_headers()
            
        response = requests.get(endpoint, headers=headers, timeout=5)
        if response.status_code == 200:
            portfolio_stocks = response.json()
            # Extract unique symbols
            symbols = []
            seen_symbols = set()
            for stock in portfolio_stocks:
                symbol = stock.get("symbol")
                if symbol and symbol not in seen_symbols:
                    symbols.append(symbol)
                    seen_symbols.add(symbol)
                
            logger.info(f"Fetched {len(symbols)} symbols from portfolio stocks API: {', '.join(symbols)}")
        else:
            logger.error(f"Failed to fetch portfolio stocks: Status {response.status_code}")
            raise HTTPException(status_code=404, detail=f"Could not fetch portfolio stocks for portfolio ID {portfolio_id}")
    except Exception as e:
        logger.error(f"Error fetching portfolio stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio stocks: {str(e)}")
    
    if not symbols:
        raise HTTPException(status_code=404, detail="No symbols found in portfolio")
    
    logger.info(f"Portfolio optimization request for portfolio {portfolio_id} with {len(symbols)} symbols")
    
    # Use the analyze_portfolio method which handles everything in one call
    optimization_result = await gaia_model.analyze_portfolio(portfolio_id)
    
    if optimization_result.get("status") == "error":
        error_message = optimization_result.get("error", "Unknown error")
        logger.error(f"Portfolio optimization failed: {error_message}")
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {error_message}")
    
    # Extract the first optimization result if it exists
    optimizations = optimization_result.get("optimizations", [])
    if optimizations and len(optimizations) > 0:
        # Check if the optimization actually contains recommendations
        if "recommendations" in optimizations[0] and len(optimizations[0]["recommendations"]) > 0:
            return optimizations[0]
        else:
            # If we have an optimization but it has no recommendations (indicates no changes needed)
            logger.info("Optimization has no recommendations - no changes needed")
            
            # Return the optimization with proper explanation about no changes needed
            optimization = optimizations[0]
            if "explanation" not in optimization or not optimization["explanation"]:
                optimization["explanation"] = "No significant changes needed at this time. Market conditions don't warrant portfolio adjustments."
                
            return optimization
    
    # If we don't have a proper optimization result, return a default response with explanation
    logger.warning("No valid optimization produced, returning default response")
    
    return {
        "id": str(id),
        "portfolioId": portfolio_id,
        "userId": "unknown",  # This will be overridden by the client
        "timestamp": datetime.now().isoformat(),
        "explanation": "Insufficient data or market signals to generate meaningful optimization recommendations at this time.",
        "confidence": 0.0,
        "riskTolerance": 0.0,
        "isApplied": False,
        "modelVersion": config.MODEL_VERSION,
        "sharpeRatio": 0.0,
        "meanReturn": 0.0,
        "variance": 0.0,
        "expectedReturn": 0.0,
        "projectedSharpeRatio": 0.0,
        "projectedMeanReturn": 0.0,
        "projectedVariance": 0.0,
        "projectedExpectedReturn": 0.0,
        "recommendations": [],
        "symbolsProcessed": symbols,
        "portfolioStrategy": "Balanced"
    }

@app.post("/adjust-weights")
async def adjust_weights(request: WeightAdjustRequest):
    """Adjust the weights of Apollo and Ignis models in the ensemble"""
    if request.apollo_weight + request.ignis_weight != 1.0:
        # Normalize weights to sum to 1.0
        total = request.apollo_weight + request.ignis_weight
        apollo_weight = request.apollo_weight / total
        ignis_weight = request.ignis_weight / total
    else:
        apollo_weight = request.apollo_weight
        ignis_weight = request.ignis_weight
    
    # Update weights in config
    config.DEFAULT_APOLLO_WEIGHT = apollo_weight
    config.DEFAULT_IGNIS_WEIGHT = ignis_weight
    
    logger.info(f"Model weights adjusted: Apollo={apollo_weight:.2f}, Ignis={ignis_weight:.2f}")
    
    return {
        "message": "Weights adjusted successfully", 
        "weights": {
            "apollo": apollo_weight,
            "ignis": ignis_weight
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=config.GAIA_API_PORT, reload=True) 
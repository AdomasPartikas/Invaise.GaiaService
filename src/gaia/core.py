import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import requests
import asyncio

import config
from data_fetcher import DataFetcher
from .heat_combiner import HeatCombiner
from .portfolio_optimizer import PortfolioOptimizer
from .portfolio_analyzer import PortfolioAnalyzer

# Get logger
logger = logging.getLogger("gaia")

class GaiaModel:
    """
    Gaia is an ensemble model that combines predictions from Apollo (historical data) 
    and Ignis (real-time data) models to optimize investment portfolio decisions.
    
    This implementation uses a weighted averaging approach rather than RL since we 
    don't have sufficient data for RL training yet.
    """
    
    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize the Gaia model.
        
        Args:
            data_fetcher: Data fetcher instance for fetching predictions
        """
        self.data_fetcher = data_fetcher
        self.heat_combiner = HeatCombiner()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.portfolio_analyzer = PortfolioAnalyzer(data_fetcher)
        logger.info("Initialized GaiaModel")
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze a symbol using both Apollo and Ignis models.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Analysis results
        """
        # Fetch predictions concurrently
        apollo_data, ignis_data = self.data_fetcher.fetch_all_predictions(symbol)
        
        # Log raw predictions for debugging
        logger.debug(f"Apollo prediction for {symbol}: {json.dumps(apollo_data)}")
        logger.debug(f"Ignis prediction for {symbol}: {json.dumps(ignis_data)}")
        
        # Combine heat scores
        combined_heat = self.heat_combiner.combine_heat_scores(apollo_data, ignis_data)
        
        # Log combined heat
        logger.info(f"Combined heat for {symbol}: {combined_heat}")
        
        return {
            "symbol": symbol,
            "apollo_prediction": apollo_data,
            "ignis_prediction": ignis_data,
            "combined_prediction": combined_heat,
            "timestamp": datetime.now().isoformat()
        }

    def combine_heat_scores(self, apollo_data: Dict[str, Any], ignis_data: Dict[str, Any], portfolio_strategy: str = None) -> Dict[str, Any]:
        """
        Combine heat scores from Apollo and Ignis models.
        
        Args:
            apollo_data: Apollo prediction data (1-month horizon)
            ignis_data: Ignis prediction data (30-minute horizon)
            portfolio_strategy: User's portfolio strategy to adjust model weights
            
        Returns:
            Combined heat score and explanation
        """
        return self.heat_combiner.combine_heat_scores(apollo_data, ignis_data, portfolio_strategy)

    def optimize_portfolio(self, portfolio_id: str, combined_heat: Dict[str, Any], portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a portfolio based on the combined heat score and portfolio data.
        
        Args:
            portfolio_id: Portfolio ID
            combined_heat: Combined heat score
            portfolio_data: Portfolio data
            
        Returns:
            Portfolio optimization recommendations
        """
        return self.portfolio_optimizer.optimize_portfolio(portfolio_id, combined_heat, portfolio_data, self.data_fetcher)
    
    def get_available_market_data_symbols(self) -> List[str]:
        """
        Get all unique symbols available in the historical market data.
        
        Returns:
            List of available symbols
        """
        return self.portfolio_analyzer.get_available_market_data_symbols()
    
    async def analyze_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Analyze a portfolio and generate optimization recommendations.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio analysis results
        """
        return await self.portfolio_analyzer.analyze_portfolio(portfolio_id, self)
    
    async def fetch_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch data for a specific symbol from all models.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Complete symbol data
        """
        # First, try to refresh predictions to ensure fresh data
        _ = self.data_fetcher.refresh_predictions(symbol)
        
        # Then analyze the symbol
        return await self.analyze_symbol(symbol)
    
    async def fetch_batch_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their data
        """
        # Try to refresh predictions for all symbols
        _ = self.data_fetcher.refresh_batch_predictions(symbols)
        
        # Analyze each symbol
        tasks = [self.analyze_symbol(symbol) for symbol in symbols]
        analyses = await asyncio.gather(*tasks)
        
        # Create a dictionary mapping symbols to their analyses
        return {analysis["symbol"]: analysis for analysis in analyses}
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get the status of all models, including Gaia.
        
        Returns:
            Model status information
        """
        return {
            "apollo": {"status": "operational"},
            "ignis": {"status": "operational"},
            "gaia": {"status": "operational"},
            "timestamp": datetime.now().isoformat()
        } 
import json
import logging
import requests
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import config
from auth_service import AuthService

logger = logging.getLogger("gaia.data_fetcher")

class DataFetcher:
    """
    Fetches prediction data from Apollo and Ignis models via the BusinessDomain API.
    """
    
    def __init__(self, base_url: str = config.ASPNET_URL):
        """
        Initialize the data fetcher.
        
        Args:
            base_url: Base URL for the BusinessDomain API
        """
        self.base_url = base_url
        self.auth_service = AuthService()
        logger.info(f"Initialized DataFetcher with base_url={base_url}")
    
    def fetch_predictions(self, symbol: str, model_source: str) -> Optional[Dict[str, Any]]:
        """
        Fetch predictions for a specific symbol and model source.
        
        Args:
            symbol: Stock symbol to get predictions for
            model_source: Model source (e.g., "Apollo", "Ignis", "Gaia")
            
        Returns:
            Prediction dictionary or None if request failed
        """
        try:
            # Use the correct endpoint format for ModelPredictionController
            endpoint = f"{config.PREDICTION_ENDPOINT}/{symbol}/{model_source}"
            
            logger.info(f"Attempting to fetch predictions from {endpoint}")
            
            # Get auth headers
            headers = self.auth_service.get_auth_headers()
            
            # Make the request with auth headers
            response = requests.get(endpoint, headers=headers, timeout=10)
            logger.info(f"Response status from {endpoint}: {response.status_code}")
            
            # If direct endpoint fails, try the "all" endpoint as fallback
            if response.status_code != 200:
                logger.warning(f"Direct endpoint failed with status {response.status_code}, trying 'all' endpoint for {symbol}")
                
                all_endpoint = f"{config.ALL_PREDICTIONS_ENDPOINT.format(symbol=symbol)}"
                all_response = requests.get(all_endpoint, headers=headers, timeout=10)
                logger.info(f"All predictions response status: {all_response.status_code}")
                
                if all_response.status_code == 200:
                    # If successful, extract the specific model source we need
                    all_data = all_response.json()
                    if model_source in all_data:
                        logger.info(f"Found {model_source} prediction in 'all' endpoint response")
                        data = all_data[model_source]
                        standardized_data = self._standardize_prediction_data(data, model_source)
                        
                        # For Gaia, also extract Apollo and Ignis predictions if available
                        if model_source == "Gaia" and "Apollo" in all_data and "Ignis" in all_data:
                            logger.info("Including Apollo and Ignis data in Gaia prediction")
                            standardized_data["apollo_prediction"] = self._standardize_prediction_data(all_data["Apollo"], "Apollo")
                            standardized_data["ignis_prediction"] = self._standardize_prediction_data(all_data["Ignis"], "Ignis")
                            
                        return standardized_data
            
            # If still failing, return None to indicate no data
            if response.status_code != 200:
                logger.error(f"Failed to fetch predictions for {symbol} from {model_source}: No data available")
                return None
                
            # Process response if successful
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully fetched predictions for {symbol} from {model_source}")
            
            # Transform the data into a standard format
            standardized_data = self._standardize_prediction_data(data, model_source)
            
            # For Gaia predictions, also try to fetch Apollo and Ignis data
            if model_source == "Gaia":
                # Try to fetch Apollo and Ignis predictions separately
                apollo_data = self.fetch_predictions(symbol, "Apollo")
                ignis_data = self.fetch_predictions(symbol, "Ignis")
                
                if apollo_data:
                    standardized_data["apollo_prediction"] = apollo_data
                if ignis_data:
                    standardized_data["ignis_prediction"] = ignis_data
            
            return standardized_data
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch predictions for {symbol} from {model_source}: {str(e)}")
            return None
    
    def _standardize_prediction_data(self, data: Dict[str, Any], model_source: str) -> Dict[str, Any]:
        """
        Standardize prediction data from BusinessDomain API into a consistent format.
        
        Args:
            data: Raw prediction data from the API (already standardized by BusinessDomain)
            model_source: Model source (e.g., "Apollo", "Ignis")
            
        Returns:
            Standardized prediction data dictionary
        """
        logger.info(f"Processing {model_source} data: {json.dumps(data)[:200]}...")
        
        try:
            # Default values
            heat_score = 0.0
            confidence = 0.0
            direction = "neutral"
            explanation = "No explanation provided"
            
            # Initialize additional fields
            prediction_id = None
            model_version = None
            prediction_target = None
            current_price = None
            predicted_price = None
            
            # Extract main prediction data
            if 'id' in data:
                prediction_id = data['id']
            if 'modelVersion' in data:
                model_version = data['modelVersion']
            if 'predictionTarget' in data:
                prediction_target = data['predictionTarget']
            if 'currentPrice' in data:
                current_price = data['currentPrice']
            if 'predictedPrice' in data:
                predicted_price = data['predictedPrice']
            
            # Extract data from the standard heat object
            if 'heat' in data and isinstance(data['heat'], dict):
                heat_object = data['heat']
                logger.info(f"Found heat object: {json.dumps(heat_object)}")
                
                # Extract heat score
                if 'heatScore' in heat_object:
                    heat_score = float(heat_object['heatScore'])
                elif 'score' in heat_object:
                    # Score is on a 0-100 scale
                    score = float(heat_object['score'])
                    heat_score = score / 100.0
                
                # Extract confidence
                if 'confidence' in heat_object:
                    confidence_value = float(heat_object['confidence'])
                    # Normalize confidence to 0-1 scale if it's on a 0-100 scale
                    confidence = confidence_value / 100.0 if confidence_value > 1 else confidence_value
                
                # Extract explanation
                if 'explanation' in heat_object:
                    explanation = heat_object['explanation']
                    
                # Extract direction
                if 'direction' in heat_object:
                    direction = heat_object['direction']
                
                # Extract ID if not found already
                if prediction_id is None and 'id' in heat_object:
                    prediction_id = heat_object['id']
            
            # Create a standardized heat dictionary with enhanced fields
            standardized = {
                "heat_score": heat_score,
                "confidence": confidence,
                "direction": direction,
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
                "source": model_source,
                "explanation": explanation,
                "prediction_id": prediction_id,
                "model_version": model_version,
                "prediction_target": prediction_target,
                "current_price": current_price,
                "predicted_price": predicted_price
            }
            
            logger.info(f"Successfully processed {model_source} data with heat_score: {heat_score}, direction: {direction}")
            return standardized
            
        except Exception as e:
            logger.error(f"Error processing {model_source} data: {str(e)}")
            return None
    
    def fetch_all_predictions(self, symbol: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Fetch predictions from both Apollo and Ignis models.
        
        Args:
            symbol: Stock symbol to get predictions for
            
        Returns:
            Tuple of (apollo_predictions, ignis_predictions), either may be None if data not available
        """
        apollo_data = self.fetch_predictions(symbol, "Apollo")
        ignis_data = self.fetch_predictions(symbol, "Ignis")
        
        return apollo_data, ignis_data

    def fetch_portfolio_stocks(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """
        Fetch portfolio stock data directly from the PortfolioStock API endpoint.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            List of portfolio stocks or empty list if request failed
        """
        try:
            # Construct the endpoint URL
            endpoint = f"{self.base_url}/api/PortfolioStock/portfolio/{portfolio_id}"
            logger.info(f"Fetching portfolio stocks from: {endpoint}")
            
            # Get auth headers
            headers = self.auth_service.get_auth_headers()
            
            # Make request with auth headers
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            if response.status_code == 200:
                stocks_data = response.json()
                logger.info(f"Successfully fetched {len(stocks_data)} portfolio stocks for portfolio {portfolio_id}")
                
                # Process stock data to deduplicate and extract essential information
                processed_stocks = []
                seen_symbols = set()
                
                for stock in stocks_data:
                    symbol = stock.get("symbol")
                    if symbol and symbol not in seen_symbols:
                        # Extract portfolio info from the first stock if available
                        if "portfolio" in stock:
                            portfolio_info = stock.get("portfolio", {})
                        
                        # Extract only the stock data we need
                        processed_stock = {
                            "symbol": symbol,
                            "quantity": stock.get("quantity", 0),
                            "currentTotalValue": stock.get("currentTotalValue", 0),
                            "totalBaseValue": stock.get("totalBaseValue", 0),
                            "percentageChange": stock.get("percentageChange", 0),
                            "lastUpdated": stock.get("lastUpdated")
                        }
                        
                        processed_stocks.append(processed_stock)
                        seen_symbols.add(symbol)
                
                return processed_stocks
            else:
                # If no stocks found, log the error and return empty list
                logger.error(f"No portfolio stocks found for portfolio_id {portfolio_id}: Status {response.status_code}")
                return []
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch portfolio stocks for portfolio_id {portfolio_id}: {str(e)}")
            return []

    def fetch_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch portfolio data for a specific portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio data or None if request failed
        """
        try:
            # Fetch portfolio data using portfolio_id
            endpoint = config.PORTFOLIO_ENDPOINT.format(portfolio_id=portfolio_id)
            logger.info(f"Fetching portfolio data from: {endpoint}")
            
            # Get auth headers
            headers = self.auth_service.get_auth_headers()
            
            # Make request with auth headers
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched portfolio data for portfolio {portfolio_id}")
                
                # Verify portfolio strategy if exists
                strategy = data.get("strategyDescription")
                if strategy and strategy not in ["Conservative", "Balanced", "Growth", "Aggressive"]:
                    logger.warning(f"Unknown portfolio strategy: {strategy}, defaulting to Balanced")
                    data["strategyDescription"] = "Balanced"
                elif not strategy:
                    logger.warning(f"No portfolio strategy found, defaulting to Balanced")
                    data["strategyDescription"] = "Balanced"
                
                # If portfolioStocks field is empty or not present, fetch directly from the API
                if not data.get("portfolioStocks") or len(data.get("portfolioStocks", [])) == 0:
                    portfolio_stocks = self.fetch_portfolio_stocks(portfolio_id)
                    if portfolio_stocks:
                        logger.info(f"Adding {len(portfolio_stocks)} stocks to portfolio data")
                        data["portfolioStocks"] = portfolio_stocks
                    
                return data
            else:
                # If no portfolio data, log the error and return None
                logger.error(f"No portfolio found for portfolio_id {portfolio_id}: Status {response.status_code}")
                return None
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch portfolio data for portfolio_id {portfolio_id}: {str(e)}")
            return None
    
    def refresh_predictions(self, symbol: str) -> bool:
        """
        Request BusinessDomain to refresh predictions for a symbol.
        
        Args:
            symbol: Stock symbol to refresh predictions for
            
        Returns:
            True if refresh was successful, False otherwise
        """
        try:
            # Use the refresh predictions endpoint
            endpoint = f"{config.REFRESH_PREDICTION_ENDPOINT.format(symbol=symbol)}"
            logger.info(f"Refreshing predictions for {symbol} from: {endpoint}")
            
            # Get auth headers
            headers = self.auth_service.get_auth_headers()
            
            # Make request with auth headers
            response = requests.post(endpoint, headers=headers, timeout=30)  # Allow longer timeout for refreshing
            
            if response.status_code == 200:
                logger.info(f"Successfully refreshed predictions for {symbol}")
                return True
            else:
                logger.warning(f"Failed to refresh predictions for {symbol}: Status {response.status_code}")
                return False
        
        except requests.RequestException as e:
            logger.error(f"Failed to refresh predictions for {symbol}: {str(e)}")
            return False
            
    def refresh_batch_predictions(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Refresh predictions for multiple symbols.
        
        Args:
            symbols: List of stock symbols to refresh
            
        Returns:
            Dictionary mapping symbols to refresh status (True if successful)
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.refresh_predictions(symbol)
            
        successful = sum(1 for status in results.values() if status)
        logger.info(f"Refreshed predictions for {successful}/{len(symbols)} symbols")
        
        return results
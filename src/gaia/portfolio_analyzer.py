import logging
import json
import requests
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

import config

# Get logger
logger = logging.getLogger("gaia")

class PortfolioAnalyzer:
    """
    Handles portfolio analysis including fetching portfolio data, historical returns,
    and generating comprehensive analysis results.
    """
    
    def __init__(self, data_fetcher):
        """
        Initialize the PortfolioAnalyzer.
        
        Args:
            data_fetcher: Data fetcher instance for API calls
        """
        self.data_fetcher = data_fetcher
    
    def get_available_market_data_symbols(self) -> List[str]:
        """
        Get all unique symbols available in the historical market data.
        
        Returns:
            List of available symbols
        """
        try:
            # Call the API endpoint to get all unique symbols
            endpoint = f"{config.ASPNET_URL}/api/MarketData/GetAllUniqueSymbols"
            
            # Get auth headers if auth_service is available
            headers = {}
            if hasattr(self.data_fetcher, 'auth_service'):
                headers = self.data_fetcher.auth_service.get_auth_headers()
            
            # Make the request
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch available symbols: Status {response.status_code}")
                return []
            
            # Process the response
            symbols = response.json()
            
            if isinstance(symbols, list):
                logger.info(f"Found {len(symbols)} available symbols in market data")
                # Log a sample of symbols for debugging
                if symbols:
                    logger.info(f"Sample symbols: {', '.join(symbols[:10])}")
                return symbols
            else:
                logger.error(f"Unexpected response format for available symbols: {type(symbols)}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching available market data symbols: {str(e)}")
            return []
    
    async def analyze_portfolio(self, portfolio_id: str, gaia_model) -> Dict[str, Any]:
        """
        Analyze a portfolio and generate optimization recommendations.
        
        Args:
            portfolio_id: Portfolio ID
            gaia_model: Reference to the GaiaModel instance for symbol analysis
            
        Returns:
            Portfolio analysis results
        """
        try:
            # Fetch portfolio stocks directly from the API first
            portfolio_stocks = []
            user_id = "unknown"
            portfolio_strategy = "Balanced"
            
            try:
                # Fetch portfolio stocks from API
                endpoint = f"{config.ASPNET_URL}/api/PortfolioStock/portfolio/{portfolio_id}"
                headers = {}
                if hasattr(self.data_fetcher, 'auth_service'):
                    headers = self.data_fetcher.auth_service.get_auth_headers()
                
                response = requests.get(endpoint, headers=headers, timeout=5)
                if response.status_code == 200:
                    api_portfolio_stocks = response.json()
                    if api_portfolio_stocks:
                        # Extract just the stock data and portfolio info
                        seen_symbols = set()
                        for stock in api_portfolio_stocks:
                            symbol = stock.get("symbol")
                            if symbol and symbol not in seen_symbols:
                                portfolio_stocks.append({
                                    "symbol": symbol,
                                    "quantity": stock.get("quantity", 0),
                                    "currentTotalValue": stock.get("currentTotalValue", 0),
                                    "totalBaseValue": stock.get("totalBaseValue", 0),
                                    "percentageChange": stock.get("percentageChange", 0)
                                })
                                seen_symbols.add(symbol)
                            
                            # Extract portfolio information from the first stock
                            if not user_id or user_id == "unknown":
                                portfolio = stock.get("portfolio", {})
                                if portfolio:
                                    user_id = portfolio.get("userId", user_id)
                                    portfolio_strategy = portfolio.get("strategyDescription", portfolio_strategy)
                            
                        logger.info(f"Fetched {len(portfolio_stocks)} portfolio stocks from API for portfolio {portfolio_id}")
            except Exception as e:
                logger.error(f"Error fetching portfolio stocks from API: {str(e)}")
            
            # If we couldn't get stocks from API, try alternative methods
            if not portfolio_stocks:
                # Fetch portfolio data using data_fetcher
                portfolio_data = self.data_fetcher.fetch_portfolio_data(portfolio_id)
                
                if not portfolio_data:
                    logger.error(f"No portfolio data available for portfolio {portfolio_id}")
                    return {
                        "portfolio_id": portfolio_id,
                        "status": "error",
                        "error": "No portfolio data available",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Extract user_id from portfolio data
                user_id = portfolio_data.get("userId", portfolio_data.get("user_id", user_id))
                portfolio_strategy = portfolio_data.get("strategyDescription", portfolio_strategy)
                
                # Log portfolio data for debugging
                logger.info(f"Portfolio data for portfolio {portfolio_id}: {json.dumps(portfolio_data)[:200]}...")
                
                # Extract asset symbols from different possible structures
                portfolio_stocks = portfolio_data.get("portfolioStocks", [])
                if not portfolio_stocks and "portfolio_assets" in portfolio_data:
                    portfolio_stocks = portfolio_data.get("portfolio_assets", [])
            
            # Make sure we have at least some portfolio data
            if not portfolio_stocks:
                logger.warning(f"No stocks found in portfolio for portfolio {portfolio_id}")
                return {
                    "portfolio_id": portfolio_id,
                    "status": "warning",
                    "warning": "No symbols found in portfolio",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract symbols from portfolio stocks
            symbols = []
            for stock in portfolio_stocks:
                symbol = stock.get("symbol", stock.get("Symbol", ""))
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            
            logger.info(f"Symbols to analyze: {symbols}")
            
            if not symbols:
                logger.warning(f"No symbols found in portfolio for portfolio {portfolio_id}")
                return {
                    "portfolio_id": portfolio_id,
                    "status": "warning",
                    "warning": "No symbols found in portfolio",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Fetch market index data for comparison (e.g., S&P 500)
            market_returns = []
            try:
                market_symbol = "SPY"  # Use SPY as a proxy for the market
                market_returns = self.fetch_historical_returns(market_symbol, days=90, portfolio_id=portfolio_id)
                logger.info(f"Fetched {len(market_returns)} days of market returns for comparison")
            except Exception as e:
                logger.error(f"Error fetching market returns: {str(e)}")
            
            # Get historical returns for each symbol and combine for portfolio
            symbols_historical_returns = {}
            all_historical_returns = []
            weighted_returns = []
            total_portfolio_value = sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks)
            
            for symbol in symbols:
                returns = self.fetch_historical_returns(symbol, days=90, portfolio_id=portfolio_id)
                symbols_historical_returns[symbol] = returns
                if returns:
                    all_historical_returns.extend(returns)
                    
                    # Calculate weighted returns based on portfolio allocation
                    for stock in portfolio_stocks:
                        if stock.get("symbol") == symbol:
                            stock_value = stock.get("currentTotalValue", 0)
                            if total_portfolio_value > 0:
                                weight = stock_value / total_portfolio_value
                                weighted_stock_returns = [r * weight for r in returns]
                                if len(weighted_returns) < len(weighted_stock_returns):
                                    weighted_returns = weighted_stock_returns
                                else:
                                    # Add weighted returns (assuming same time periods)
                                    weighted_returns = [a + b for a, b in zip(weighted_returns, weighted_stock_returns[:len(weighted_returns)])]
                            break
            
            # Create portfolio data structure for optimization
            portfolio_data = {
                "userId": user_id,
                "portfolioId": portfolio_id,  # Add portfolio_id for tracking
                "strategyDescription": portfolio_strategy,
                "portfolioStocks": portfolio_stocks,
                "historical_returns": weighted_returns if weighted_returns else all_historical_returns,
                "market_returns": market_returns
            }
            
            # Add portfolio weight information for better debugging
            if portfolio_stocks and total_portfolio_value > 0:
                weight_info = []
                for stock in portfolio_stocks:
                    symbol = stock.get("symbol", "unknown")
                    value = stock.get("currentTotalValue", 0)
                    weight = value / total_portfolio_value if total_portfolio_value > 0 else 0
                    weight_info.append(f"{symbol}:{weight:.2f}")
                logger.info(f"Portfolio weights: {', '.join(weight_info)}")
            
            # Check if we have existing Gaia predictions in the database
            symbol_analyses = {}
            gaia_predictions = {}
            
            try:
                # Fetch existing Gaia predictions for all symbols
                for symbol in symbols:
                    # Try to get Gaia prediction from database
                    gaia_prediction = self.data_fetcher.fetch_predictions(symbol, "Gaia")
                    
                    if gaia_prediction and isinstance(gaia_prediction, dict):
                        # If we have a Gaia prediction, use it
                        logger.info(f"Using existing Gaia prediction for {symbol}")
                        gaia_predictions[symbol] = gaia_prediction
                    else:
                        # If no Gaia prediction exists, we'll need to analyze this symbol
                        logger.info(f"No existing Gaia prediction found for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching Gaia predictions: {str(e)}")
            
            # Check if we have Gaia predictions for all symbols
            if len(gaia_predictions) == len(symbols):
                logger.info(f"Using existing Gaia predictions for all {len(symbols)} symbols")
                
                # Create the symbol analyses from existing predictions
                for symbol, prediction in gaia_predictions.items():
                    # Extract Apollo and Ignis predictions if they're stored in the Gaia prediction
                    apollo_data = prediction.get("apollo_prediction", None)
                    ignis_data = prediction.get("ignis_prediction", None)
                    
                    # Use the prediction directly
                    symbol_analyses[symbol] = {
                        "symbol": symbol,
                        "apollo_prediction": apollo_data,
                        "ignis_prediction": ignis_data,
                        "combined_prediction": prediction,
                        "timestamp": prediction.get("timestamp", datetime.now().isoformat())
                    }
            else:
                # If we don't have Gaia predictions for all symbols, analyze them
                logger.info(f"Analyzing {len(symbols) - len(gaia_predictions)} symbols with missing Gaia predictions")
                
                # For symbols with existing predictions, add them to symbol_analyses
                for symbol, prediction in gaia_predictions.items():
                    apollo_data = prediction.get("apollo_prediction", None)
                    ignis_data = prediction.get("ignis_prediction", None)
                    
                    symbol_analyses[symbol] = {
                        "symbol": symbol,
                        "apollo_prediction": apollo_data,
                        "ignis_prediction": ignis_data,
                        "combined_prediction": prediction,
                        "timestamp": prediction.get("timestamp", datetime.now().isoformat())
                    }
                
                # Process remaining symbols concurrently
                analyze_tasks = []
                for symbol in symbols:
                    if symbol not in gaia_predictions:
                        analyze_tasks.append(gaia_model.analyze_symbol(symbol))
                
                if analyze_tasks:
                    # Wait for all analyses to complete
                    symbol_analyses_list = await asyncio.gather(*analyze_tasks)
                    
                    # Create dictionary of analyses by symbol
                    for analysis in symbol_analyses_list:
                        symbol = analysis.get("symbol")
                        if symbol:
                            symbol_analyses[symbol] = analysis
            
            # Create combined heat map for the entire portfolio
            portfolio_heat = self._create_portfolio_heat(symbol_analyses, portfolio_strategy, symbols)
            
            # Generate portfolio optimization
            optimization = gaia_model.optimize_portfolio(portfolio_id, portfolio_heat, portfolio_data)
            optimizations = [optimization]
            
            # Create portfolio analysis result with relevant information
            result = {
                "portfolio_id": portfolio_id,
                "status": "success",
                "portfolio_summary": {
                    "total_value": sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks),
                    "cash": portfolio_data.get("cash", 0.0),
                    "strategy": portfolio_strategy,
                    "asset_count": len(symbols),
                    "symbols": symbols
                },
                "symbol_analyses": symbol_analyses,
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully analyzed portfolio for portfolio {portfolio_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__} at line {e.__traceback__.tb_lineno}")
            return {
                "portfolio_id": portfolio_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_portfolio_heat(self, symbol_analyses: Dict[str, Dict[str, Any]], 
                             portfolio_strategy: str, symbols: List[str]) -> Dict[str, Any]:
        """Create combined heat map for the entire portfolio."""
        portfolio_heat = {
            "heat_score": 0.0,
            "direction": "neutral",
            "confidence": 0.0,
            "explanation": "",
            "portfolio_strategy": portfolio_strategy,
            "symbols_processed": symbols
        }
        
        # Combine heat scores from all symbols
        total_heat = 0.0
        heat_count = 0
        total_confidence = 0.0
        bull_count = 0
        bear_count = 0
        
        for symbol, analysis in symbol_analyses.items():
            combined_prediction = analysis.get("combined_prediction", {})
            if combined_prediction:
                # Get heat score and direction
                heat = combined_prediction.get("heat_score", 0.0)
                direction = combined_prediction.get("direction", "neutral")
                confidence = combined_prediction.get("confidence", 0.0)
                
                if heat > 0:
                    total_heat += heat
                    total_confidence += confidence
                    heat_count += 1
                    
                    if direction == "up":
                        bull_count += 1
                    elif direction == "down":
                        bear_count += 1
        
        # Calculate average heat and confidence
        if heat_count > 0:
            avg_heat = total_heat / heat_count
            avg_confidence = total_confidence / heat_count
            
            # Determine overall direction
            if bull_count > bear_count:
                portfolio_direction = "up"
            elif bear_count > bull_count:
                portfolio_direction = "down"
            else:
                portfolio_direction = "neutral"
            
            # Update portfolio heat map
            portfolio_heat["heat_score"] = avg_heat
            portfolio_heat["direction"] = portfolio_direction
            portfolio_heat["confidence"] = avg_confidence
            portfolio_heat["explanation"] = f"Portfolio analysis based on {heat_count} symbols shows {portfolio_direction} trend with {avg_heat:.2f} heat score"
        
        return portfolio_heat
    
    def fetch_historical_returns(self, symbol: str, days: int = 30, portfolio_id: str = None) -> List[float]:
        """
        Fetch actual historical returns for a symbol from BusinessDomain API.
        
        Args:
            symbol: Stock symbol to fetch returns for
            days: Number of days of returns to fetch
            portfolio_id: Optional portfolio ID for tracking
            
        Returns:
            List of historical daily returns
        """
        try:
            id_context = f" for portfolio {portfolio_id}" if portfolio_id else ""
            
            # First try to use GetLatestHistoricalMarketDataWithCount endpoint
            endpoint = f"{config.ASPNET_URL}/api/MarketData/GetLatestHistoricalMarketDataWithCount"
            params = {
                "symbol": symbol,
                "count": days + 1  # Need +1 day to calculate returns
            }
            
            logger.info(f"Fetching historical returns for {symbol}{id_context} using count-based endpoint")
            
            # Get auth headers if auth_service is available
            headers = {}
            if hasattr(self.data_fetcher, 'auth_service'):
                headers = self.data_fetcher.auth_service.get_auth_headers()
            
            # Make the request
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            
            # If count-based request fails, fall back to date range request
            if response.status_code != 200 or not response.json():
                logger.warning(f"Count-based fetch failed for {symbol}, trying date range")
                
                # Fall back to GetHistoricalMarketData with date range
                endpoint = f"{config.ASPNET_URL}/api/MarketData/GetHistoricalMarketData"
                params = {
                    "symbol": symbol,
                    "start": (datetime.now() - timedelta(days=days*2)).strftime("%Y-%m-%d"),  # Request more days to ensure we get enough
                    "end": datetime.now().strftime("%Y-%m-%d")
                }
                
                response = requests.get(endpoint, params=params, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch historical data for {symbol}: Status {response.status_code}")
                    return []
            
            # Process the data
            data = response.json()
            
            # If no data returned
            if not data:
                logger.warning(f"No historical data available for {symbol}{id_context}")
                return []
            
            # Extract close prices (handle both array and single object responses)
            close_prices = []
            if isinstance(data, list):
                # Sort by date to ensure correct order
                sorted_data = sorted(data, key=lambda x: x.get("date", ""), reverse=True)
                close_prices = [entry.get("close", 0) for entry in sorted_data if "close" in entry]
            else:
                # Single object response
                close_prices = [data.get("close", 0)] if "close" in data else []
            
            # Need at least 2 price points to calculate returns
            if len(close_prices) < 2:
                logger.warning(f"Insufficient historical data for {symbol}{id_context} ({len(close_prices)} points)")
                return []
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(close_prices)):
                if close_prices[i-1] > 0:
                    daily_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                    returns.append(daily_return)
                else:
                    # Skip this data point if previous close is zero or negative
                    continue
            
            # If we have at least some returns, filter out extreme outliers
            if len(returns) >= 5:
                # Convert to numpy array
                returns_array = np.array(returns)
                
                # Calculate mean and std dev
                mean_return = np.mean(returns_array)
                std = np.std(returns_array)
                
                # Filter out returns that are more than 3 standard deviations from the mean
                filtered_returns = [r for r in returns if abs(r - mean_return) <= 3 * std]
                
                # Only use filtered returns if we didn't lose too many points
                if len(filtered_returns) >= len(returns) * 0.8:
                    returns = filtered_returns
            
            logger.info(f"Fetched {len(returns)} days of historical returns for {symbol}{id_context}")
            
            # Add summary statistics for better debugging
            if returns:
                mean = sum(returns) / len(returns)
                annualized_return = mean * 252
                volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
                logger.info(f"Return statistics for {symbol}: daily mean={mean:.6f}, annualized={annualized_return:.4f}, volatility={volatility:.4f}")
            
            return returns
            
        except requests.RequestException as e:
            logger.error(f"Request error fetching historical returns for {symbol}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error fetching historical returns for {symbol}: {str(e)}")
            return [] 
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
        combined_heat = self.combine_heat_scores(apollo_data, ignis_data)
        
        # Log combined heat
        logger.info(f"Combined heat for {symbol}: {combined_heat}")
        
        return {
            "symbol": symbol,
            "apollo_prediction": apollo_data,
            "ignis_prediction": ignis_data,
            "combined_prediction": combined_heat,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_heat_score(self, apollo_heat: float, ignis_heat: float, direction: str) -> float:
        """
        Calculate a heat score even when both models return neutral signals.
        
        Args:
            apollo_heat: Apollo heat score
            ignis_heat: Ignis heat score
            direction: Combined direction
            
        Returns:
            Calculated heat score
        """
        # If we already have a heat score from the models, use the average
        if apollo_heat > 0 or ignis_heat > 0:
            return (apollo_heat + ignis_heat) / 2
        
        # For neutral signals, still calculate a minimal heat based on the presence of predictions
        # This ensures we always have a heat score even when direction is neutral
        if apollo_heat == 0 and ignis_heat == 0:
            # If both are zero, return a minimal heat score to indicate we have predictions
            # but they're not suggesting any action
            return 0.1 if direction == "neutral" else 0.0
        
        return 0.0

    def _determine_horizon_weights(self, portfolio_strategy: str) -> Dict[str, float]:
        """
        Determine appropriate model weights based on portfolio strategy.
        
        Different strategies should have different time horizons:
        - Conservative: Focus on long-term stability (favor Apollo)
        - Balanced: Equal weighting of short and long-term
        - Growth: Moderate focus on long-term growth (slightly favor Apollo)
        - Aggressive: More focus on short-term opportunities (favor Ignis)
        
        Args:
            portfolio_strategy: The portfolio strategy (Conservative, Balanced, Growth, Aggressive)
            
        Returns:
            Dictionary with apollo_weight and ignis_weight
        """
        # Default weights from config
        apollo_base = config.DEFAULT_APOLLO_WEIGHT
        ignis_base = config.DEFAULT_IGNIS_WEIGHT
        
        # Adjust based on strategy
        if portfolio_strategy == "Conservative":
            # Conservative strategy focuses on long-term stability
            # Strong preference for Apollo (long-term predictions)
            apollo_weight = 0.8
            ignis_weight = 0.2
        elif portfolio_strategy == "Growth":
            # Growth focuses on long-term growth
            # Moderate preference for Apollo
            apollo_weight = 0.7
            ignis_weight = 0.3
        elif portfolio_strategy == "Balanced":
            # Balanced considers both equally
            apollo_weight = 0.5
            ignis_weight = 0.5
        elif portfolio_strategy == "Aggressive":
            # Aggressive takes advantage of short-term opportunities
            # Strong preference for Ignis (short-term predictions)
            apollo_weight = 0.3
            ignis_weight = 0.7
        else:
            # Default to config weights
            apollo_weight = apollo_base
            ignis_weight = ignis_base
            
        logger.info(f"Strategy-based weights for {portfolio_strategy}: Apollo={apollo_weight:.2f}, Ignis={ignis_weight:.2f}")
        
        return {
            "apollo_weight": apollo_weight,
            "ignis_weight": ignis_weight
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
        try:
            # Default values
            combined_heat_score = 0.0
            combined_direction = "neutral"
            combined_confidence = 0.0
            apollo_contribution = 0.0
            ignis_contribution = 0.0
            explanations = []
            
            # Extract heat scores, confidences, and directions
            apollo_heat = float(apollo_data.get("heat_score", 0.0)) if apollo_data else 0.0
            apollo_confidence = float(apollo_data.get("confidence", 0.0)) if apollo_data else 0.0
            apollo_direction = apollo_data.get("direction", "neutral") if apollo_data else "neutral"
            apollo_current_price = apollo_data.get("current_price") if apollo_data else None
            apollo_predicted_price = apollo_data.get("predicted_price") if apollo_data else None
            
            ignis_heat = float(ignis_data.get("heat_score", 0.0)) if ignis_data else 0.0
            ignis_confidence = float(ignis_data.get("confidence", 0.0)) if ignis_data else 0.0
            ignis_direction = ignis_data.get("direction", "neutral") if ignis_data else "neutral"
            ignis_current_price = ignis_data.get("current_price") if ignis_data else None
            ignis_predicted_price = ignis_data.get("predicted_price") if ignis_data else None
            
            # Log extracted values
            logger.info(f"Apollo heat: {apollo_heat}, confidence: {apollo_confidence}, direction: {apollo_direction}")
            logger.info(f"Ignis heat: {ignis_heat}, confidence: {ignis_confidence}, direction: {ignis_direction}")
            
            # If both models are missing, return a neutral result
            if not apollo_data and not ignis_data:
                logger.warning("Both Apollo and Ignis data are missing")
                return {
                    "heat_score": 0.0,
                    "direction": "neutral",
                    "confidence": 0.0,
                    "apollo_contribution": 0.0,
                    "ignis_contribution": 0.0,
                    "explanation": "No data available from either model.",
                    "timestamp": datetime.now().isoformat(),
                    "source": "GAIA",
                    "prediction_id": None,
                    "model_version": config.MODEL_VERSION,
                    "prediction_target": None,
                    "current_price": None,
                    "predicted_price": None
                }
            
            # Extract additional fields from prediction data for enhanced output
            prediction_id = (apollo_data.get("prediction_id") if apollo_data else None) or (ignis_data.get("prediction_id") if ignis_data else None)
            
            # Set the prediction target based on which model has higher weight from strategy
            # This determines what timeframe we're optimizing for
            current_price = apollo_current_price or ignis_current_price
            
            # Calculate model contributions based on confidence and portfolio strategy
            if portfolio_strategy:
                # Adjust weights based on portfolio strategy
                weights = self._determine_horizon_weights(portfolio_strategy)
                apollo_weight = weights["apollo_weight"]
                ignis_weight = weights["ignis_weight"]
                
                # Set prediction target based on dominant model
                if apollo_weight >= ignis_weight:
                    # Use Apollo's prediction target (1 month horizon)
                    prediction_target = apollo_data.get("prediction_target") if apollo_data else None
                else:
                    # Use Ignis's prediction target (30 minute horizon)
                    prediction_target = ignis_data.get("prediction_target") if ignis_data else None
            else:
                # Use default weights from config
                apollo_weight = config.DEFAULT_APOLLO_WEIGHT
                ignis_weight = config.DEFAULT_IGNIS_WEIGHT
                prediction_target = (apollo_data.get("prediction_target") if apollo_data else None) or (ignis_data.get("prediction_target") if ignis_data else None)
            
            # Calculate weighted average of predicted prices if available
            predicted_price = None
            if apollo_predicted_price is not None and ignis_predicted_price is not None:
                # Use strategy-weighted average
                predicted_price = (apollo_predicted_price * apollo_weight + 
                                 ignis_predicted_price * ignis_weight)
            elif apollo_predicted_price is not None:
                predicted_price = apollo_predicted_price
            elif ignis_predicted_price is not None:
                predicted_price = ignis_predicted_price
            
            # Adjust weights based on specific conditions
            apollo_weight_adjusted = apollo_weight
            ignis_weight_adjusted = ignis_weight
            
            # Increase Ignis weight for very recent data (real-time market conditions)
            # This adjustment is independent of portfolio strategy
            if ignis_data and "prediction_target" in ignis_data:
                try:
                    # Fix datetime handling to ensure consistent timezone awareness
                    ignis_target_str = ignis_data["prediction_target"].replace('Z', '+00:00')
                    ignis_target = datetime.fromisoformat(ignis_target_str)
                    
                    # Ensure now has timezone info if the prediction target does
                    if ignis_target.tzinfo is not None:
                        now = datetime.now().astimezone()
                    else:
                        now = datetime.now()
                        
                    # Calculate time difference in hours
                    if ignis_target.tzinfo is not None and now.tzinfo is not None:
                        hours_diff = (ignis_target - now).total_seconds() / 3600
                    else:
                        # Skip time-based adjustment if there's a timezone mismatch
                        hours_diff = 24  # Default to no adjustment
                    
                    # If prediction target is within 1 hour, further increase Ignis weight
                    if 0 <= hours_diff <= 1:
                        boost_factor = max(0, (1 - hours_diff)) * 0.2  # Up to 20% boost
                        ignis_weight_adjusted += boost_factor
                        apollo_weight_adjusted -= boost_factor
                        logger.info(f"Boosted Ignis weight by {boost_factor:.2f} due to very short-term prediction")
                except Exception as e:
                    logger.error(f"Error parsing prediction target: {str(e)}")
                    # Continue without time-based adjustment
            
            # Re-normalize weights
            total_adjusted = apollo_weight_adjusted + ignis_weight_adjusted
            apollo_weight_adjusted /= total_adjusted
            ignis_weight_adjusted /= total_adjusted
            
            # Calculate raw contributions (weighted by confidence)
            apollo_raw_contribution = apollo_heat * apollo_confidence * apollo_weight_adjusted
            ignis_raw_contribution = ignis_heat * ignis_confidence * ignis_weight_adjusted
            
            # Set the non-directional contributions for the response
            apollo_contribution = apollo_raw_contribution
            ignis_contribution = ignis_raw_contribution
            
            # Log contributions
            logger.info(f"Apollo weight: {apollo_weight_adjusted:.2f}, raw contribution: {apollo_raw_contribution:.4f}")
            logger.info(f"Ignis weight: {ignis_weight_adjusted:.2f}, raw contribution: {ignis_raw_contribution:.4f}")
            
            # Calculate total contribution
            total_contribution = apollo_raw_contribution + ignis_raw_contribution
            
            # Determine direction based on weighted voting
            # If directions match, use that direction
            # If directions conflict, use the direction of the model with higher contribution
            if apollo_direction == ignis_direction and apollo_direction != "neutral":
                combined_direction = apollo_direction
            elif apollo_direction == "neutral" and ignis_direction != "neutral":
                combined_direction = ignis_direction
            elif ignis_direction == "neutral" and apollo_direction != "neutral":
                combined_direction = apollo_direction
            elif apollo_direction != ignis_direction and apollo_direction != "neutral" and ignis_direction != "neutral":
                # Directions conflict - use the direction of the model with higher contribution
                combined_direction = apollo_direction if apollo_raw_contribution > ignis_raw_contribution else ignis_direction
            else:
                combined_direction = "neutral"
            
            # If both models have meaningful contributions, calculate combined heat and confidence
            if total_contribution > 0:
                # Calculate final heat score
                if combined_direction == "neutral":
                    combined_heat_score = 0.0
                else:
                    combined_heat_score = total_contribution / (apollo_confidence * apollo_weight_adjusted + ignis_confidence * ignis_weight_adjusted)
                    # Cap heat score at 0.95 to avoid overconfidence
                    combined_heat_score = min(combined_heat_score, 0.95)
            
                # Calculate confidence based on model agreement and confidence levels
                direction_agreement = 1.0 if apollo_direction == ignis_direction else 0.3
                model_confidences = (apollo_confidence + ignis_confidence) / 2
                combined_confidence = model_confidences * direction_agreement
                # Ensure confidence is between 0 and 1
                combined_confidence = min(max(combined_confidence, 0.0), 1.0)
            else:
                # If no meaningful contribution, set neutral values
                combined_heat_score = 0.0
                combined_direction = "neutral"
                combined_confidence = 0.0
            
            # Generate detailed explanation including strategy information
            explanation = self._generate_explanation(
                apollo_heat, apollo_confidence, apollo_direction,
                ignis_heat, ignis_confidence, ignis_direction,
                combined_heat_score, combined_direction
            )
            
            # Add strategy information to explanation if provided
            if portfolio_strategy:
                strategy_explanation = f"\nPortfolio strategy: {portfolio_strategy} "
                if portfolio_strategy == "Conservative":
                    strategy_explanation += "(favoring long-term predictions)"
                elif portfolio_strategy == "Balanced":
                    strategy_explanation += "(balanced between short and long-term)"
                elif portfolio_strategy == "Growth":
                    strategy_explanation += "(moderate focus on long-term growth)"
                elif portfolio_strategy == "Aggressive":
                    strategy_explanation += "(favoring short-term opportunities)"
                explanation = explanation + strategy_explanation
            
            # Return combined heat data
            return {
                "heat_score": combined_heat_score,
                "direction": combined_direction,
                "confidence": combined_confidence,
                "apollo_contribution": apollo_contribution,
                "ignis_contribution": ignis_contribution,
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
                "source": "GAIA",
                "prediction_id": prediction_id,
                "model_version": config.MODEL_VERSION,
                "prediction_target": prediction_target,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "portfolio_strategy": portfolio_strategy
            }
            
        except Exception as e:
            logger.error(f"Error combining heat scores: {str(e)}")
            
            # Return a safe fallback
            return {
                "heat_score": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
                "apollo_contribution": 0.0,
                "ignis_contribution": 0.0,
                "explanation": f"Error combining heat scores: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "source": "GAIA",
                "prediction_id": None,
                "model_version": config.MODEL_VERSION,
                "prediction_target": None,
                "current_price": None,
                "predicted_price": None
            }
    
    def _generate_explanation(self, apollo_heat: float, apollo_confidence: float, apollo_direction: str,
                             ignis_heat: float, ignis_confidence: float, ignis_direction: str,
                             combined_heat: float, combined_direction: str) -> str:
        """
        Generate a human-readable explanation of the combined heat score.
        
        Args:
            apollo_heat: Apollo heat score
            apollo_confidence: Apollo confidence
            apollo_direction: Apollo direction
            ignis_heat: Ignis heat score
            ignis_confidence: Ignis confidence
            ignis_direction: Ignis direction
            combined_heat: Combined heat score
            combined_direction: Combined direction
            
        Returns:
            Explanation string
        """
        # Direction text
        apollo_dir_text = "bullish" if apollo_direction == "up" else "bearish" if apollo_direction == "down" else "neutral"
        ignis_dir_text = "bullish" if ignis_direction == "up" else "bearish" if ignis_direction == "down" else "neutral"
        combined_dir_text = "bullish" if combined_direction == "up" else "bearish" if combined_direction == "down" else "neutral"
        
        # Signal strength text
        apollo_strength = "no" if apollo_heat == 0 else "weak" if apollo_heat < 0.3 else "moderate" if apollo_heat < 0.6 else "strong"
        ignis_strength = "no" if ignis_heat == 0 else "weak" if ignis_heat < 0.3 else "moderate" if ignis_heat < 0.6 else "strong"
        combined_strength = "no" if combined_heat == 0 else "weak" if combined_heat < 0.3 else "moderate" if combined_heat < 0.6 else "strong"
        
        # Create explanation
        explanation = f"Combined prediction: {combined_strength} {combined_dir_text} signal ({combined_heat:.2f})\n\n"
        
        # Add model contributions
        explanation += f"- Apollo: {apollo_strength} {apollo_dir_text} signal ({apollo_heat:.2f})"
        if apollo_heat > 0:
            explanation += f" with {apollo_confidence:.0%} confidence"
        explanation += "\n"
        
        explanation += f"- Ignis: {ignis_strength} {ignis_dir_text} signal ({ignis_heat:.2f})"
        if ignis_heat > 0:
            explanation += f" with {ignis_confidence:.0%} confidence"
        
        return explanation
    
    def _calculate_sharpe_ratio(self, portfolio_data: Dict[str, Any], risk_free_rate: float = 0.02) -> float:
        """
        Calculate the Sharpe ratio for a portfolio.
        
        Args:
            portfolio_data: Portfolio data
            risk_free_rate: Risk-free rate (default: 2%)
            
        Returns:
            Sharpe ratio
        """
        try:
            # Extract portfolio returns if available
            returns = portfolio_data.get("historical_returns", [])
            portfolio_id = portfolio_data.get("portfolioId", "unknown")
            
            # If no historical returns, use actual percentage changes from stocks
            if not returns or len(returns) < 10:
                portfolio_stocks = portfolio_data.get("portfolioStocks", [])
                
                if portfolio_stocks:
                    # Calculate portfolio return from actual percentage changes
                    total_value = sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks)
                    
                    if total_value > 0:
                        # Calculate weighted return and collect stock data for logging
                        weighted_return = 0
                        stock_summary = []
                        
                        for stock in portfolio_stocks:
                            symbol = stock.get("symbol", "unknown")
                            current_value = stock.get("currentTotalValue", 0)
                            percent_change = stock.get("percentageChange", 0) / 100  # Convert to decimal
                            
                            stock_summary.append(f"{symbol}:{percent_change:.4f}")
                            
                            if current_value > 0:
                                weight = current_value / total_value
                                weighted_return += percent_change * weight
                        
                        logger.info(f"Portfolio {portfolio_id} stocks: {', '.join(stock_summary[:5])}")
                        
                        # Convert to annualized return (assuming percentageChange is over ~3 months)
                        annualized_return = weighted_return * 4
                        
                        # Use reasonable volatility estimate based on portfolio diversification and strategy
                        strategy = portfolio_data.get("strategyDescription", "Balanced")
                        stock_count = len(portfolio_stocks)
                        
                        # More diversified portfolios (more stocks) have lower volatility
                        diversification_factor = max(0.7, 1.0 - (stock_count / 20))  # Reduces volatility by up to 30%
                        
                        # Base volatility on strategy
                        if strategy == "Conservative":
                            base_volatility = 0.10  # 10% volatility
                        elif strategy == "Balanced":
                            base_volatility = 0.14  # 14% volatility
                        elif strategy == "Growth":
                            base_volatility = 0.18  # 18% volatility
                        elif strategy == "Aggressive":
                            base_volatility = 0.22  # 22% volatility
                        else:
                            base_volatility = 0.14  # Default: 14% volatility
                        
                        # Apply diversification adjustment
                        annualized_std_dev = base_volatility * diversification_factor
                        
                        # Calculate Sharpe ratio
                        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev
                        
                        logger.info(f"Portfolio {portfolio_id} Sharpe calculation from percentage changes: return={annualized_return:.4f}, stddev={annualized_std_dev:.4f}, sharpe={sharpe_ratio:.4f}")
                        
                        # Cap to reasonable values
                        sharpe_ratio = max(min(sharpe_ratio, 4.0), -3.0)
                        
                        return sharpe_ratio
            
            # If we still don't have sufficient data, log this fact
            if not returns or len(returns) < 10:  # Still need sufficient data points
                logger.warning(f"Insufficient data for Sharpe ratio calculation for portfolio {portfolio_id}")
                
                # Log the portfolio stocks for context
                stocks = portfolio_data.get("portfolioStocks", [])
                if stocks:
                    logger.warning(f"Portfolio has {len(stocks)} stocks but insufficient return data")
                
                # Use a default Sharpe ratio based on the portfolio strategy
                strategy = portfolio_data.get("strategyDescription", "Balanced")
                if strategy == "Conservative":
                    return 0.3  # Lower Sharpe for conservative portfolios
                elif strategy == "Balanced":
                    return 0.5  # Moderate Sharpe for balanced portfolios
                elif strategy == "Growth":
                    return 0.7  # Higher Sharpe for growth portfolios
                elif strategy == "Aggressive":
                    return 0.8  # Highest Sharpe for aggressive portfolios
                return 0.5  # Default moderate Sharpe ratio
            
            # Convert to numpy array for calculations
            returns_array = np.array(returns)
            
            # Filter out extreme outliers (beyond 5 std deviations)
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            if std > 0:
                filtered_returns = returns_array[np.abs(returns_array - mean) <= 5 * std]
            else:
                filtered_returns = returns_array
            
            # If we filtered too many points, use original data
            if len(filtered_returns) < len(returns_array) * 0.8:
                filtered_returns = returns_array
            
            # Calculate mean annualized return
            # Assuming daily returns, annualize by multiplying by sqrt(252)
            mean_return = np.mean(filtered_returns) * 252
            
            # Cap extreme mean returns to realistic values, but with wider range
            mean_return = max(min(mean_return, 0.5), -0.4)  # Cap between -40% and 50%
            
            # Calculate annualized standard deviation
            std_dev = np.std(filtered_returns) * np.sqrt(252)
            
            # Ensure minimum volatility to avoid division by very small numbers
            std_dev = max(std_dev, 0.02)  # Minimum 2% volatility (reduced from 5%)
            
            # Calculate Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / std_dev
            
            # Cap extreme values to prevent unrealistic results, but with wider range
            sharpe_ratio = max(min(sharpe_ratio, 4.0), -3.0)
            
            logger.info(f"Portfolio {portfolio_id} Sharpe ratio calculation: mean={mean_return:.4f}, std={std_dev:.4f}, sharpe={sharpe_ratio:.4f}")
            return sharpe_ratio
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio for portfolio {portfolio_data.get('portfolioId', 'unknown')}: {str(e)}")
            # Return modest positive Sharpe ratio as default
            return 0.5

    def _calculate_mean_variance(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate mean-variance metrics for a portfolio.
        
        Args:
            portfolio_data: Portfolio data
            
        Returns:
            Mean-variance metrics
        """
        try:
            # Extract portfolio returns if available
            returns = portfolio_data.get("historical_returns", [])
            portfolio_id = portfolio_data.get("portfolioId", "unknown")
            
            # If no historical returns, use the actual stock percentage changes
            if not returns or len(returns) < 5:
                portfolio_stocks = portfolio_data.get("portfolioStocks", [])
                
                if portfolio_stocks:
                    logger.info(f"Using actual percentage changes for portfolio {portfolio_id} with {len(portfolio_stocks)} stocks")
                    
                    # Calculate the weighted return based on current portfolio weights
                    total_portfolio_value = sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks)
                    if total_portfolio_value > 0:
                        total_weighted_return = 0
                        for stock in portfolio_stocks:
                            current_value = stock.get("currentTotalValue", 0)
                            percentage_change = stock.get("percentageChange", 0) / 100  # Convert to decimal
                            
                            if current_value > 0:
                                weight = current_value / total_portfolio_value
                                total_weighted_return += percentage_change * weight
                        
                        # Convert to annualized metrics (assuming percentageChange is over ~3 months)
                        annualized_return = total_weighted_return * 4  # Annualize quarterly return
                        
                        # Use market-typical volatility for the strategy
                        strategy = portfolio_data.get("strategyDescription", "Balanced")
                        if strategy == "Conservative":
                            annualized_variance = 0.01  # 10% volatility
                        elif strategy == "Balanced":
                            annualized_variance = 0.02  # 14% volatility
                        elif strategy == "Growth":
                            annualized_variance = 0.03  # 17% volatility
                        elif strategy == "Aggressive":
                            annualized_variance = 0.04  # 20% volatility
                        else:
                            annualized_variance = 0.02  # Default: 14% volatility
                        
                        logger.info(f"Calculated from actual returns - portfolio {portfolio_id}: annualized return={annualized_return:.4f}, variance={annualized_variance:.4f}")
                        
                        return {
                            "mean": float(annualized_return),
                            "variance": float(annualized_variance)
                        }
            
            # If we still don't have returns or too few data points
            if not returns or len(returns) < 5:
                logger.warning(f"Insufficient data for mean-variance calculation for portfolio {portfolio_id}")
                
                # Log portfolio data for debugging
                stocks = portfolio_data.get("portfolioStocks", [])
                stock_info = []
                if stocks:
                    for stock in stocks[:3]:
                        symbol = stock.get("symbol", "unknown")
                        change = stock.get("percentageChange", 0)
                        stock_info.append(f"{symbol}:{change:.2f}%")
                
                logger.warning(f"Portfolio data: {', '.join(stock_info) if stock_info else 'No stocks with percentage changes'}")
                
                # Adjust default values based on portfolio strategy
                strategy = portfolio_data.get("strategyDescription", "Balanced")
                
                if strategy == "Conservative":
                    return {"mean": 0.03, "variance": 0.01}  # Low return, low risk
                elif strategy == "Balanced":
                    return {"mean": 0.05, "variance": 0.02}  # Moderate return, moderate risk
                elif strategy == "Growth":
                    return {"mean": 0.07, "variance": 0.03}  # Higher return, higher risk
                elif strategy == "Aggressive":
                    return {"mean": 0.09, "variance": 0.04}  # Highest return, highest risk
                
                # Default to modest positive return with moderate variance
                return {"mean": 0.05, "variance": 0.02}
            
            # Convert to numpy array for calculations
            returns_array = np.array(returns)
            
            # Filter out extreme outliers (beyond 3 std deviations)
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            if std > 0:
                filtered_returns = returns_array[np.abs(returns_array - mean) <= 3 * std]
            else:
                filtered_returns = returns_array
            
            # If filtering removed too many values, use original data
            if len(filtered_returns) < len(returns_array) * 0.8:
                logger.warning(f"Too many outliers removed for portfolio {portfolio_id}, using original data")
                filtered_returns = returns_array
            
            # Calculate mean daily return
            mean_return = np.mean(filtered_returns)
            
            # Calculate variance of daily returns
            variance = np.var(filtered_returns)
            
            # Annualize for better interpretation
            annualized_mean = mean_return * 252
            annualized_variance = variance * 252
            
            # Cap extreme values with wider range
            # Previous cap was too restrictive at -0.25 to 0.35
            annualized_mean = max(min(annualized_mean, 1.0), -0.5)  # Between -50% and 100%
            annualized_variance = max(min(annualized_variance, 0.5), 0.005)  # Between 0.5% and 50%
            
            # Add detailed logging
            logger.info(f"Raw calculated mean return: {mean_return * 252:.4f}, variance: {variance * 252:.4f}")
            logger.info(f"After capping - mean: {annualized_mean:.4f}, variance: {annualized_variance:.4f}")
            
            return {
                "mean": float(annualized_mean),
                "variance": float(annualized_variance)
            }
        except Exception as e:
            logger.error(f"Error calculating mean-variance for portfolio {portfolio_data.get('portfolioId', 'unknown')}: {str(e)}")
            # Default to modest positive return with moderate variance
            return {"mean": 0.05, "variance": 0.02}

    def _calculate_capm(self, portfolio_data: Dict[str, Any], risk_free_rate: float = 0.02, market_return: float = 0.08) -> float:
        """
        Calculate expected return using the Capital Asset Pricing Model (CAPM).
        
        Args:
            portfolio_data: Portfolio data
            risk_free_rate: Risk-free rate (default: 2%)
            market_return: Expected market return (default: 8%)
            
        Returns:
            Expected return based on CAPM
        """
        try:
            # Extract portfolio returns and market returns if available
            portfolio_returns = portfolio_data.get("historical_returns", [])
            market_returns = portfolio_data.get("market_returns", [])
            
            # Log portfolio ID if available for debugging
            portfolio_id = portfolio_data.get("portfolioId", "unknown")
            
            # Log data availability
            logger.info(f"CAPM calculation for portfolio {portfolio_id}: {len(portfolio_returns)} portfolio returns, {len(market_returns)} market returns")
            
            # Calculate beta if we have both portfolio and market returns
            beta = None
            if portfolio_returns and market_returns and len(portfolio_returns) > 10 and len(market_returns) > 10:
                # Use the minimum length of both arrays
                min_length = min(len(portfolio_returns), len(market_returns))
                p_returns = np.array(portfolio_returns[:min_length])
                m_returns = np.array(market_returns[:min_length])
                
                # Calculate covariance between portfolio and market returns
                covariance = np.cov(p_returns, m_returns)[0, 1]
                # Calculate variance of market returns
                market_variance = np.var(m_returns)
                
                # Calculate beta if market variance is non-zero
                if market_variance > 0:
                    beta = covariance / market_variance
                    logger.info(f"Calculated portfolio beta from returns: {beta:.2f}")
            
            # If beta is not calculated, estimate beta from actual stock performance
            if beta is None:
                # Get portfolio stocks
                portfolio_stocks = portfolio_data.get("portfolioStocks", [])
                
                if portfolio_stocks:
                    # Calculate a weighted beta using stock percentage changes relative to S&P 500 average
                    # Assuming S&P 500 returned ~10% annually, or about 2.5% quarterly
                    sp500_quarterly_return = 0.025
                    total_value = sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks)
                    
                    if total_value > 0:
                        weighted_stock_beta = 0
                        stock_info = []
                        
                        for stock in portfolio_stocks:
                            symbol = stock.get("symbol", "unknown")
                            current_value = stock.get("currentTotalValue", 0)
                            percent_change = stock.get("percentageChange", 0)
                            
                            # Skip zero values
                            if current_value <= 0 or sp500_quarterly_return == 0:
                                continue
                                
                            # Calculate stock's beta relative to market
                            # (stock return / market return)
                            stock_beta = (percent_change / 100) / sp500_quarterly_return
                            
                            # Apply weight based on portfolio allocation
                            weight = current_value / total_value
                            weighted_stock_beta += stock_beta * weight
                            
                            stock_info.append(f"{symbol}:{stock_beta:.2f}")
                        
                        if weighted_stock_beta != 0:
                            beta = weighted_stock_beta
                            logger.info(f"Calculated weighted beta from stock performance: {beta:.2f} ({', '.join(stock_info[:3])})")
                
                # If we still don't have beta, use strategy as a proxy
                if beta is None:
                    # Log the portfolio stocks for debugging
                    stock_info = [f"{s.get('symbol')}:{s.get('percentageChange', 0):.2f}%" for s in portfolio_stocks[:3]] if portfolio_stocks else "No stocks"
                    logger.warning(f"No beta calculated from performance data. Using estimate for {stock_info}")
                    
                    # Use strategy as a proxy for beta
                    strategy = portfolio_data.get("strategyDescription", portfolio_data.get("portfolioStrategy", "Balanced"))
                    
                    if strategy == "Conservative":
                        beta = 0.7  # Lower beta for conservative
                    elif strategy == "Balanced":
                        beta = 1.0  # Market beta for balanced
                    elif strategy == "Growth":
                        beta = 1.2  # Higher beta for growth
                    elif strategy == "Aggressive":
                        beta = 1.5  # Highest beta for aggressive
                    else:
                        beta = 1.0  # Default to market beta
                        
                    logger.info(f"Using strategy-based beta of {beta:.2f} for {strategy} strategy")
            
            # Ensure beta is within reasonable bounds
            beta = max(min(beta, 2.5), 0.2)
            
            # Calculate expected return using CAPM formula
            # Expected return = Risk-free rate + Beta * (Market return - Risk-free rate)
            expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
            
            # Reasonable bounds for expected return
            expected_return = max(min(expected_return, 0.25), 0.005)  # Cap between 0.5% and 25%
            
            logger.info(f"Calculated CAPM expected return: {expected_return:.4f} (beta={beta:.2f}, raw={risk_free_rate + beta * (market_return - risk_free_rate):.4f})")
            return expected_return
        except Exception as e:
            logger.error(f"Error calculating CAPM for portfolio {portfolio_data.get('portfolioId', 'unknown')}: {str(e)}")
            
            # Use a default expected return based on the portfolio strategy
            strategy = portfolio_data.get("strategyDescription", "Balanced")
            if strategy == "Conservative":
                return 0.04  # Lower expected return for conservative portfolios
            elif strategy == "Balanced":
                return 0.06  # Moderate expected return for balanced portfolios
            elif strategy == "Growth":
                return 0.08  # Higher expected return for growth portfolios
            elif strategy == "Aggressive":
                return 0.10  # Highest expected return for aggressive portfolios
            return 0.06  # Default to 6% expected return on error

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
        try:
            # Extract user_id from portfolio data
            user_id = portfolio_data.get("userId", portfolio_data.get("user_id", "unknown"))
            
            # Extract relevant information from combined heat score
            heat_score = combined_heat.get("heat_score", 0.0)
            direction = combined_heat.get("direction", "neutral")
            confidence = combined_heat.get("confidence", 0.0)
            portfolio_strategy = combined_heat.get("portfolio_strategy", None)
            symbols_processed = combined_heat.get("symbols_processed", [])
            
            # Extract portfolio information
            strategy = portfolio_data.get("strategyDescription", "Balanced")
            if not portfolio_strategy:
                portfolio_strategy = strategy
                
            cash = portfolio_data.get("cash", 0.0)
            total_value = 0.0
            
            # Try to get portfolio stocks from different possible formats
            portfolio_stocks = portfolio_data.get("portfolioStocks", [])
            
            # If portfolioStocks is empty, check for alternative structures
            if not portfolio_stocks and "portfolio_assets" in portfolio_data:
                portfolio_stocks = portfolio_data.get("portfolio_assets", [])
            
            # If portfolio_stocks is still empty, try to fetch directly from the API
            if not portfolio_stocks:
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
                            # Extract just the stock data without nesting
                            portfolio_stocks = []
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
                            logger.info(f"Fetched {len(portfolio_stocks)} portfolio stocks from API for portfolio {portfolio_id}")
                except Exception as e:
                    logger.error(f"Error fetching portfolio stocks from API: {str(e)}")
            
            # If we still don't have portfolio stocks and we have symbols_processed, create placeholder stocks
            if not portfolio_stocks and symbols_processed:
                logger.warning(f"No portfolio stocks data found, creating placeholders for {len(symbols_processed)} symbols")
                for symbol in symbols_processed:
                    portfolio_stocks.append({
                        "symbol": symbol,
                        "quantity": 1.0,  # Placeholder
                        "currentTotalValue": 100.0,  # Placeholder
                    })
            
            # Calculate total portfolio value including cash
            for stock in portfolio_stocks:
                current_value = stock.get("currentTotalValue", 0.0)
                if current_value == 0.0:
                    # Try alternative field names
                    current_value = stock.get("currentValue", stock.get("value", 0.0))
                total_value += current_value
                
            total_value += cash
            
            # If we couldn't get total value from stocks, use the provided value
            if total_value == 0:
                total_value = portfolio_data.get("totalValue", portfolio_data.get("total_value", 1000.0))  # Default to avoid division by zero
                logger.warning(f"Could not determine portfolio value, using fallback value: {total_value}")
            
            # Determine risk factor and weight change factor based on portfolio strategy
            risk_factor = 0.5  # Default balanced strategy
            max_weight_change = 0.1  # Default 10% max allocation change
            
            if portfolio_strategy == "Conservative":
                risk_factor = 0.3
                max_weight_change = 0.05  # Max 5% change
            elif portfolio_strategy == "Balanced":
                risk_factor = 0.5
                max_weight_change = 0.1  # Max 10% change
            elif portfolio_strategy == "Growth":
                risk_factor = 0.7
                max_weight_change = 0.15  # Max 15% change
            elif portfolio_strategy == "Aggressive":
                risk_factor = 0.8
                max_weight_change = 0.2  # Max 20% change
            
            logger.info(f"Using risk factor {risk_factor} and max weight change {max_weight_change} for {portfolio_strategy} strategy")
            
            # Generate detailed recommendations for each stock
            recommendations = []
            portfolio_summary = []
            
            # Threshold for taking action - adjusted based on strategy
            # More aggressive strategies have lower thresholds
            if portfolio_strategy == "Conservative":
                action_threshold = 0.3  # Require stronger signals for Conservative
            elif portfolio_strategy == "Balanced":
                action_threshold = 0.2  # Moderate threshold for Balanced
            elif portfolio_strategy == "Growth":
                action_threshold = 0.15  # Lower threshold for Growth
            elif portfolio_strategy == "Aggressive":
                action_threshold = 0.1  # Lowest threshold for Aggressive
            else:
                action_threshold = 0.2  # Default
                
            # NEW: Set minimum weight change to prevent pointless small adjustments
            # This is a percentage point threshold (e.g., 0.01 = 1%)
            if portfolio_strategy == "Conservative":
                min_weight_change_threshold = 0.05  # Min 5% weight change for Conservative (was 3%)
            elif portfolio_strategy == "Balanced":
                min_weight_change_threshold = 0.03  # Min 3% weight change for Balanced (was 2%)
            elif portfolio_strategy == "Growth":
                min_weight_change_threshold = 0.025  # Min 2.5% weight change for Growth (was 1.5%)
            elif portfolio_strategy == "Aggressive":
                min_weight_change_threshold = 0.02  # Min 2% weight change for Aggressive (was 1%)
            else:
                min_weight_change_threshold = 0.03  # Default (was 2%)
                
            logger.info(f"Using minimum weight change threshold of {min_weight_change_threshold:.2%} for {portfolio_strategy} strategy")
            
            # NEW: Check if any optimization is needed based on heat score and confidence
            if heat_score < action_threshold:
                logger.info(f"Heat score {heat_score} is below action threshold {action_threshold}. No optimization needed.")
                
                # Calculate portfolio metrics even when not optimizing
                historical_returns = portfolio_data.get("historical_returns", [])
                sharpe_ratio = 0.0
                mean_return = 0.0
                variance = 0.0
                capm_expected_return = 0.0
                
                if historical_returns and len(historical_returns) > 10:
                    # Calculate metrics using historical data
                    sharpe_ratio = self._calculate_sharpe_ratio(portfolio_data)
                    mean_variance = self._calculate_mean_variance(portfolio_data)
                    mean_return = mean_variance.get("mean", 0.0)
                    variance = mean_variance.get("variance", 0.0)
                    capm_expected_return = self._calculate_capm(portfolio_data)
                    logger.info(f"Calculated metrics for no-change optimization - Sharpe: {sharpe_ratio:.4f}, Mean: {mean_return:.4f}, Var: {variance:.4f}, ExpRet: {capm_expected_return:.4f}")
                
                # Return a "no change" optimization instead of making tiny adjustments
                return {
                    "id": str(id),
                    "portfolioId": portfolio_id,
                    "userId": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "explanation": f"No optimization needed - market signal strength ({heat_score:.2f}) below threshold for {portfolio_strategy} strategy.",
                    "confidence": confidence,
                    "riskTolerance": 0.0,
                    "isApplied": False,
                    "modelVersion": config.MODEL_VERSION,
                    "sharpeRatio": round(sharpe_ratio, 4),
                    "meanReturn": round(mean_return, 4),
                    "variance": round(variance, 4),
                    "expectedReturn": round(capm_expected_return, 4),
                    "recommendations": [],  # Empty recommendations = no changes
                    "symbolsProcessed": symbols_processed,
                    "portfolioStrategy": portfolio_strategy
                }
            
            # Get current weights of each stock
            stock_weights = {}
            for stock in portfolio_stocks:
                symbol = stock.get("symbol", "")
                if not symbol:
                    # Try alternative field name
                    symbol = stock.get("Symbol", "")
                
                current_value = stock.get("currentTotalValue", 0.0)
                if current_value == 0.0:
                    # Try alternative field names
                    current_value = stock.get("currentValue", stock.get("value", 0.0))
                
                if symbol and total_value > 0:
                    current_weight = current_value / total_value
                    stock_weights[symbol] = current_weight
            
            # Track weight adjustments to ensure they sum to 100%
            total_target_weight = 0.0
            weight_adjustments = {}
            
            # First pass: calculate initial weight adjustments based on signals
            for stock in portfolio_stocks:
                symbol = stock.get("symbol", "")
                if not symbol:
                    # Try alternative field name
                    symbol = stock.get("Symbol", "")
                    
                if not symbol:
                    logger.warning("Stock missing symbol, skipping")
                    continue
                
                current_weight = stock_weights.get(symbol, 0.0)
                
                # Adjust weights based on heat score and direction
                if direction == "up" and heat_score > action_threshold:
                    # Bullish signal - increase weight
                    weight_change = min(heat_score * risk_factor, max_weight_change)
                    weight_adjustments[symbol] = weight_change
                elif direction == "down" and heat_score > action_threshold:
                    # Bearish signal - decrease weight
                    weight_change = min(heat_score * risk_factor, max_weight_change)
                    
                    # Different strategies have different minimum position sizes
                    reduction_factor = 0.0  # Default for Aggressive
                    if portfolio_strategy == "Conservative":
                        reduction_factor = 0.5  # Reduce at most by 50%
                    elif portfolio_strategy == "Balanced":
                        reduction_factor = 0.3  # Reduce at most by 70% 
                    elif portfolio_strategy == "Growth":
                        reduction_factor = 0.2  # Reduce at most by 80%
                    
                    # Limit the reduction based on strategy
                    min_weight = current_weight * reduction_factor
                    max_reduction = current_weight - min_weight
                    actual_reduction = min(weight_change, max_reduction)
                    
                    weight_adjustments[symbol] = -actual_reduction
                else:
                    # Neutral or below threshold - no change
                    weight_adjustments[symbol] = 0.0
            
            # Calculate total weight change and normalize if needed
            total_weight_change = sum(weight_adjustments.values())
            
            # NEW: Check if any individual change is significant
            has_significant_change = False
            for symbol, adjustment in weight_adjustments.items():
                if abs(adjustment) >= min_weight_change_threshold:
                    has_significant_change = True
                    logger.info(f"Significant weight change found for {symbol}: {adjustment:.4f}")
                    break
                    
            # If no significant individual changes and total change is small, skip optimization
            if not has_significant_change and abs(total_weight_change) < min_weight_change_threshold * len(portfolio_stocks):
                logger.info(f"No significant individual weight changes found. Largest change below {min_weight_change_threshold:.2%}")
                
                # Calculate portfolio metrics even when not optimizing
                historical_returns = portfolio_data.get("historical_returns", [])
                sharpe_ratio = 0.0
                mean_return = 0.0
                variance = 0.0
                capm_expected_return = 0.0
                
                if historical_returns and len(historical_returns) > 10:
                    # Calculate metrics using historical data
                    sharpe_ratio = self._calculate_sharpe_ratio(portfolio_data)
                    mean_variance = self._calculate_mean_variance(portfolio_data)
                    mean_return = mean_variance.get("mean", 0.0)
                    variance = mean_variance.get("variance", 0.0)
                    capm_expected_return = self._calculate_capm(portfolio_data)
                    logger.info(f"Calculated metrics for no-change optimization - Sharpe: {sharpe_ratio:.4f}, Mean: {mean_return:.4f}, Var: {variance:.4f}, ExpRet: {capm_expected_return:.4f}")
                
                # Return a "no change" optimization with metrics
                return {
                    "id": str(id),
                    "portfolioId": portfolio_id,
                    "userId": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "explanation": f"No significant changes needed at this time. Largest weight adjustment below {min_weight_change_threshold:.1%} threshold.",
                    "confidence": confidence,
                    "riskTolerance": 0.0,
                    "isApplied": False,
                    "modelVersion": config.MODEL_VERSION,
                    "sharpeRatio": round(sharpe_ratio, 4),
                    "meanReturn": round(mean_return, 4),
                    "variance": round(variance, 4),
                    "expectedReturn": round(capm_expected_return, 4),
                    "recommendations": [],  # Empty recommendations = no changes
                    "symbolsProcessed": symbols_processed,
                    "portfolioStrategy": portfolio_strategy
                }
            
            # Second pass: adjust weights to ensure they sum to 100%
            for stock in portfolio_stocks:
                symbol = stock.get("symbol", "")
                if not symbol:
                    symbol = stock.get("Symbol", "")
                    
                if not symbol:
                    continue
                    
                current_qty = stock.get("quantity", 0.0)
                if current_qty == 0.0:
                    # Try alternative field names
                    current_qty = stock.get("Quantity", 1.0)  # Default to 1 if not found
                    
                current_value = stock.get("currentTotalValue", 0.0)
                if current_value == 0.0:
                    # Try alternative field names
                    current_value = stock.get("currentValue", stock.get("value", 100.0))
                    
                current_weight = stock_weights.get(symbol, 0.0)
                
                # If total weight change is not zero, adjust proportionally
                if total_weight_change != 0:
                    # For positive total change, reduce positive adjustments proportionally
                    # For negative total change, reduce negative adjustments proportionally
                    adjustment = weight_adjustments.get(symbol, 0.0)
                    
                    # Calculate the target weight based on the adjusted weight change
                    target_weight = current_weight + adjustment
                else:
                    # No overall change needed
                    target_weight = current_weight
                
                total_target_weight += target_weight
                
                # Determine action based on weight change
                if target_weight > current_weight:
                    action = "buy"
                elif target_weight < current_weight:
                    action = "sell"
                else:
                    action = "hold"
                
                # Calculate target quantity based on target weight
                target_qty = current_qty
                if current_value > 0 and current_qty > 0:
                    # Calculate price per share
                    price_per_share = current_value / current_qty
                    if price_per_share > 0:
                        # Calculate new quantity based on target weight
                        target_qty = (target_weight * total_value) / price_per_share
                
                # Generate explanation based on action
                if action == "buy":
                    explanation = f"Increase position based on bullish signal (heat={heat_score:.2f}, confidence={confidence:.2f})"
                elif action == "sell":
                    explanation = f"Decrease position based on bearish signal (heat={heat_score:.2f}, confidence={confidence:.2f})"
                else:
                    explanation = f"Maintain current position (heat={heat_score:.2f}, confidence={confidence:.2f})"
                
                # Add strategy-specific explanation
                if portfolio_strategy:
                    explanation += f" - {portfolio_strategy} strategy"
                
                # Ensure quantities are positive and rounded to 2 decimal places
                current_qty = max(0, round(current_qty, 2))
                target_qty = max(0, round(target_qty, 2))
                
                # Create recommendation
                recommendation = {
                    "symbol": symbol,
                    "action": action,
                    "currentQuantity": current_qty,
                    "targetQuantity": target_qty,
                    "currentWeight": round(current_weight, 4),
                    "targetWeight": round(target_weight, 4),
                    "explanation": explanation
                }
                
                recommendations.append(recommendation)
                logger.info(f"Added recommendation for {symbol}: {action.upper()} - {current_qty} -> {target_qty}")
                
                # Add to portfolio summary
                portfolio_summary.append(f"{symbol}: {action.upper()} - Change weight from {current_weight:.1%} to {target_weight:.1%}")
            
            # Check if total target weight is significantly different from 100%
            if abs(total_target_weight - 1.0) > 0.01:
                logger.warning(f"Total target weight ({total_target_weight:.4f}) is not 100%, normalizing recommendations")
                # Normalize target weights to sum to 100%
                for recommendation in recommendations:
                    recommendation["targetWeight"] = round(recommendation["targetWeight"] / total_target_weight, 4)
                    # Recalculate target quantity based on normalized weight
                    symbol = recommendation["symbol"]
                    for stock in portfolio_stocks:
                        if stock.get("symbol", "") == symbol or stock.get("Symbol", "") == symbol:
                            current_qty = stock.get("quantity", recommendation["currentQuantity"])
                            current_value = stock.get("currentTotalValue", 0.0)
                            if current_value > 0 and current_qty > 0:
                                price_per_share = current_value / current_qty
                                if price_per_share > 0:
                                    normalized_target_qty = (recommendation["targetWeight"] * total_value) / price_per_share
                                    recommendation["targetQuantity"] = max(0, round(normalized_target_qty, 2))
            
            # Calculate portfolio metrics using historical return data
            historical_returns = portfolio_data.get("historical_returns", [])
            sharpe_ratio = 0.0
            mean_return = 0.0
            variance = 0.0
            capm_expected_return = 0.0
            
            # For projected metrics
            projected_sharpe_ratio = 0.0
            projected_mean_return = 0.0
            projected_variance = 0.0
            projected_expected_return = 0.0
            
            if historical_returns:
                # Calculate metrics using historical data with current weights
                sharpe_ratio = self._calculate_sharpe_ratio(portfolio_data)
                mean_variance = self._calculate_mean_variance(portfolio_data)
                mean_return = mean_variance.get("mean", 0.0)
                variance = mean_variance.get("variance", 0.0)
                capm_expected_return = self._calculate_capm(portfolio_data)
                logger.info(f"Calculated metrics - Sharpe: {sharpe_ratio:.4f}, Mean: {mean_return:.4f}, Var: {variance:.4f}, ExpRet: {capm_expected_return:.4f}")
                
                # Calculate projected metrics with the new weights if we have recommendations
                if recommendations:
                    # Create a copy of the portfolio data for projection
                    projected_portfolio = dict(portfolio_data)
                    projected_portfolio_stocks = []
                    
                    # Update the portfolio stocks with new weights/values
                    for stock in portfolio_stocks:
                        for rec in recommendations:
                            if stock.get("symbol") == rec.get("symbol"):
                                # Create a copy of the stock with updated values
                                updated_stock = dict(stock)
                                # Update quantity to target quantity
                                updated_stock["quantity"] = rec.get("targetQuantity", stock.get("quantity"))
                                # Update weight to target weight
                                current_value = stock.get("currentTotalValue", 0.0)
                                if current_value > 0:
                                    # Calculate new value based on target weight
                                    target_weight = rec.get("targetWeight", stock.get("currentWeight", 0.0))
                                    updated_value = target_weight * total_value
                                    updated_stock["currentTotalValue"] = updated_value
                                projected_portfolio_stocks.append(updated_stock)
                                break
                        else:
                            # If no recommendation for this stock, keep it as is
                            projected_portfolio_stocks.append(dict(stock))
                    
                    # Update the portfolio with the projected stocks
                    projected_portfolio["portfolioStocks"] = projected_portfolio_stocks
                    
                    # Calculate projected metrics
                    projected_sharpe_ratio = self._calculate_sharpe_ratio(projected_portfolio)
                    projected_mean_variance = self._calculate_mean_variance(projected_portfolio)
                    projected_mean_return = projected_mean_variance.get("mean", 0.0)
                    projected_variance = projected_mean_variance.get("variance", 0.0)
                    projected_expected_return = self._calculate_capm(projected_portfolio)
                    logger.info(f"Calculated projected metrics - Sharpe: {projected_sharpe_ratio:.4f}, Mean: {projected_mean_return:.4f}, Var: {projected_variance:.4f}, ExpRet: {projected_expected_return:.4f}")
            else:
                logger.warning("No historical returns data available for metrics calculation")
            
            # Generate explanation for the entire portfolio
            explanation = f"Portfolio optimization based on a {direction.upper()} signal with strength {heat_score:.2f} and confidence {confidence:.2f}:\n"
            
            # Add strategy information
            if portfolio_strategy:
                explanation += f"Strategy: {portfolio_strategy} - "
                if portfolio_strategy == "Conservative":
                    explanation += "Focus on capital preservation with minimal risk.\n"
                elif portfolio_strategy == "Balanced":
                    explanation += "Balance between growth and stability.\n"
                elif portfolio_strategy == "Growth":
                    explanation += "Focus on long-term growth with moderate risk.\n"
                elif portfolio_strategy == "Aggressive":
                    explanation += "Maximize returns with higher risk tolerance.\n"
            
            # Add symbol-specific recommendations with correct percentages
            if portfolio_summary:
                # Replace the portfolio summary with correct percentages
                portfolio_summary = []
                for rec in recommendations:
                    symbol = rec["symbol"]
                    action = rec["action"]
                    current_weight = rec["currentWeight"]
                    target_weight = rec["targetWeight"]
                    
                    # Ensure action matches the direction of weight change
                    if target_weight > current_weight:
                        rec["action"] = "buy"
                        action = "BUY"
                    elif target_weight < current_weight:
                        rec["action"] = "sell"
                        action = "SELL"
                    else:
                        rec["action"] = "hold"
                        action = "HOLD"
                    
                    # Update the explanation to be consistent with the action
                    if action == "BUY":
                        rec["explanation"] = f"Increase position based on bullish signal (heat={heat_score:.2f}, confidence={confidence:.2f}) - {portfolio_strategy} strategy"
                    elif action == "SELL":
                        rec["explanation"] = f"Decrease position based on bearish signal (heat={heat_score:.2f}, confidence={confidence:.2f}) - {portfolio_strategy} strategy"
                    else:
                        rec["explanation"] = f"Maintain current position (heat={heat_score:.2f}, confidence={confidence:.2f}) - {portfolio_strategy} strategy"
                    
                    # Format as percentages with correct precision
                    portfolio_summary.append(f"{symbol}: {action} - Change weight from {current_weight:.1%} to {target_weight:.1%}")
                
                explanation += "\nRecommendations:\n" + "\n".join(portfolio_summary)
            
            # Create optimization dictionary with all calculated data
            optimization = {
                "id": f"{portfolio_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "portfolioId": portfolio_id, 
                "userId": user_id,
                "timestamp": datetime.now().isoformat(),
                "explanation": explanation,
                "confidence": confidence,
                "riskTolerance": risk_factor,
                "isApplied": False,
                "modelVersion": config.MODEL_VERSION,
                "sharpeRatio": round(sharpe_ratio, 4),
                "meanReturn": round(mean_return, 4),
                "variance": round(variance, 4),
                "expectedReturn": round(capm_expected_return, 4),
                # Add projected metrics
                "projectedSharpeRatio": round(projected_sharpe_ratio, 4),
                "projectedMeanReturn": round(projected_mean_return, 4),
                "projectedVariance": round(projected_variance, 4),
                "projectedExpectedReturn": round(projected_expected_return, 4),
                "recommendations": recommendations,
                "symbolsProcessed": symbols_processed,
                "portfolioStrategy": portfolio_strategy
            }
            
            logger.info(f"Generated portfolio optimization recommendations for portfolio {portfolio_id}")
            return optimization
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            
            # Create a default optimization with hold recommendations
            default_recommendations = []
            
            # Try to get symbols from combined_heat if available
            symbols_processed = combined_heat.get("symbols_processed", [])
            
            if symbols_processed:
                for symbol in symbols_processed:
                    default_recommendations.append({
                        "symbol": symbol,
                        "action": "hold",
                        "currentQuantity": 1.0,
                        "targetQuantity": 1.0,
                        "currentWeight": 0.0,
                        "targetWeight": 0.0,
                        "explanation": "Error occurred during optimization, recommend holding current position"
                    })
            elif portfolio_data.get("portfolioStocks"):
                for stock in portfolio_data.get("portfolioStocks", []):
                    symbol = stock.get("symbol", "unknown")
                    current_qty = stock.get("quantity", 0.0)
                    default_recommendations.append({
                        "symbol": symbol,
                        "action": "hold",
                        "currentQuantity": current_qty,
                        "targetQuantity": current_qty,
                        "currentWeight": 0.0,
                        "targetWeight": 0.0,
                        "explanation": "Error occurred during optimization, recommend holding current position"
                    })
            
            return {
                "id": f"{portfolio_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "portfolioId": portfolio_id,
                "userId": portfolio_data.get("userId", portfolio_data.get("user_id", "unknown")),
                "timestamp": datetime.now().isoformat(),
                "explanation": f"Error optimizing portfolio: {str(e)}",
                "confidence": 0.0,
                "riskTolerance": 0.5,
                "isApplied": False,
                "modelVersion": config.MODEL_VERSION,
                "sharpeRatio": 0,
                "meanReturn": 0,
                "variance": 0,
                "expectedReturn": 0,
                "recommendations": default_recommendations,
                "error": str(e)
            }
    
    def _format_metrics_for_api(self, sharpe_ratio: float, mean_variance: Dict[str, float], capm_expected_return: float) -> Dict[str, float]:
        """
        Format portfolio metrics for API response.
        
        Args:
            sharpe_ratio: Sharpe ratio value
            mean_variance: Mean-variance dictionary
            capm_expected_return: CAPM expected return
            
        Returns:
            API-compatible metrics dictionary
        """
        return {
            "sharpeRatio": round(sharpe_ratio, 4),
            "meanReturn": round(mean_variance.get("mean", 0.0), 4),
            "variance": round(mean_variance.get("variance", 0.0), 4),
            "expectedReturn": round(capm_expected_return, 4)
        }
    
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
    
    async def analyze_portfolio(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Analyze a portfolio and generate optimization recommendations.
        
        Args:
            portfolio_id: Portfolio ID
            
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
                market_returns = self._fetch_historical_returns(market_symbol, days=90, portfolio_id=portfolio_id)
                logger.info(f"Fetched {len(market_returns)} days of market returns for comparison")
            except Exception as e:
                logger.error(f"Error fetching market returns: {str(e)}")
            
            # Get historical returns for each symbol and combine for portfolio
            symbols_historical_returns = {}
            all_historical_returns = []
            weighted_returns = []
            total_portfolio_value = sum(stock.get("currentTotalValue", 0) for stock in portfolio_stocks)
            
            for symbol in symbols:
                returns = self._fetch_historical_returns(symbol, days=90, portfolio_id=portfolio_id)
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
            
            # NEW: First check if we have existing Gaia predictions in the database
            # Instead of creating new analyses for each symbol, use existing predictions
            symbol_analyses = {}
            optimizations = []
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
                        analyze_tasks.append(self.analyze_symbol(symbol))
                
                if analyze_tasks:
                    # Wait for all analyses to complete
                    symbol_analyses_list = await asyncio.gather(*analyze_tasks)
                    
                    # Create dictionary of analyses by symbol
                    for analysis in symbol_analyses_list:
                        symbol = analysis.get("symbol")
                        if symbol:
                            symbol_analyses[symbol] = analysis
            
            # Create combined heat map for the entire portfolio
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
            
            # Generate portfolio optimization
            optimization = self.optimize_portfolio(portfolio_id, portfolio_heat, portfolio_data)
            optimizations.append(optimization)
            
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
    
    def _fetch_historical_returns(self, symbol: str, days: int = 30, portfolio_id: str = None) -> List[float]:
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
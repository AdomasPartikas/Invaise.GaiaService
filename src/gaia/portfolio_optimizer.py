import logging
import requests
from datetime import datetime
from typing import Dict, List, Any

import config
from .metrics_calculator import MetricsCalculator

# Get logger
logger = logging.getLogger("gaia")

class PortfolioOptimizer:
    """
    Handles portfolio optimization based on combined heat scores and portfolio data.
    """
    
    def __init__(self):
        """Initialize the PortfolioOptimizer."""
        self.metrics_calculator = MetricsCalculator()
    
    def optimize_portfolio(self, portfolio_id: str, combined_heat: Dict[str, Any], portfolio_data: Dict[str, Any], data_fetcher=None) -> Dict[str, Any]:
        """
        Optimize a portfolio based on the combined heat score and portfolio data.
        
        Args:
            portfolio_id: Portfolio ID
            combined_heat: Combined heat score
            portfolio_data: Portfolio data
            data_fetcher: Optional data fetcher for API calls
            
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
            if not portfolio_stocks and data_fetcher:
                portfolio_stocks = self._fetch_portfolio_stocks_from_api(portfolio_id, data_fetcher)
            
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
            strategy_params = self._get_strategy_parameters(portfolio_strategy)
            risk_factor = strategy_params["risk_factor"]
            max_weight_change = strategy_params["max_weight_change"]
            action_threshold = strategy_params["action_threshold"]
            min_weight_change_threshold = strategy_params["min_weight_change_threshold"]
            
            logger.info(f"Using risk factor {risk_factor} and max weight change {max_weight_change} for {portfolio_strategy} strategy")
            logger.info(f"Using minimum weight change threshold of {min_weight_change_threshold:.2%} for {portfolio_strategy} strategy")
            
            # Check if any optimization is needed based on heat score and confidence
            if heat_score < action_threshold:
                return self._create_no_change_optimization(
                    portfolio_id, user_id, heat_score, action_threshold, portfolio_strategy, 
                    confidence, symbols_processed, portfolio_data
                )
            
            # Calculate weight adjustments
            weight_adjustments, stock_weights = self._calculate_weight_adjustments(
                portfolio_stocks, heat_score, direction, action_threshold, risk_factor, 
                max_weight_change, portfolio_strategy, total_value
            )
            
            # Check if any individual change is significant
            has_significant_change = any(
                abs(adjustment) >= min_weight_change_threshold 
                for adjustment in weight_adjustments.values()
            )
            
            total_weight_change = sum(weight_adjustments.values())
            
            # If no significant individual changes and total change is small, skip optimization
            if not has_significant_change and abs(total_weight_change) < min_weight_change_threshold * len(portfolio_stocks):
                return self._create_no_change_optimization(
                    portfolio_id, user_id, heat_score, action_threshold, portfolio_strategy,
                    confidence, symbols_processed, portfolio_data, 
                    f"No significant changes needed at this time. Largest weight adjustment below {min_weight_change_threshold:.1%} threshold."
                )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                portfolio_stocks, stock_weights, weight_adjustments, total_value,
                heat_score, confidence, portfolio_strategy
            )
            
            # Normalize recommendations if needed
            recommendations = self._normalize_recommendations(recommendations, portfolio_stocks, total_value)
            
            # Calculate portfolio metrics
            current_metrics, projected_metrics = self._calculate_portfolio_metrics(
                portfolio_data, recommendations, portfolio_stocks, total_value
            )
            
            # Generate explanation
            explanation = self._generate_portfolio_explanation(
                direction, heat_score, confidence, portfolio_strategy, recommendations
            )
            
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
                **current_metrics,
                **projected_metrics,
                "recommendations": recommendations,
                "symbolsProcessed": symbols_processed,
                "portfolioStrategy": portfolio_strategy
            }
            
            logger.info(f"Generated portfolio optimization recommendations for portfolio {portfolio_id}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return self._create_error_optimization(portfolio_id, portfolio_data, combined_heat, str(e))
    
    def _fetch_portfolio_stocks_from_api(self, portfolio_id: str, data_fetcher) -> List[Dict[str, Any]]:
        """Fetch portfolio stocks from API."""
        try:
            endpoint = f"{config.ASPNET_URL}/api/PortfolioStock/portfolio/{portfolio_id}"
            headers = {}
            if hasattr(data_fetcher, 'auth_service'):
                headers = data_fetcher.auth_service.get_auth_headers()
            
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
                    return portfolio_stocks
        except Exception as e:
            logger.error(f"Error fetching portfolio stocks from API: {str(e)}")
        return []
    
    def _get_strategy_parameters(self, portfolio_strategy: str) -> Dict[str, float]:
        """Get strategy-specific parameters."""
        if portfolio_strategy == "Conservative":
            return {
                "risk_factor": 0.3,
                "max_weight_change": 0.05,  # Max 5% change
                "action_threshold": 0.3,  # Require stronger signals
                "min_weight_change_threshold": 0.05  # Min 5% weight change
            }
        elif portfolio_strategy == "Balanced":
            return {
                "risk_factor": 0.5,
                "max_weight_change": 0.1,  # Max 10% change
                "action_threshold": 0.2,  # Moderate threshold
                "min_weight_change_threshold": 0.03  # Min 3% weight change
            }
        elif portfolio_strategy == "Growth":
            return {
                "risk_factor": 0.7,
                "max_weight_change": 0.15,  # Max 15% change
                "action_threshold": 0.15,  # Lower threshold
                "min_weight_change_threshold": 0.025  # Min 2.5% weight change
            }
        elif portfolio_strategy == "Aggressive":
            return {
                "risk_factor": 0.8,
                "max_weight_change": 0.2,  # Max 20% change
                "action_threshold": 0.1,  # Lowest threshold
                "min_weight_change_threshold": 0.02  # Min 2% weight change
            }
        else:
            return {
                "risk_factor": 0.5,
                "max_weight_change": 0.1,
                "action_threshold": 0.2,
                "min_weight_change_threshold": 0.03
            }
    
    def _create_no_change_optimization(self, portfolio_id: str, user_id: str, heat_score: float, 
                                     action_threshold: float, portfolio_strategy: str, confidence: float,
                                     symbols_processed: List[str], portfolio_data: Dict[str, Any],
                                     custom_explanation: str = None) -> Dict[str, Any]:
        """Create a no-change optimization response."""
        # Calculate portfolio metrics even when not optimizing
        historical_returns = portfolio_data.get("historical_returns", [])
        sharpe_ratio = 0.0
        mean_return = 0.0
        variance = 0.0
        capm_expected_return = 0.0
        
        if historical_returns and len(historical_returns) > 10:
            # Calculate metrics using historical data
            sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(portfolio_data)
            mean_variance = self.metrics_calculator.calculate_mean_variance(portfolio_data)
            mean_return = mean_variance.get("mean", 0.0)
            variance = mean_variance.get("variance", 0.0)
            capm_expected_return = self.metrics_calculator.calculate_capm(portfolio_data)
            logger.info(f"Calculated metrics for no-change optimization - Sharpe: {sharpe_ratio:.4f}, Mean: {mean_return:.4f}, Var: {variance:.4f}, ExpRet: {capm_expected_return:.4f}")
        
        explanation = (custom_explanation or 
                      f"No optimization needed - market signal strength ({heat_score:.2f}) below threshold for {portfolio_strategy} strategy.")
        
        return {
            "id": f"{portfolio_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "portfolioId": portfolio_id,
            "userId": user_id,
            "timestamp": datetime.now().isoformat(),
            "explanation": explanation,
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
    
    def _calculate_weight_adjustments(self, portfolio_stocks: List[Dict[str, Any]], heat_score: float, 
                                    direction: str, action_threshold: float, risk_factor: float,
                                    max_weight_change: float, portfolio_strategy: str, 
                                    total_value: float) -> tuple:
        """Calculate weight adjustments for portfolio stocks."""
        stock_weights = {}
        weight_adjustments = {}
        
        # Calculate current weights
        for stock in portfolio_stocks:
            symbol = stock.get("symbol", "")
            if not symbol:
                symbol = stock.get("Symbol", "")
            
            current_value = stock.get("currentTotalValue", 0.0)
            if current_value == 0.0:
                current_value = stock.get("currentValue", stock.get("value", 0.0))
            
            if symbol and total_value > 0:
                current_weight = current_value / total_value
                stock_weights[symbol] = current_weight
        
        # Calculate weight adjustments based on signals
        for stock in portfolio_stocks:
            symbol = stock.get("symbol", "")
            if not symbol:
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
                reduction_factor = self._get_reduction_factor(portfolio_strategy)
                
                # Limit the reduction based on strategy
                min_weight = current_weight * reduction_factor
                max_reduction = current_weight - min_weight
                actual_reduction = min(weight_change, max_reduction)
                
                weight_adjustments[symbol] = -actual_reduction
            else:
                # Neutral or below threshold - no change
                weight_adjustments[symbol] = 0.0
        
        return weight_adjustments, stock_weights
    
    def _get_reduction_factor(self, portfolio_strategy: str) -> float:
        """Get the reduction factor for a given strategy."""
        if portfolio_strategy == "Conservative":
            return 0.5  # Reduce at most by 50%
        elif portfolio_strategy == "Balanced":
            return 0.3  # Reduce at most by 70% 
        elif portfolio_strategy == "Growth":
            return 0.2  # Reduce at most by 80%
        else:  # Aggressive
            return 0.0  # Can reduce to 0%
    
    def _generate_recommendations(self, portfolio_stocks: List[Dict[str, Any]], stock_weights: Dict[str, float],
                                weight_adjustments: Dict[str, float], total_value: float, heat_score: float,
                                confidence: float, portfolio_strategy: str) -> List[Dict[str, Any]]:
        """Generate portfolio recommendations."""
        recommendations = []
        
        for stock in portfolio_stocks:
            symbol = stock.get("symbol", "")
            if not symbol:
                symbol = stock.get("Symbol", "")
                
            if not symbol:
                continue
                
            current_qty = stock.get("quantity", 0.0)
            if current_qty == 0.0:
                current_qty = stock.get("Quantity", 1.0)  # Default to 1 if not found
                
            current_value = stock.get("currentTotalValue", 0.0)
            if current_value == 0.0:
                current_value = stock.get("currentValue", stock.get("value", 100.0))
                
            current_weight = stock_weights.get(symbol, 0.0)
            adjustment = weight_adjustments.get(symbol, 0.0)
            target_weight = current_weight + adjustment
            
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
                price_per_share = current_value / current_qty
                if price_per_share > 0:
                    target_qty = (target_weight * total_value) / price_per_share
            
            # Generate explanation based on action
            explanation = self._generate_stock_explanation(action, heat_score, confidence, portfolio_strategy)
            
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
        
        return recommendations
    
    def _generate_stock_explanation(self, action: str, heat_score: float, confidence: float, portfolio_strategy: str) -> str:
        """Generate explanation for individual stock action."""
        if action == "buy":
            explanation = f"Increase position based on bullish signal (heat={heat_score:.2f}, confidence={confidence:.2f})"
        elif action == "sell":
            explanation = f"Decrease position based on bearish signal (heat={heat_score:.2f}, confidence={confidence:.2f})"
        else:
            explanation = f"Maintain current position (heat={heat_score:.2f}, confidence={confidence:.2f})"
        
        if portfolio_strategy:
            explanation += f" - {portfolio_strategy} strategy"
        
        return explanation
    
    def _normalize_recommendations(self, recommendations: List[Dict[str, Any]], portfolio_stocks: List[Dict[str, Any]], 
                                 total_value: float) -> List[Dict[str, Any]]:
        """Normalize recommendations to ensure weights sum to 100%."""
        total_target_weight = sum(rec["targetWeight"] for rec in recommendations)
        
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
                        break
        
        return recommendations
    
    def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any], recommendations: List[Dict[str, Any]],
                                   portfolio_stocks: List[Dict[str, Any]], total_value: float) -> tuple:
        """Calculate current and projected portfolio metrics."""
        historical_returns = portfolio_data.get("historical_returns", [])
        
        # Current metrics
        current_metrics = {}
        projected_metrics = {}
        
        if historical_returns:
            # Calculate current metrics
            sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(portfolio_data)
            mean_variance = self.metrics_calculator.calculate_mean_variance(portfolio_data)
            capm_expected_return = self.metrics_calculator.calculate_capm(portfolio_data)
            
            current_metrics = {
                "sharpeRatio": round(sharpe_ratio, 4),
                "meanReturn": round(mean_variance.get("mean", 0.0), 4),
                "variance": round(mean_variance.get("variance", 0.0), 4),
                "expectedReturn": round(capm_expected_return, 4)
            }
            
            # Calculate projected metrics if we have recommendations
            if recommendations:
                projected_portfolio = self._create_projected_portfolio(portfolio_data, recommendations, portfolio_stocks, total_value)
                
                projected_sharpe_ratio = self.metrics_calculator.calculate_sharpe_ratio(projected_portfolio)
                projected_mean_variance = self.metrics_calculator.calculate_mean_variance(projected_portfolio)
                projected_expected_return = self.metrics_calculator.calculate_capm(projected_portfolio)
                
                projected_metrics = {
                    "projectedSharpeRatio": round(projected_sharpe_ratio, 4),
                    "projectedMeanReturn": round(projected_mean_variance.get("mean", 0.0), 4),
                    "projectedVariance": round(projected_mean_variance.get("variance", 0.0), 4),
                    "projectedExpectedReturn": round(projected_expected_return, 4)
                }
                
                logger.info(f"Calculated projected metrics - Sharpe: {projected_sharpe_ratio:.4f}, Mean: {projected_mean_variance.get('mean', 0.0):.4f}")
        else:
            logger.warning("No historical returns data available for metrics calculation")
            current_metrics = {
                "sharpeRatio": 0.0,
                "meanReturn": 0.0,
                "variance": 0.0,
                "expectedReturn": 0.0
            }
            projected_metrics = {
                "projectedSharpeRatio": 0.0,
                "projectedMeanReturn": 0.0,
                "projectedVariance": 0.0,
                "projectedExpectedReturn": 0.0
            }
        
        return current_metrics, projected_metrics
    
    def _create_projected_portfolio(self, portfolio_data: Dict[str, Any], recommendations: List[Dict[str, Any]],
                                  portfolio_stocks: List[Dict[str, Any]], total_value: float) -> Dict[str, Any]:
        """Create a projected portfolio with updated weights/values."""
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
        return projected_portfolio
    
    def _generate_portfolio_explanation(self, direction: str, heat_score: float, confidence: float,
                                      portfolio_strategy: str, recommendations: List[Dict[str, Any]]) -> str:
        """Generate explanation for the entire portfolio optimization."""
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
        
        # Add symbol-specific recommendations
        if recommendations:
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
        
        return explanation
    
    def _create_error_optimization(self, portfolio_id: str, portfolio_data: Dict[str, Any], 
                                 combined_heat: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create an error optimization response."""
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
            "explanation": f"Error optimizing portfolio: {error_message}",
            "confidence": 0.0,
            "riskTolerance": 0.5,
            "isApplied": False,
            "modelVersion": config.MODEL_VERSION,
            "sharpeRatio": 0,
            "meanReturn": 0,
            "variance": 0,
            "expectedReturn": 0,
            "recommendations": default_recommendations,
            "error": error_message
        } 
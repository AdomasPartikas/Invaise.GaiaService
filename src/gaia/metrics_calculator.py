import logging
import numpy as np
from typing import Dict, Any

# Get logger
logger = logging.getLogger("gaia")

class MetricsCalculator:
    """
    Handles calculation of financial metrics for portfolio optimization.
    """
    
    def __init__(self):
        """Initialize the MetricsCalculator."""
        pass
    
    def calculate_sharpe_ratio(self, portfolio_data: Dict[str, Any], risk_free_rate: float = 0.02) -> float:
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

    def calculate_mean_variance(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
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

    def calculate_capm(self, portfolio_data: Dict[str, Any], risk_free_rate: float = 0.02, market_return: float = 0.08) -> float:
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

    def format_metrics_for_api(self, sharpe_ratio: float, mean_variance: Dict[str, float], camp_expected_return: float) -> Dict[str, float]:
        """
        Format portfolio metrics for API response.
        
        Args:
            sharpe_ratio: Sharpe ratio value
            mean_variance: Mean-variance dictionary
            camp_expected_return: CAPM expected return
            
        Returns:
            API-compatible metrics dictionary
        """
        return {
            "sharpeRatio": round(sharpe_ratio, 4),
            "meanReturn": round(mean_variance.get("mean", 0.0), 4),
            "variance": round(mean_variance.get("variance", 0.0), 4),
            "expectedReturn": round(camp_expected_return, 4)
        } 
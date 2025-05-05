import json
from datetime import datetime

from gaia import GaiaModel

# Mock heat data from Apollo (historical data model)
apollo_heat_bullish = {
    "heat_score": 0.8,
    "confidence": 0.75,
    "direction": "up",
    "timestamp": datetime.now().isoformat(),
    "source": "APOLLO",
    "explanation": "Historical data indicates a strong upward trend over the next week",
    "prediction_id": 4708,
    "model_version": "0.1.0",
    "prediction_target": "2025-05-25T00:00:00",
    "current_price": 208.37,
    "predicted_price": 224.93
}

apollo_heat_bearish = {
    "heat_score": 0.7,
    "confidence": 0.8,
    "direction": "down",
    "timestamp": datetime.now().isoformat(),
    "source": "APOLLO",
    "explanation": "Historical data shows a pattern similar to previous market corrections",
    "prediction_id": 4707,
    "model_version": "0.1.0",
    "prediction_target": "2025-05-25T00:00:00",
    "current_price": 208.37,
    "predicted_price": 194.25
}

# Mock heat data from Ignis (real-time model)
ignis_heat_bullish = {
    "heat_score": 0.6,
    "confidence": 0.9,
    "direction": "up",
    "timestamp": datetime.now().isoformat(),
    "source": "IGNIS",
    "explanation": "Real-time data shows positive momentum with increasing volume",
    "prediction_id": 4710,
    "model_version": "0.2.0",
    "prediction_target": None,
    "current_price": 208.37,
    "predicted_price": 220.82
}

ignis_heat_bearish = {
    "heat_score": 0.5,
    "confidence": 0.85,
    "direction": "down",
    "timestamp": datetime.now().isoformat(),
    "source": "IGNIS",
    "explanation": "Real-time indicators show increasing selling pressure",
    "prediction_id": 4709,
    "model_version": "0.2.0",
    "prediction_target": None,
    "current_price": 208.37,
    "predicted_price": 198.53
}

ignis_heat_neutral = {
    "heat_score": 0.3,
    "confidence": 0.6,
    "direction": "neutral",
    "timestamp": datetime.now().isoformat(),
    "source": "IGNIS",
    "explanation": "Real-time data shows mixed signals with no clear direction",
    "prediction_id": 4711,
    "model_version": "0.2.0",
    "prediction_target": None,
    "current_price": 208.37,
    "predicted_price": 209.12
}

# Mock empty data for testing fallback behavior
empty_heat = {}

# Mock portfolio data
portfolio_data = {
    "portfolio_id": "portfolio456",
    "user_id": "test123",
    "total_value": 100000.0,
    "cash": 25000.0,
    "risk_profile": "moderate",
    "assets": [
        {
            "id": "AAPL",
            "name": "Apple Inc.",
            "allocation": 0.15,
            "value": 15000.0,
            "quantity": 75
        },
        {
            "id": "MSFT",
            "name": "Microsoft Corporation",
            "allocation": 0.12,
            "value": 12000.0,
            "quantity": 40
        },
        {
            "id": "GOOGL",
            "name": "Alphabet Inc.",
            "allocation": 0.10,
            "value": 10000.0,
            "quantity": 8
        },
        {
            "id": "AMZN",
            "name": "Amazon.com Inc.",
            "allocation": 0.08,
            "value": 8000.0,
            "quantity": 3
        },
        {
            "id": "TSLA",
            "name": "Tesla, Inc.",
            "allocation": 0.05,
            "value": 5000.0,
            "quantity": 5
        }
    ]
}

def print_section(title):
    """Print a section title with separators for better readability."""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

def main():
    # Initialize Gaia model with custom weights
    gaia = GaiaModel(apollo_weight=0.7, ignis_weight=0.3, risk_tolerance=0.6)
    
    # Print model info
    print_section("Gaia Model Information")
    model_info = gaia.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # Test 1: Both models bullish
    print_section("Test 1: Both Models Bullish")
    combined_heat = gaia.combine_heat_scores(apollo_heat_bullish, ignis_heat_bullish)
    print("Combined Heat Score:")
    print(json.dumps(combined_heat, indent=2))
    
    # Test portfolio optimization with bullish signals
    decision = gaia.optimize_portfolio(portfolio_data["portfolio_id"], combined_heat, portfolio_data)
    print("\nPortfolio Optimization (Bullish):")
    print(json.dumps(decision["portfolio_recommendations"], indent=2))
    
    # Test 2: Both models bearish
    print_section("Test 2: Both Models Bearish")
    combined_heat = gaia.combine_heat_scores(apollo_heat_bearish, ignis_heat_bearish)
    print("Combined Heat Score:")
    print(json.dumps(combined_heat, indent=2))
    
    # Test portfolio optimization with bearish signals
    decision = gaia.optimize_portfolio(portfolio_data["portfolio_id"], combined_heat, portfolio_data)
    print("\nPortfolio Optimization (Bearish):")
    print(json.dumps(decision["portfolio_recommendations"], indent=2))
    
    # Test 3: Apollo bullish, Ignis bearish (conflict)
    print_section("Test 3: Conflicting Signals")
    combined_heat = gaia.combine_heat_scores(apollo_heat_bullish, ignis_heat_bearish)
    print("Combined Heat Score:")
    print(json.dumps(combined_heat, indent=2))
    
    # Test portfolio optimization with conflicting signals
    decision = gaia.optimize_portfolio(portfolio_data["portfolio_id"], combined_heat, portfolio_data)
    print("\nPortfolio Optimization (Conflicting):")
    print(json.dumps(decision["portfolio_recommendations"], indent=2))
    
    # Test 4: Apollo bullish, Ignis neutral
    print_section("Test 4: Apollo Bullish, Ignis Neutral")
    combined_heat = gaia.combine_heat_scores(apollo_heat_bullish, ignis_heat_neutral)
    print("Combined Heat Score:")
    print(json.dumps(combined_heat, indent=2))
    
    # Test portfolio optimization with mixed signals
    decision = gaia.optimize_portfolio(portfolio_data["portfolio_id"], combined_heat, portfolio_data)
    print("\nPortfolio Optimization (Mixed):")
    print(json.dumps(decision["portfolio_recommendations"], indent=2))
    
    # Test 5: Handle empty data
    print_section("Test 5: Handle Empty Data")
    combined_heat = gaia.combine_heat_scores(empty_heat, empty_heat)
    print("Combined Heat Score:")
    print(json.dumps(combined_heat, indent=2))
    
    # Test weight adjustment
    print_section("Test Weight Adjustment")
    print("Original weights - Apollo: {:.2f}, Ignis: {:.2f}".format(
        gaia.apollo_weight, gaia.ignis_weight))
    
    # Adjust weights based on performance
    gaia.adjust_weights(apollo_performance=0.3, ignis_performance=0.7)
    print("Adjusted weights - Apollo: {:.2f}, Ignis: {:.2f}".format(
        gaia.apollo_weight, gaia.ignis_weight))
    
    # Test with adjusted weights
    combined_heat = gaia.combine_heat_scores(apollo_heat_bullish, ignis_heat_bullish)
    print("\nCombined Heat Score with Adjusted Weights:")
    print(json.dumps(combined_heat, indent=2))

if __name__ == "__main__":
    main() 
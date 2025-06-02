import logging
from datetime import datetime
from typing import Dict, Any

import config

# Get logger
logger = logging.getLogger("gaia")

class HeatCombiner:
    """
    Handles the combination of heat scores from Apollo and Ignis models.
    """
    
    def __init__(self):
        """Initialize the HeatCombiner."""
        pass
    
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
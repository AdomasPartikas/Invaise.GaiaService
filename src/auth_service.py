import os
import requests
import logging
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional

import config

logger = logging.getLogger("gaia.auth_service")

class AuthService:
    """
    Service for handling authentication with BusinessDomain API.
    Manages JWT token acquisition and caching.
    """
    
    def __init__(self):
        """Initialize the auth service with credentials from environment variables"""
        self.base_url = config.ASPNET_URL
        self.token = None
        self.token_expiry = None
        
        # Get credentials from environment variables
        self.service_account_id = config.GAIA_SERVICE_ACCOUNT_ID
        self.service_account_key = config.GAIA_SERVICE_ACCOUNT_KEY
        
        logger.info(f"AuthService initialized with base_url={self.base_url}")
        
        if not self.service_account_id or not self.service_account_key:
            logger.warning("Service account credentials not found in environment variables")
        else:
            logger.info(f"Service account credentials found: ID={self.service_account_id[:8]}...")
            
    def get_token(self) -> Optional[str]:
        """
        Get a valid JWT token, refreshing if necessary.
        
        Returns:
            JWT token string or None if authentication failed
        """
        # Check if we have a valid cached token
        logger.info("get_token called, checking for valid cached token")
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            logger.info(f"Using cached JWT token, valid until {self.token_expiry}")
            return self.token
            
        # Otherwise get a new token
        logger.info("No valid cached token, authenticating to get a new token")
        return self._authenticate()
        
    def _authenticate(self) -> Optional[str]:
        """
        Authenticate with BusinessDomain API to get a JWT token.
        
        Returns:
            JWT token string or None if authentication failed
        """
        try:
            # Use the authentication endpoint
            endpoint = f"{self.base_url}/Auth/service/login"
            logger.info(f"Authenticating with BusinessDomain at {endpoint}")
            
            # Prepare credentials payload
            payload = {
                "id": self.service_account_id,
                "key": self.service_account_key
            }
            logger.debug(f"Authentication payload prepared with ID={self.service_account_id[:8]}...")
            
            # Make authentication request
            logger.info("Sending authentication request...")
            response = requests.post(endpoint, json=payload, timeout=10)
            logger.info(f"Authentication response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("token")
                
                logger.info(f"Authentication successful, received response data: {str(data)[:50]}...")
                
                if not self.token:
                    logger.error("No token found in the response data")
                    return None
                
                # Parse token to get expiry time
                try:
                    # Decode token without verification to get expiry
                    decoded = jwt.decode(self.token, options={"verify_signature": False})
                    exp_timestamp = decoded.get("exp", 0)
                    self.token_expiry = datetime.fromtimestamp(exp_timestamp)
                    
                    # Add a safety margin
                    self.token_expiry -= timedelta(minutes=5)
                    
                    logger.info(f"Successfully authenticated with BusinessDomain, token valid until {self.token_expiry}")
                    
                    # Log token for debugging (first 10 chars only)
                    if self.token:
                        token_preview = self.token[:10] + "..." if self.token else "None"
                        logger.info(f"Received token (preview): {token_preview}")
                    
                    return self.token
                except Exception as e:
                    logger.error(f"Error parsing JWT token: {str(e)}")
                    return None
            else:
                logger.error(f"Authentication failed: Status {response.status_code}")
                if response.text:
                    logger.error(f"Error response: {response.text[:200]}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to authenticate with BusinessDomain: {str(e)}")
            return None
            
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers with authorization token.
        
        Returns:
            Dictionary of headers including Authorization
        """
        logger.info("Getting auth headers")
        token = self.get_token()
        if token:
            logger.info("Auth token available, returning Bearer token in headers")
            return {"Authorization": f"Bearer {token}"}
        
        logger.warning("No auth token available, returning empty headers")
        return {} 
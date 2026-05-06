"""
OpenAI Model Adapter for InternAgent

Implements the BaseModel interface for OpenAI models.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

import litellm
from litellm import acompletion, aembedding

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI-compatible implementation using LiteLLM for multi-provider support."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model_name: str = "gpt-4o", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 60,
                base_url: Optional[str] = None,
                fallbacks: Optional[List[str]] = None):
        """
        Initialize the model adapter using LiteLLM.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE_URL")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.fallbacks = fallbacks or []
        
        litellm.telemetry = False
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """Generate text using LiteLLM with automatic fallback."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        models_to_try = [self.model_name] + self.fallbacks
        last_error = None
        
        for model in models_to_try:
            try:
                response = await acompletion(
                    model=model,
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    timeout=self.timeout,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}. Trying next fallback...")
                last_error = e
                
        logger.error(f"All models failed. Last error: {last_error}")
        raise last_error
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nPlease respond ONLY with a valid JSON object that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Please respond ONLY with a valid JSON object that matches this schema: {json.dumps(json_schema)}"

        models_to_try = [self.model_name] + self.fallbacks
        last_error = None
        
        for model in models_to_try:
            try:
                response = await acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature if temperature is not None else self.temperature,
                    timeout=self.timeout,
                    **kwargs
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if result_text.startswith("```"):
                    start_idx = result_text.find("{")
                    end_idx = result_text.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        result_text = result_text[start_idx:end_idx+1]
                
                try:
                    result_dict = json.loads(result_text)
                except json.JSONDecodeError:
                    logger.warning(f"Initial JSON parse failed. Attempting repair...")
                    try:
                        result_text_repair = repair_json(result_text)
                        if result_text_repair:
                            if isinstance(result_text_repair, str):
                                result_dict = json.loads(result_text_repair)
                            else:
                                result_dict = result_text_repair
                        else:
                            raise ValueError("repair_json returned empty")
                    except Exception as e:
                        raise ValueError(f"Repair failed: {e}")
                return result_dict
            
            except Exception as e:
                logger.warning(f"JSON generation with model {model} failed: {e}. Trying next...")
                last_error = e

        logger.error(f"Failed to generate JSON with all models. Last error: {last_error}")
        raise last_error
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using LiteLLM."""
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await aembedding(
                model=kwargs.get("embedding_model", "text-embedding-ada-002"),
                input=text_list,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            embeddings = [item["embedding"] if isinstance(item, dict) else item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIModel':
        """Create model instance from a configuration dictionary."""
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "gpt-4o"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 300),
            base_url=config.get("base_url"),
            fallbacks=config.get("fallbacks")
        ) 

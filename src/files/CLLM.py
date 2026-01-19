from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union
import requests
import os


class CustomLLM(BaseLLM):
    def __init__(
        self,
        model: str = "custom",
        base_url: str = "",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 60,
        max_tokens: int = 4000
    ):
        super().__init__(model=model, temperature=temperature)
        # Handle None or empty base_url by loading from environment
        if not base_url:
            base_url = os.getenv("CUSTOM_LLM_URL", "http://localhost")
        
        # Remove trailing slash and add /chat if not already present
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/chat"):
            base_url = base_url + "/chat"
        
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[str, Any]:

        # Convert messages to a single user input string
        if isinstance(messages, str):
            user_input = messages
        else:
            # Combine all message contents into one string
            user_input = "\n".join(
                [msg.get("content", "") for msg in messages if msg.get("content")]
            )

        payload = {
            "user_input": user_input,
            "max_token": self.max_tokens
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

        except requests.exceptions.RequestException as e:
            return f"[LLM ERROR] {str(e)}"

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

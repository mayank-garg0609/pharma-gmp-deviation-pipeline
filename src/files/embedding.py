
import os
from abc import ABC, abstractmethod
from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[list]:
        pass


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model: str = "custom",
        api_key: str | None = None,
        base_url: str = "",
        max_batch_size: int = 32
    ):
        self.base_url = (base_url or os.getenv("CUSTOM_LLM_URL")).rstrip("/") + "/embed"
        self.api_key = api_key
        self.model = model
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        self.max_batch_size = max_batch_size
        
        # Connection pooling for faster API calls
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def embed(self, texts: List[str]) -> List[list]:
        if not texts:
            return []

        # Process in batches if texts exceed max_batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            try:
                response = self.session.post(
                    self.base_url,
                    json={"texts": batch},  # Send batch of texts
                    timeout=self.timeout * 2
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                
                if len(embeddings) != len(batch):
                    raise Exception(f"[EMBED ERROR] Expected {len(batch)} embeddings, got {len(embeddings)}")
                
                all_embeddings.extend(embeddings)
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"[EMBED ERROR] Batch {i//self.max_batch_size + 1}: {str(e)}")
        
        return all_embeddings

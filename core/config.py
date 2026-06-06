from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    openrouter_api_key: str
    openrouter_base_url: str
    model_name: str
    embedding_model: str
    knowledge_dir: Path
    vector_db_dir: Path
    top_k: int
    timeout_seconds: float
    max_history_messages: int

    @property
    def system_prompt(self) -> str:
        return (
            "Sei l'assistente AI di Engine SpA. Rispondi in modo professionale, consulenziale e sintetico. "
            "Usa solo il contesto recuperato dal database vettoriale. "
            "Se l'informazione non è presente o è ambigua, rispondi: "
            '"Non ho dati sufficienti su questo punto, consulto un esperto di prodotto." '
            "Se mancano dettagli tecnici, chiedi prima numero di corsie e tipo di infrazione. "
            "Quando possibile, cita le fonti in modo chiaro."
        )

    @classmethod
    def from_env(cls) -> "AppConfig":
        project_root = Path(__file__).resolve().parent.parent

        load_dotenv(project_root / ".env")

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY non impostata.")

        return cls(
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            model_name=os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip(),
            knowledge_dir=(project_root / os.getenv("KNOWLEDGE_DIR", "KNOLEDGE")).resolve(),
            vector_db_dir=(project_root / os.getenv("VECTOR_DB_DIR", "db_engine")).resolve(),
            top_k=int(os.getenv("TOP_K", "3")),
            timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "6")),
        )

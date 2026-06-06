from core.config import AppConfig
from core.vector_store import KnowledgeBaseManager


def main() -> None:
    config = AppConfig.from_env()
    manager = KnowledgeBaseManager(config)
    manager.build_index()
    print(f"Database vettoriale creato con successo in: {config.vector_db_dir}")


if __name__ == "__main__":
    main()
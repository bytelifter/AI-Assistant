class BotError(Exception):
    """Base exception for the bot."""


class KnowledgeBaseNotReadyError(BotError):
    """Raised when the vector index is missing or not built."""


class KnowledgeBaseEmptyError(BotError):
    """Raised when the knowledge base does not return relevant documents."""


class ModelRequestError(BotError):
    """Raised when the LLM request fails."""


class ResponseParseError(BotError):
    """Raised when the model response cannot be parsed."""

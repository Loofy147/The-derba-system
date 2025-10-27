import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai") # 'openai', 'anthropic', or 'local'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3") # e.g., llama3, mistral
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434") # Ollama API host

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3)) # Reduced randomness
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 500))   # Prevent long responses
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.9)) # Focus on more relevant answers

    # Tuber Network Settings
    PRUNING_THRESHOLD = float(os.getenv("PRUNING_THRESHOLD", 0.1)) # Threshold for pruning low-value tubers
    MAX_TUBERS = int(os.getenv("MAX_TUBERS", 1000)) # Max number of tubers in the network
    INITIAL_REWARD_PREDICTION = float(os.getenv("INITIAL_REWARD_PREDICTION", 0.5))
    EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", 1000)) # Cache size for embeddings

    # Reward System Settings
    SHORT_TERM_LR = float(os.getenv("SHORT_TERM_LR", 0.2)) # Learning rate for short-term reward
    LONG_TERM_LR = float(os.getenv("LONG_TERM_LR", 0.05)) # Learning rate for long-term reward
    CONFIDENCE_DECAY_RATE = float(os.getenv("CONFIDENCE_DECAY_RATE", 0.01)) # How fast confidence decays
    CONFIDENCE_GROWTH_RATE = float(os.getenv("CONFIDENCE_GROWTH_RATE", 0.05)) # How fast confidence grows
    MAX_PREDICTION_ERROR = float(os.getenv("MAX_PREDICTION_ERROR", 1.0)) # Max error for confidence calculation
    LLM_INSIGHT_ERROR_THRESHOLD = float(os.getenv("LLM_INSIGHT_ERROR_THRESHOLD", 0.3)) # Error threshold to trigger LLM insight

    # Challenge Agent Settings
    EXPLORATION_BUDGET = float(os.getenv("EXPLORATION_BUDGET", 1.0)) # % of resources for challenging paths
    NOVELTY_WEIGHT = float(os.getenv("NOVELTY_WEIGHT", 0.6))
    RISK_WEIGHT = float(os.getenv("RISK_WEIGHT", 0.4))

    # Transportation Settings
    DEFAULT_TRANSPORT_SPEED = float(os.getenv("DEFAULT_TRANSPORT_SPEED", 1.0))
    DEFAULT_TRANSPORT_COST = float(os.getenv("DEFAULT_TRANSPORT_COST", 0.1))
    DEFAULT_TRANSPORT_CAPACITY = float(os.getenv("DEFAULT_TRANSPORT_CAPACITY", 100.0))

    # Regret Settings
    REGRET_PENALTY_WEIGHT = float(os.getenv("REGRET_PENALTY_WEIGHT", 0.2)) # Weight for regret in reward calculation
    NON_ACTION_REGRET_MULTIPLIER = float(os.getenv("NON_ACTION_REGRET_MULTIPLIER", 0.5)) # Multiplier for regret of non-action

    # Discount Offers for LLM calls (simulated)
    DISCOUNT_3_PERCENT_THRESHOLD = int(os.getenv("DISCOUNT_3_PERCENT_THRESHOLD", 100)) # After 100 calls
    DISCOUNT_7_PERCENT_THRESHOLD = int(os.getenv("DISCOUNT_7_PERCENT_THRESHOLD", 500)) # After 500 calls
    DISCOUNT_14_PERCENT_THRESHOLD = int(os.getenv("DISCOUNT_14_PERCENT_THRESHOLD", 1000)) # After 1000 calls

    # Logging and Debugging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FILE = os.getenv("LOG_FILE", "tuber_orchestrator.log")

    # Experimentation Settings
    MAX_EXPERIMENT_PARAMETERS = int(os.getenv("MAX_EXPERIMENT_PARAMETERS", 5))
    MAX_EXPERIMENT_ITERATIONS = int(os.getenv("MAX_EXPERIMENT_ITERATIONS", 20))
    MAX_VALUES_PER_PARAMETER = int(os.getenv("MAX_VALUES_PER_PARAMETER", 5))
    EXPERIMENT_DURATION_LIMIT_MINUTES = int(os.getenv("EXPERIMENT_DURATION_LIMIT_MINUTES", 30))

    # Question Generation Settings
    CURIOSITY_THRESHOLD = float(os.getenv("CURIOSITY_THRESHOLD", 0.8))
    MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", 5))

    # Meta-Cognitive (Gap Detection) Settings
    GAP_CONFIDENCE_THRESHOLD = float(os.getenv("GAP_CONFIDENCE_THRESHOLD", 0.65))
    GAP_ERROR_THRESHOLD = float(os.getenv("GAP_ERROR_THRESHOLD", 0.35))
    GAP_STAGNATION_CYCLES = int(os.getenv("GAP_STAGNATION_CYCLES", 4))
    GAP_LLM_ERROR_THRESHOLD = float(os.getenv("GAP_LLM_ERROR_THRESHOLD", 0.15)) # 15% error rate
    GAP_ISOLATED_TUBER_THRESHOLD = int(os.getenv("GAP_ISOLATED_TUBER_THRESHOLD", 5))
    GAP_CURIOSITY_TRIGGER_RATE = float(os.getenv("GAP_CURIOSITY_TRIGGER_RATE", 0.05)) # 5% chance

    # Ensure required API keys are set
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env file")
    if LLM_PROVIDER == "local" and not LOCAL_MODEL:
        raise ValueError("LOCAL_MODEL not set in .env file for local LLM provider.")

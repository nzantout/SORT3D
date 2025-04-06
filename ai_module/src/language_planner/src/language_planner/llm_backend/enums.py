from enum import Enum

class SystemMode(Enum):
    BENCHMARK = "benchmark"
    LIVE_NAVIGATION = "navigation"

class NavQueryRunMode(Enum):
    DEFAULT = "default"
    USE_TOOL_NOT_GRAPH = "use_tools_no_graph"
    USE_TOOL_USE_GRAPH = "use_tools"
    USE_TOOL_ACTOR_CRITIC_GRAPH = "use_tools_actor_critic"

class ObjectQueryType(Enum):
    LLM_BASED = "llm"
    CLIP_BASED = "clip"

class LanguageModel(Enum):
    GPT4 = "gpt4"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    LLAMA = "llama3.1"
    R1_QWEN2 = "deepseek-r1:7b"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
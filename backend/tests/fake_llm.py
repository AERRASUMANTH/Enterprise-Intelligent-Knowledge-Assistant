"""A stand-in for LLMClient used only in tests (no network, no API key)."""


class FakeLLM:
    def __init__(self, response="", error: Exception | None = None):
        # `response` may be a string, or a function(prompt, system_instruction) -> str.
        self.model = "fake-gemini"
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate(self, prompt: str, system_instruction: str | None = None, usage: dict | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})
        if usage is not None:
            usage["gemini_calls"] = usage.get("gemini_calls", 0) + 1
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + len(prompt) // 4
            usage["output_tokens"] = usage.get("output_tokens", 0) + 10
        if self.error:
            raise self.error
        if callable(self.response):
            return self.response(prompt, system_instruction)
        return self.response

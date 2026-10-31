import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1000,
    output_config={"effort": "low"},
    system="You are a patient teacher who explains things to a 5-year-old.",
    messages=[{"role": "user", "content": "why claude API is different from the claude desktop and does they use same LLM?"}],
)
for block in message.content:
    if block.type == "text":
        print(block.text)
print("input tokens:", message.usage.input_tokens)
print("output tokens:", message.usage.output_tokens)


"""
import json

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=2000,
    output_config={
        "effort": "low",
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["term", "definition", "example"],
                "additionalProperties": False,
            },
        },
    },
    messages=[{"role": "user", "content": "Explain what a token is."}],
)

for block in message.content:
    if block.type == "text":
        data = json.loads(block.text)   # the text block now holds valid JSON
        print(data["definition"])
        print("Example:", data["example"])

"""
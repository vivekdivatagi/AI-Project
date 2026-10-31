"""
Tool use demo: a travel assistant with two tools.

Scenario: a user is planning a trip to Tokyo and asks a question that needs
live-looking data Claude doesn't have on its own:
  - get_weather      -> what's the weather in a city
  - convert_currency -> how much is X in another currency

Claude decides which tools to call, our code runs them, sends the results back,
and Claude writes the final answer. The tool data below is mocked so the demo
runs offline - in a real app these functions would call a weather / FX API.
"""
import json

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

# ---------- 1. Tool implementations (plain Python functions) ----------

WEATHER_DATA = {
    "tokyo": {"temp_c": 18, "condition": "light rain", "humidity": 82},
    "london": {"temp_c": 11, "condition": "cloudy", "humidity": 76},
    "new york": {"temp_c": 15, "condition": "sunny", "humidity": 55},
    "bengaluru": {"temp_c": 27, "condition": "partly cloudy", "humidity": 64},
}

# rates relative to 1 USD
EXCHANGE_RATES = {"USD": 1.0, "JPY": 149.5, "EUR": 0.92, "GBP": 0.79, "INR": 83.2}


def get_weather(city: str, unit: str = "celsius") -> dict:
    data = WEATHER_DATA.get(city.lower())
    if data is None:
        raise ValueError(f"no weather data for '{city}'")
    temp = data["temp_c"] if unit == "celsius" else round(data["temp_c"] * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "unit": unit,
            "condition": data["condition"], "humidity_percent": data["humidity"]}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in EXCHANGE_RATES or dst not in EXCHANGE_RATES:
        raise ValueError(f"unsupported currency pair {src}->{dst}")
    converted = amount / EXCHANGE_RATES[src] * EXCHANGE_RATES[dst]
    return {"amount": amount, "from": src, "to": dst, "converted": round(converted, 2)}


# maps the tool name Claude uses -> the Python function to run
TOOL_FUNCTIONS = {"get_weather": get_weather, "convert_currency": convert_currency}

# ---------- 2. Tool definitions (what Claude sees) ----------

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use this whenever the user asks "
                       "about weather, temperature, or what to pack/wear somewhere.",
        "strict": True,  # guarantees Claude's input matches the schema
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount of money from one currency to another using "
                       "today's exchange rate. Currencies are ISO codes like USD, JPY, INR.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO code, e.g. 'USD'"},
                "to_currency": {"type": "string", "description": "ISO code, e.g. 'JPY'"},
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False,
        },
    },
]

# ---------- 3. The agentic loop ----------

messages = [{
    "role": "user",
    "content": "I'm flying from Bengaluru to Tokyo tomorrow with 50,000 INR. "
               "What's the weather like there, what should I pack, "
               "and how much is my money worth in yen?",
}]

try:
    while True:
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=4000,
            # adaptive = Claude decides when/how much to think; "summarized" returns a
            # readable summary of its reasoning (the default "omitted" returns empty text)
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "medium"},  # higher effort -> more thinking
            system="You are a friendly travel assistant. Use the tools for any weather or "
                   "currency facts - never guess them.",
            tools=TOOLS,
            messages=messages,
        )

        # show what Claude was thinking before it acted
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                print(f"[thinking] {block.thinking}\n")

        # keep Claude's full turn (thinking + text + tool_use blocks) in the history.
        # thinking blocks must be passed back unchanged so Claude keeps its reasoning
        # across tool calls
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break  # Claude is done (end_turn, max_tokens, refusal, ...)

        # Claude may ask for several tools at once - run them all and send
        # every result back in ONE user message
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[tool call] {block.name}({json.dumps(block.input)})")
            try:
                result = TOOL_FUNCTIONS[block.name](**block.input)
                print(f"[tool result] {result}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": json.dumps(result)})
            except Exception as e:
                # tell Claude the tool failed so it can recover, instead of crashing
                print(f"[tool error] {e}")
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": str(e), "is_error": True})

        messages.append({"role": "user", "content": tool_results})
except anthropic.APIStatusError as e:
    print(f"error: API returned {e.status_code} -", e.message)
except anthropic.APIConnectionError:
    print("error: could not reach the API - check your network")
else:
    print("\ncontent:")
    for block in response.content:
        if block.type == "text":
            print(block.text)
    print("stop reason:", response.stop_reason)
    print("usage (last call):")
    print("  input tokens:", response.usage.input_tokens)
    print("  output tokens:", response.usage.output_tokens)

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

try:
    with client.messages.stream(
        model="claude-haiku-5-5",
        max_tokens=1000,
        output_config={"effort": "low"},
        stop_sequences=["desktop"],  # stop generating when any of these appears
        system="You are a patient teacher who explains things to a 5-year-old.",
        messages=[{"role": "user", "content": "why claude API is different from the claude desktop and does they use same LLM?"}],
    ) as stream:
        print("content:")
        for text in stream.text_stream:
            print(text, end="", flush=True)  # print each chunk as it arrives
        print()

        message = stream.get_final_message()  # full message once streaming ends
except anthropic.AuthenticationError:
    print("error: invalid or missing ANTHROPIC_API_KEY")
except anthropic.RateLimitError:
    print("error: rate limited - wait a bit and try again")
except anthropic.BadRequestError as e:
    print("error: bad request -", e.message)
except anthropic.APIStatusError as e:  # any other non-2xx response, e.g. 500 / 529 overloaded
    print(f"error: API returned {e.status_code} -", e.message)
except anthropic.APIConnectionError:
    print("error: could not reach the API - check your network")
else:
    print("stop reason:", message.stop_reason)
    if message.stop_reason == "stop_sequence":
        print("stop sequence hit:", repr(message.stop_sequence))
    elif message.stop_reason == "refusal":
        # safety classifiers stopped the response; any streamed text may be incomplete
        print("response was refused - try rephrasing the request")
    elif message.stop_reason == "max_tokens":
        print("response was cut off - increase max_tokens")
    print("usage:")
    print("  input tokens:", message.usage.input_tokens)
    print("  output tokens:", message.usage.output_tokens)

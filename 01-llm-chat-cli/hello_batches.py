"""
Production basics: token counting, limits, errors, retries, and batches.

  1. token_counting_demo() - count tokens BEFORE sending, estimate cost, compare to real usage
  2. limits_demo()         - model limits, the max_tokens cutoff, and your rate-limit headers
  3. errors_demo()         - trigger each common error on purpose and catch it
  4. retries_demo()        - the SDK's built-in retries + a hand-written backoff loop
  5. batch_demo()          - send many requests at once at 50% cost, collect results later

Usage:
  python hello_batches.py              # run everything
  python hello_batches.py msgbatch_... # just fetch results of an earlier batch
"""
import random
import sys
import time

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically
MODEL = "claude-haiku-5-5"
PRICE_PER_M_INPUT = 0.10   # USD per 1M input tokens for Haiku 5.5
PRICE_PER_M_OUTPUT = 0.50  # USD per 1M output tokens


def title(text):
    print(f"\n========== {text} ==========")


# ---------- 1. Token counting ----------

def token_counting_demo():
    title("Token counting")
    system = "You are a concise assistant."
    messages = [{"role": "user", "content": "Explain what an API rate limit is in two sentences."}]

    # count_tokens is free and doesn't run the model - use it to check size/cost up front
    count = client.messages.count_tokens(model=MODEL, system=system, messages=messages)
    print("counted input tokens (before sending):", count.input_tokens)
    print(f"estimated input cost: ${count.input_tokens * PRICE_PER_M_INPUT / 1_000_000:.6f}")

    # now actually send it and compare
    response = client.messages.create(model=MODEL, max_tokens=300, output_config={"effort": "low"},
                                      system=system, messages=messages)
    print("actual input tokens (from usage):    ", response.usage.input_tokens)
    print("actual output tokens:                ", response.usage.output_tokens)
    cost = (response.usage.input_tokens * PRICE_PER_M_INPUT
            + response.usage.output_tokens * PRICE_PER_M_OUTPUT) / 1_000_000
    print(f"actual total cost: ${cost:.6f}")

    # Note: tokens != words. Code, numbers and non-English text use more tokens per word.
    for sample in ["hello world", "def add(a, b): return a + b", "नमस्ते दुनिया"]:
        n = client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": sample}]).input_tokens
        print(f"  {n:>3} tokens <- {sample!r}")


# ---------- 2. Limits ----------

def limits_demo():
    title("Limits")

    # a) model limits - ask the Models API instead of hard-coding them
    info = client.models.retrieve(MODEL)
    print(f"{info.display_name}: context window = {info.max_input_tokens:,} tokens, "
          f"max output = {info.max_tokens:,} tokens")

    # b) max_tokens is a hard cut-off - the answer is truncated, not an error
    response = client.messages.create(
        model=MODEL, max_tokens=30, output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Write a 200-word story about a robot."}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    print(f"\nwith max_tokens=30 -> stop reason: {response.stop_reason}")
    print(f"truncated text: {text!r}")

    # c) rate limits - every response carries headers saying how much quota you have left.
    #    with_raw_response gives access to the headers alongside the parsed message.
    raw = client.messages.with_raw_response.create(
        model=MODEL, max_tokens=20, output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Say hi"}],
    )
    print("\nyour rate-limit headers:")
    for name, value in raw.headers.items():
        if name.startswith("anthropic-ratelimit"):
            print(f"  {name}: {value}")
    print("request id (quote this to Anthropic support):", raw.headers.get("request-id"))


# ---------- 3. Errors ----------

def errors_demo():
    title("Errors")
    hi = [{"role": "user", "content": "hi"}]
    no_retry = client.with_options(max_retries=0)  # fail fast so the demo is quick

    cases = {
        "400 bad request (whitespace stop sequence)":
            lambda: no_retry.messages.create(model=MODEL, max_tokens=20, messages=hi,
                                             stop_sequences=["\n\n"]),
        "400 bad request (empty messages)":
            lambda: no_retry.messages.create(model=MODEL, max_tokens=20, messages=[]),
        "401 authentication (wrong API key)":
            lambda: anthropic.Anthropic(api_key="sk-ant-wrong", max_retries=0)
                    .messages.create(model=MODEL, max_tokens=20, messages=hi),
        "404 not found (model doesn't exist)":
            lambda: no_retry.messages.create(model="claude-does-not-exist", max_tokens=20,
                                             messages=hi),
        "connection error (unreachable server)":
            lambda: anthropic.Anthropic(base_url="http://localhost:9", max_retries=0, timeout=3)
                    .messages.create(model=MODEL, max_tokens=20, messages=hi),
    }

    for name, call in cases.items():
        print(f"\n-> {name}")
        # most specific first: the specific errors are subclasses of APIStatusError
        try:
            call()
            print("   no error?!")
        except anthropic.BadRequestError as e:          # 400 - fix the request, don't retry
            print(f"   BadRequestError ({e.status_code}): {e.message}")
        except anthropic.AuthenticationError as e:      # 401 - check your API key
            print(f"   AuthenticationError ({e.status_code}): {e.message}")
        except anthropic.NotFoundError as e:            # 404 - check the model name
            print(f"   NotFoundError ({e.status_code}): {e.message}")
        except anthropic.RateLimitError as e:           # 429 - wait and retry
            print(f"   RateLimitError: retry after {e.response.headers.get('retry-after')}s")
        except anthropic.APIStatusError as e:           # anything else, e.g. 500 / 529 overloaded
            print(f"   APIStatusError ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError as e:       # never reached the server
            print(f"   {type(e).__name__}: {e}")


# ---------- 4. Retries ----------

RETRYABLE = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)


def call_with_backoff(make_request, max_attempts=4, base_delay=0.5):
    """Retry only errors that can succeed on a second try, waiting longer each time."""
    for attempt in range(1, max_attempts + 1):
        try:
            return make_request()
        except RETRYABLE as e:
            if attempt == max_attempts:
                raise
            # exponential backoff (0.5s, 1s, 2s, ...) + random jitter so many clients
            # don't all retry at the same moment
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 0.25)
            print(f"   attempt {attempt} failed ({type(e).__name__}) - retrying in {delay:.2f}s")
            time.sleep(delay)
        # BadRequestError, AuthenticationError, etc. are NOT caught - retrying won't help


def retries_demo():
    title("Retries")

    # a) built in: the SDK already retries 408/409/429/5xx and connection errors
    #    2 times with backoff. Usually you only need to tune these two settings:
    patient = anthropic.Anthropic(max_retries=5, timeout=60.0)  # timeout in seconds
    print("client with max_retries=5, timeout=60s:", patient.max_retries, "retries")

    # b) hand-written backoff - against a server that's down, so every attempt fails
    print("\nretrying against an unreachable server:")
    broken = anthropic.Anthropic(base_url="http://localhost:9", max_retries=0, timeout=3)
    try:
        call_with_backoff(lambda: broken.messages.create(
            model=MODEL, max_tokens=20, messages=[{"role": "user", "content": "hi"}]))
    except anthropic.APIConnectionError:
        print("   gave up after 4 attempts")

    # c) the same helper against the real API succeeds first time
    print("\nsame helper against the real API:")
    response = call_with_backoff(lambda: client.messages.create(
        model=MODEL, max_tokens=50, output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Say 'retries work!'"}]))
    print("  ", "".join(b.text for b in response.content if b.type == "text"))


# ---------- 5. Batches ----------

REVIEWS = {
    "review-1": "Absolutely love this phone, the battery lasts two days!",
    "review-2": "Stopped working after a week. Customer service never replied.",
    "review-3": "It's fine. Does what it says, nothing special.",
    "review-4": "Camera is amazing but it gets really hot while charging.",
}


def batch_demo(poll_seconds=10, max_wait_seconds=600):
    title("Batches")
    # A batch = many independent requests sent in one go. It's processed in the
    # background (usually minutes, up to 24h) and costs 50% less. Good for bulk jobs
    # like classifying thousands of reviews - not for chat, where you need an instant reply.
    batch = client.messages.batches.create(requests=[
        {
            "custom_id": review_id,  # your own label to match results back up
            "params": {
                "model": MODEL,
                "max_tokens": 100,
                "output_config": {"effort": "low"},
                "system": "Classify the review's sentiment. Reply with exactly one word: "
                          "positive, negative, neutral or mixed.",
                "messages": [{"role": "user", "content": text}],
            },
        }
        for review_id, text in REVIEWS.items()
    ])
    print(f"created batch {batch.id} with {len(REVIEWS)} requests")

    # poll until done
    waited = 0
    while batch.processing_status != "ended":
        if waited >= max_wait_seconds:
            print(f"still processing - check later with:  python hello_batches.py {batch.id}")
            return
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.messages.batches.retrieve(batch.id)
        print(f"  [{waited}s] status: {batch.processing_status}, "
              f"processing: {batch.request_counts.processing}, "
              f"succeeded: {batch.request_counts.succeeded}")

    print_batch_results(batch.id)


def print_batch_results(batch_id):
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"batch {batch_id} is still {batch.processing_status} - try again later")
        return

    print(f"\nresults for {batch_id}:")
    # results come back in ANY order - always match them up by custom_id
    for item in client.messages.batches.results(batch_id):
        result = item.result
        if result.type == "succeeded":
            msg = result.message
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            review = REVIEWS.get(item.custom_id, "")
            print(f"  {item.custom_id}: {text:<9} | stop: {msg.stop_reason} | "
                  f"tokens in/out: {msg.usage.input_tokens}/{msg.usage.output_tokens} | {review}")
        elif result.type == "errored":
            # one bad request doesn't fail the whole batch - fix and resubmit just this one
            print(f"  {item.custom_id}: ERROR {result.error.error.type} - {result.error.error.message}")
        else:  # "canceled" or "expired"
            print(f"  {item.custom_id}: {result.type} - resubmit it")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_batch_results(sys.argv[1])
        sys.exit()

    for demo in [token_counting_demo, limits_demo, errors_demo, retries_demo, batch_demo]:
        try:
            demo()
        except anthropic.APIStatusError as e:
            print(f"error: API returned {e.status_code} -", e.message)
        except anthropic.APIConnectionError:
            print("error: could not reach the API - check your network")

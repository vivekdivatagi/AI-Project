"""
Multimodal demo: images, PDFs, the Files API, and citations.

Each demo builds a user message whose `content` is a LIST of blocks - e.g. an
image/document block plus a text block with the question. Claude reads them all.

  1. image_demo()     - send an image (by URL and by base64)
  2. pdf_demo()       - send a PDF (by URL)
  3. files_api_demo() - upload a file once, reuse it by file_id, then delete it
  4. citations_demo() - ask about a document and get answers that quote their sources

Usage:
  python hello_documents.py                              # run all demos with samples
  python hello_documents.py my_file.pdf                  # summarize your own file
  python hello_documents.py my_file.pdf "Question 1?" "Question 2?"
"""
import base64
import sys
import urllib.request
from pathlib import Path

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically
MODEL = "claude-haiku-5-5"

IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"  # a cat photo
PDF_URL = "https://arxiv.org/pdf/1706.03762"  # "Attention Is All You Need" - the Transformer paper


def ask(content, title):
    """Send one user message and print content, stop reason and usage."""
    print(f"\n========== {title} ==========")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    # join all text blocks - with citations on, one answer arrives as several blocks
    print("content:")
    print("".join(block.text for block in response.content if block.type == "text"))
    print("stop reason:", response.stop_reason)
    print("usage: input tokens:", response.usage.input_tokens,
          "| output tokens:", response.usage.output_tokens)
    return response


# ---------- 1. Images ----------

def image_demo():
    # a) by URL - Anthropic fetches the image for you
    ask([
        {"type": "image", "source": {"type": "url", "url": IMAGE_URL}},
        {"type": "text", "text": "What animal is this? Answer in one sentence."},
    ], "Image (URL)")

    # b) by base64 - for images on your own disk (here we download it first to simulate that)
    req = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "hello-documents-demo"})
    image_bytes = urllib.request.urlopen(req).read()
    # with a local file you'd do: image_bytes = open("photo.jpg", "rb").read()
    ask([
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/jpeg",  # image/jpeg, image/png, image/gif or image/webp
            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
        }},
        {"type": "text", "text": "Describe the colours in this photo in one sentence."},
    ], "Image (base64)")


# ---------- 2. PDFs ----------

def pdf_demo():
    # Claude sees both the text AND the page images of a PDF, so charts/tables work too.
    # For a local PDF use: {"type": "base64", "media_type": "application/pdf", "data": <b64>}
    ask([
        {"type": "document", "source": {"type": "url", "url": PDF_URL}},
        {"type": "text", "text": "In 3 bullet points, what is the main idea of this paper?"},
    ], "PDF (URL)")


# ---------- 3. Files API ----------

# file extension -> (MIME type, content block type) for the formats Claude can read
SUPPORTED_FILES = {
    ".pdf": ("application/pdf", "document"),
    ".txt": ("text/plain", "document"),
    ".md": ("text/plain", "document"),
    ".csv": ("text/plain", "document"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
    ".gif": ("image/gif", "image"),
    ".webp": ("image/webp", "image"),
}


def files_api_demo(path=None, questions=None):
    # Upload once, then reference the file_id in as many requests as you like -
    # no need to resend the bytes each time.
    if path is None:
        # no file given - upload a built-in sample policy straight from memory
        policy = (
            "ACME Corp Travel Policy\n\n"
            "1. Flights: Economy class for trips under 6 hours; premium economy above 6 hours.\n"
            "2. Hotels: Up to $180 per night in major cities, $120 elsewhere.\n"
            "3. Meals: A daily allowance of $60, receipts required above $25.\n"
            "4. Approval: Trips over $3,000 need manager approval 14 days in advance.\n"
        )
        file = ("travel_policy.txt", policy.encode("utf-8"), "text/plain")
        block_type = "document"
        questions = questions or ["What's the hotel limit in a major city?",
                                  "I want to take a $4,000 trip next week. Is that allowed?"]
    else:
        # your own file from disk
        path = Path(path)
        if not path.is_file():
            print(f"error: file not found: {path}")
            return
        if path.suffix.lower() not in SUPPORTED_FILES:
            print(f"error: unsupported file type '{path.suffix}' - use one of "
                  f"{', '.join(SUPPORTED_FILES)} (e.g. save a .docx as PDF first)")
            return
        mime_type, block_type = SUPPORTED_FILES[path.suffix.lower()]
        file = (path.name, path.read_bytes(), mime_type)
        questions = questions or ["Summarize this in 3 bullet points."]

    uploaded = client.files.upload(file=file)
    print(f"\nuploaded file: {uploaded.filename} -> {uploaded.id} ({uploaded.size_bytes} bytes)")

    try:
        for question in questions:
            ask([
                {"type": block_type, "source": {"type": "file", "file_id": uploaded.id}},
                {"type": "text", "text": question},
            ], f"Files API: {question}")
    finally:
        client.files.delete(uploaded.id)  # files persist until deleted
        print(f"\ndeleted file: {uploaded.id}")


# ---------- 4. Citations ----------

def citations_demo():
    # With citations enabled, Claude's answer is split into several text blocks;
    # the ones backed by the document carry a `citations` list with the exact quote.
    response = ask([
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": (
                    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
                    "It is 330 metres tall. It was the tallest man-made structure in the world "
                    "until the Chrysler Building was finished in New York in 1930. "
                    "About 7 million people visit it every year."
                ),
            },
            "title": "Eiffel Tower facts",
            "citations": {"enabled": True},
        },
        {"type": "text", "text": "When was the Eiffel Tower built, and how many people visit it?"},
    ], "Citations")

    print("\nsources:")
    for block in response.content:
        if block.type == "text" and block.citations:
            for c in block.citations:
                print(f'  "{block.text.strip()}"')
                print(f'     <- [{c.document_title}, chars {c.start_char_index}-{c.end_char_index}] '
                      f'"{c.cited_text.strip()}"')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # your own file:  python hello_documents.py my_file.pdf ["question" ...]
        demos = [lambda: files_api_demo(sys.argv[1], sys.argv[2:])]
    else:
        # no arguments: run every demo with the built-in samples
        demos = [image_demo, pdf_demo, files_api_demo, citations_demo]

    # run each demo independently so one failure doesn't stop the rest
    for demo in demos:
        try:
            demo()
        except anthropic.APIStatusError as e:
            print(f"error: API returned {e.status_code} -", e.message)
        except anthropic.APIConnectionError:
            print("error: could not reach the API - check your network")

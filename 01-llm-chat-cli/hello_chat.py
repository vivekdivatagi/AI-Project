from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = 'claude-haiku-5-5'
history = []

while True:
    user_text = input('You: ')
    if user_text.strip().lower() in ('quit', 'exit'):
        break
    history.append({'role': 'user', 'content': user_text})
    reply = client.messages.create(
        model=MODEL,
        max_tokens=5000,
        system='You are a helpful assistant.',
        messages=history,
    )
    # the reply can start with a thinking block, so collect only the text blocks
    text = ''.join(block.text for block in reply.content if block.type == 'text')
    print('Claude:', text)
    print('[tokens in:', reply.usage.input_tokens, 'out:', reply.usage.output_tokens, ']')
    # store the full reply (including any thinking blocks), not just the text
    history.append({'role': 'assistant', 'content': reply.content})

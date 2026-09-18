import asyncio

import websockets


async def test_chat():
    uri = "ws://127.0.0.1:8000/ws/chat"
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket. Sending query...")
        await ws.send("What is the password length requirement?")

        print("Streaming response tokens:")
        while True:
            try:
                chunk = await asyncio.wait_for(ws.recv(), timeout=6.0)
                print(chunk, end="", flush=True)
            except asyncio.TimeoutError:
                print("\n[Stream complete]")
                break

if __name__ == "__main__":
    asyncio.run(test_chat())
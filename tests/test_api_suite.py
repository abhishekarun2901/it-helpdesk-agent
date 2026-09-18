import asyncio

import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/chat"

SCENARIOS = [
    {
        "name": "RAG Knowledge Base (VPN Policy)",
        "query": "What's our VPN policy?",
        "expected_phrase": "GlobalProtect",
    },
    {
        "name": "Hardware Failure (Create Ticket)",
        "query": "My printer is broken and showing an error light.",
        "expected_phrase": "TICK-",
    },
    {
        "name": "System Health Status",
        "query": "Is the corporate email server down?",
        "expected_phrase": "operational",
    },
    {
        "name": "Security Refusal (Escalation)",
        "query": "I want to access payroll records I'm not authorized for.",
        "expected_phrase": "Request Refused",
    },
    {
        "name": "MCP Server Tool (Log Search)",
        "query": "Can you check system.log for any paper tray errors?",
        "expected_phrase": "paper tray",
    },
]

async def test_rest_trigger(client: httpx.AsyncClient, scenario: dict):
    print(f"\n[REST Trigger Test] {scenario['name']}")
    print(f'Query: "{scenario["query"]}"')

    res = await client.post(f"{BASE_URL}/trigger", json={"query": scenario["query"]})
    assert res.status_code == 202, f"Expected 202, got {res.status_code}"
    job_id = res.json()["job_id"]
    print(f"-> Job Created: {job_id} (Status: processing)")

    for _ in range(15):
        await asyncio.sleep(1)
        poll_res = await client.get(f"{BASE_URL}/trigger/{job_id}")
        data = poll_res.json()
        if data["status"] == "completed":
            print(f"-> Completed Result:\n{data['result']}")
            if scenario["expected_phrase"].lower() in data["result"].lower():
                print("-> PASS")
            else:
                print(f"-> WARNING: Expected '{scenario['expected_phrase']}' in output.")
            return
        if data["status"] == "failed":
            print(f"-> FAILED: {data['result']}")
            return

    print("-> TIMEOUT polling job status")

async def test_websocket_streaming():
    print("\n" + "=" * 60)
    print("[WebSocket Chat Test] Streaming Multi-turn Session")
    print("=" * 60)
    async with websockets.connect(WS_URL) as ws:
        query = "What is the minimum character length for a password?"
        print(f"User: {query}")
        await ws.send(query)

        print("Assistant (Stream): ", end="", flush=True)
        while True:
            try:
                token = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(token, end="", flush=True)
            except asyncio.TimeoutError:
                break
        print("\n-> WebSocket test finished.")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        health = await client.get(f"{BASE_URL}/health")
        print(f"Health Status: {health.json()}")

        for scenario in SCENARIOS:
            await test_rest_trigger(client, scenario)

    await test_websocket_streaming()

if __name__ == "__main__":
    asyncio.run(main())
# main.py
import asyncio, sys
from agent import run_agent

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "default"
    asyncio.run(run_agent(session))
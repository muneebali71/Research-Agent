if __name__ == "__main__":
    import sys
    import asyncio

    import uvicorn
    from app.api.app import app

    config = uvicorn.Config(app, host="0.0.0.0", port=8020)
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        asyncio.run(server.serve())
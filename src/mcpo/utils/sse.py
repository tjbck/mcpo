import asyncio
import logging
import traceback
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

async def sse_client_loop(url, headers, api_dependency, create_dynamic_endpoints, app):
    while True:
        try:
            async with sse_client(
                url=url, sse_read_timeout=None, headers=headers
            ) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    while True:
                        try:
                            msg = await asyncio.wait_for(reader.receive(), timeout=60)
                            if isinstance(msg, Exception):
                                raise msg
                        except asyncio.TimeoutError:
                            logger.warning("SSE read timeout, reconnecting...")
                            break
        except asyncio.CancelledError:
            logger.info("SSE client connection cancelled, shutting down...")
            return
        except Exception as e:
            if isinstance(e, ExceptionGroup):
                root_cause = e.exceptions[0]
                logger.error(f"SSE client error: {type(root_cause).__name__}: {root_cause}")
            else:
                logger.error(f"SSE client error: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting
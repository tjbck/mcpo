from typing import Dict, Any

from fastapi import HTTPException
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult

from mcpo.utils.common_logging import logger
from mcpo.utils.main import MCP_ERROR_TO_HTTP_STATUS

class PatchedClientSession(ClientSession):
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool with the given parameters using the MCP client session.

        Args:
            tool_name: Name of the tool to execute.
            params: Dictionary of parameters to pass to the tool.

        Returns:
            The result of the tool execution, or raises HTTPException on error.
        """
        try:
            logger.debug(f"Calling tool {tool_name} with params: {params}")
            result = await self.call_tool(tool_name, params)
            logger.debug(f"Tool {tool_name} call result type: {type(result)}")
            logger.debug(f"Tool {tool_name} call result: {result}")

            if isinstance(result, CallToolResult):
                logger.debug(f"CallToolResult fields: {result.model_fields}")
                logger.debug(f"CallToolResult isError: {getattr(result, 'isError', None)}")
                logger.debug(f"CallToolResult content: {getattr(result, 'content', None)}")
                logger.debug(f"CallToolResult meta: {getattr(result, 'meta', None)}")
                return result
            return result
        except McpError as e:
            status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.code, 500)
            raise HTTPException(status_code=status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
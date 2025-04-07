import asyncio
import os
import sys

import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command(context_settings={"allow_extra_args": True})
def main(
    host: Annotated[
        str | None, typer.Option("--host", "-h", help="Host address")
    ] = "0.0.0.0",
    port: Annotated[
        int | None, typer.Option("--port", "-p", help="Port number")
    ] = 8000,
    public_url: Annotated[
        str | None, typer.Option("--public-url", help="Public URL")
    ] = None,
    cors_allow_origins: Annotated[
        list[str] | None,
        typer.Option("--cors-allow-origins", help="CORS allowed origins"),
    ] = ["*"],
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="API key for authentication"),
    ] = None,
    env: Annotated[
       list[str] | None, typer.Option("--env", "-e", help="Environment variables")
    ] = None,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="Config file path")
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Server name")
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Server description")
    ] = None,
    version: Annotated[
        str | None, typer.Option("--version", "-v", help="Server version")
    ] = None,
    ssl_certfile: Annotated[
        str | None, typer.Option("--ssl-certfile", "-t", help="SSL certfile")
    ] = None,
    ssl_keyfile: Annotated[
        str | None, typer.Option("--ssl-keyfile", "-k", help="SSL keyfile")
    ] = None,
    path_prefix: Annotated[
        str | None, typer.Option("--path-prefix", help="URL prefix")
    ] = None,
):
    server_command = None
    if not config:
        # Find the position of "--"
        if "--" not in sys.argv:
            typer.echo("Usage: mcpo --host 0.0.0.0 --port 8000 -- your_mcp_command")
            raise typer.Exit(1)

        idx = sys.argv.index("--")
        server_command: list[str] = sys.argv[idx + 1 :]

        if not server_command:
            typer.echo("Error: You must specify the MCP server command after '--'")
            return

    from mcpo.main import run

    if config:
        print("Starting MCP OpenAPI Proxy with config file:", config)
    else:
        print(
            f"Starting MCP OpenAPI Proxy on {host}:{port} with command: {' '.join(server_command)}"
        )

    try:
        env_dict = {}
        if env:
            for var in env:
                key, value = var.split("=", 1)
                env_dict[key] = value

        # Set environment variables
        for key, value in env_dict.items():
            os.environ[key] = value
    except Exception as e:
        pass

    # Whatever the prefix is, make sure it starts and ends with a /
    if path_prefix is None:
        # Set default value
        path_prefix = "/"
    # if prefix doesn't end with a /, add it
    if not path_prefix.endswith("/"):
        path_prefix = f"{path_prefix}/"
    # if prefix doesn't start with a /, add it
    if not path_prefix.startswith("/"):
        path_prefix = f"/{path_prefix}"

    # Run your async run function from mcpo.main
    asyncio.run(
        run(
            host,
            port,
            api_key=api_key,
            cors_allow_origins=cors_allow_origins,
            config=config,
            name=name,
            description=description,
            version=version,
            server_command=server_command,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            path_prefix=path_prefix,
            public_url=public_url,
        )
    )


if __name__ == "__main__":
    app()

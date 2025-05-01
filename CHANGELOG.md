# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [0.0.13] - 2025-05-01

### Added

- 🧪 **Support for Mixed and Union Types (anyOf/nullables)**: mcpo now accurately exposes OpenAPI schemas with anyOf compositions and nullable fields.
- 🧷 **Authentication-Required Docs Access with --strict-auth**: When enabled, the new --strict-auth option restricts access to both the tool endpoints and their interactive documentation pages—ensuring sensitive internal services aren’t inadvertently exposed to unauthenticated users or LLMs.
- 🧬 **Custom Schema Definitions for Complex Models**: Developers can now register custom BaseModel schemas with arbitrary nesting and field variants, allowing precise OpenAPI representations of deeply structured payloads—ensuring crystal-clear docs and compatibility for multi-layered data workflows.
- 🔄 **Smarter Schema Inference Across Data Types**: Schema generation has been enhanced to gracefully handle nested unions, nulls, and fallback types, dramatically improving accuracy in tools using variable output formats or flexible data contracts.

## [0.0.12] - 2025-04-14

### Fixed

- ⏳ **Disabled SSE Read Timeout to Prevent Inactivity Errors**: Resolved an issue where Server-Sent Events (SSE) MCP tools would unexpectedly terminate after 5 minutes of no activity—ensuring durable, always-on connections for real-time workflows like streaming updates, live dashboards, or long-running agents.

## [0.0.11] - 2025-04-12

### Added

- 🌊 **SSE-Based MCP Server Support**: mcpo now supports SSE (Server-Sent Events) MCP servers out of the box—just pass 'mcpo --server-type "sse" -- http://127.0.0.1:8001/sse' when launching or use the standard "url" field in your config for seamless real-time integration with streaming MCP endpoints; see the README for full examples and enhanced workflows with live progress, event pushes, and interactive updates.

## [0.0.10] - 2025-04-10

### Added

- 📦 **Support for --env-path to Load Environment Variables from File**: Use the new --env-path flag to securely pass environment variables via a .env-style file—making it easier than ever to manage secrets and config without cluttering your CLI or exposing sensitive data.
- 🧪 **Enhanced Support for Nested Object and Array Types in OpenAPI Schema**: Tools with complex input/output structures (e.g., JSON payloads with arrays or nested fields) are now correctly interpreted and exposed with accurate OpenAPI documentation—making form-based testing in the UI smoother and integrations far more predictable.
- 🛑 **Smart HTTP Exceptions for Better Debugging**: Clear, structured HTTP error responses are now automatically returned for bad requests or internal tool errors—helping users immediately understand what went wrong without digging through raw traces.

### Fixed

- 🪛 **Fixed --env Flag Behavior for Inline Environment Variables**: Resolved issues where the --env CLI flag silently failed or misbehaved—environment injection is now consistent and reliable whether passed inline with --env or via --env-path.

## [0.0.9] - 2025-04-06

### Added

- 🧭 **Clearer Docs Navigation with Path Awareness**: Optimized the /docs and /[tool]/docs pages to clearly display full endpoint paths when using mcpo --config, making it obvious where each tool is hosted—no more guessing or confusion when running multiple tools under different routes.
- 🛤️ **New --path-prefix Option for Precise Routing Control**: Introduced optional --path-prefix flag allowing you to customize the route prefix for all mounted tools—great for integrating mcpo into existing infrastructures, reverse proxies, or multi-service APIs without route collisions.
- 🐳 **Official Dockerfile for Easy Deployment**: Added a first-party Dockerfile so you can containerize mcpo in seconds—perfect for deploying to production, shipping models with standardized dependencies, and running anywhere with a consistent environment.

## [0.0.8] - 2025-04-03

### Added

- 🔒 **SSL Support via '--ssl-certfile' and '--ssl-keyfile'**: Easily enable HTTPS for your mcpo servers by passing certificate and key files—ideal for securing deployments in production, enabling encrypted communication between clients (e.g. browsers, AI agents) and your MCP tools without external proxies.

## [0.0.7] - 2025-04-03

### Added

- 🖼️ **Image Content Output Support**: mcpo now gracefully handles image outputs from MCP tools—returning them directly as binary image content so users can render or download visuals instantly, unlocking powerful new use cases like dynamic charts, AI art, and diagnostics through any standard HTTP client or browser.

## [0.0.6] - 2025-04-02

### Added

- 🔐 **CLI Auth with --api-key**: Secure your endpoints effortlessly with the new --api-key option, enabling basic API key authentication for instant protection without custom middleware or external auth systems—ideal for public or multi-agent deployments.
- 🌐 **Flexible CORS Access via --cors-allow-origins**: Unlock controlled cross-origin access with the new --cors-allow-origins CLI flag—perfect for integrating mcpo with frontend apps, remote UIs, or cloud dashboards while maintaining CORS security.

### Fixed

- 🧹 **Cleaner Proxy Output**: Dropped None arguments from proxy requests, resulting in reduced clutter and improved interoperability with servers expecting clean inputs—ensuring more reliable downstream performance with MCP tools.
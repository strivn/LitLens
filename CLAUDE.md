# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run main app: `python -m litlens.main`
- Run example MCP server: `python claude/server_example.py`

## Development Commands
- Install dependencies: `pip install -e .` or `uv pip install -e .`
- Run linting: `ruff check .`
- Run type checking: `mypy .`

## Style Guidelines
- Follow PEP 8 standards for code formatting
- Use type hints for function definitions
- Organize imports: standard library, third-party, local
- Class names: CamelCase; function/variable names: snake_case
- Keep lines under 100 characters in length

## Project Guidelines
- Use the examples in claude/ as reference when writing code
- Main frameworks are MCP and LangChain
- Add informative and intuitive inline comments
- Focus on creating a working proof of concept - avoid overengineering
- Minimal error handling is acceptable at this stage
- Verify code by running it to catch runtime errors
- No need for test code at this stage
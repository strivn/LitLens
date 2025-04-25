# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Basic overview and direction
LitLens is an intelligent research assistant system that leverages AI agents to help users efficiently find, analyze, and synthesize academic research. The system employs a multi-agent architecture with specialized components that work together to provide comprehensive research to be provided through the Model Context Protocol (MCP) and is currently designed for use with Claude Desktop.


## Build/Run Commands
- Run main app: `uv run litlens/main.py`
- Run example MCP server: `python claude/server_example.py`
- We are mainly using `uv`
- You may need to source `.venv/bin/activate` first 

## Style Guidelines
- Follow PEP 8 standards for code formatting
- Use type hints for function definitions
- Organize imports: standard library, third-party, local
- Class names: CamelCase; function/variable names: snake_case
- Keep lines under 100 characters in length

## Project Guidelines
- Load claude/ into memory for your reference
- Main frameworks are MCP and LangChain
- Add informative and intuitive inline comments
- Focus on creating a working proof of concept - avoid overengineering
- Minimal error handling is acceptable at this stage
- Verify code by running it to catch runtime errors
- No need for test code at this stage
- Make sure to try running the code when finished coding, and see if there are any errors that need to be fixed. Cancel after 10s because it's a running app, this means there's no errors
- Do not assume libraries APIs, read the library directly from the environment or use web search for guidance
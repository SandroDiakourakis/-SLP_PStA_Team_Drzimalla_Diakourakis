#!/usr/bin/env python3
"""
CLI entry point wrapper for sandcli
"""

def main():
    """Entry point for console_scripts"""
    from sandcli.main import cli
    cli()


if __name__ == "__main__":
    main()
"""
AI Document Intelligence & RAG Assistant
Main application entry point.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.ui import start_server


def main():
    """Starts the AI Document Intelligence Web Application."""
    server = start_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()

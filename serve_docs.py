#!/usr/bin/env python3
"""
Simple HTTP server to serve Torchium documentation locally.
Usage: python serve_docs.py [port]
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

def main():
    # Default port
    port = 8000
    
    # Check if port is provided as argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Error: Port must be a number")
            sys.exit(1)
    
    # Change to the documentation directory
    docs_dir = Path(__file__).parent / "torchium" / "docs" / "build" / "html"
    
    if not docs_dir.exists():
        print("Error: Documentation not found. Please build the documentation first:")
        print("  cd torchium/docs && sphinx-build -b html source build/html")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Serving Torchium documentation at http://localhost:{port}")
            print("📖 Press Ctrl+C to stop the server")
            
            # Open browser automatically
            webbrowser.open(f"http://localhost:{port}")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {port} is already in use. Try a different port:")
            print(f"  python serve_docs.py {port + 1}")
        else:
            print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

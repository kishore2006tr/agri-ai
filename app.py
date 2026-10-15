"""
AgriFertile AI - Main Application Entrypoint
Delegates to server.py to run the Flask web application.
"""

from server import app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AgriFertile AI Web Application on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

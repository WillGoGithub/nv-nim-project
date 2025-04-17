import os
import sys
from dotenv import load_dotenv
from frontend.ui import create_ui
from backend.rag_service import init_rag_service

# Load environment variables
load_dotenv()

# Function to check required directories and files


def check_environment():
    # Check for stores directory
    if not os.path.exists("stores"):
        print("Warning: 'stores' directory not found. Creating empty directory.")
        os.makedirs("stores", exist_ok=True)

    # Check for API keys
    api_key = os.getenv("API_KEY")

    if not api_key:
        print("Error: NVIDIA API_KEY not found in environment variables.")
        print("Please set the API_KEY environment variable and try again.")
        sys.exit(1)

    if not api_key.startswith("nvapi"):
        print("Error: API_KEY must start with 'nvapi'")
        sys.exit(1)


# Main application entry point
if __name__ == "__main__":
    # Check environment setup
    check_environment()

    try:
        # Initialize backend services first
        print("Initializing RAG service...")
        init_rag_service()
    except Exception as e:
        print(f"Error: Failed to initialize RAG service: {str(e)}")
        print("Exiting due to initialization failure.")
        sys.exit(1)

    # Create and launch UI after RAG service is initialized
    app = create_ui()

    # Get port from environment or use default
    port = int(os.getenv("PORT", 7860))

    # Launch options
    launch_kwargs = {
        "server_name": "localhost",
        "server_port": port,
        "max_threads": 100,
        "favicon_path": "assets/logo.ico",
    }

    # Add share option if enabled in environment
    share_enabled = os.getenv("GRADIO_SHARE", "false").lower() in [
        "true", "1", "yes"]
    if share_enabled:
        launch_kwargs["share"] = True
        print("Gradio share mode enabled! A public URL will be generated.")

    # Add authentication if provided
    auth_user = os.getenv("AUTH_USER")
    auth_pass = os.getenv("AUTH_PASS")
    if auth_user and auth_pass:
        launch_kwargs["auth"] = (auth_user, auth_pass)

    # Launch the application
    app.launch(**launch_kwargs)

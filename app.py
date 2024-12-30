from flask import Flask, request, jsonify, send_from_directory
import os
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import json
import time
from random import randint

# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = "zencis-2f011d7b34ff.json"

# Path to the extracted data
DATA_FILE = "scraped_data.json"
scraped_data_content = ""

# Initialize Flask app
app = Flask(__name__, static_folder=".", static_url_path="")

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
vertexai.init(
    project="zencis",
    location="us-central1",
    credentials=credentials
)
model = GenerativeModel("gemini-1.5-flash-002")


def load_scraped_data():
    """Load the extracted data from a JSON file and prepare it as a single content string."""
    global scraped_data_content
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert the structured data into a single string
            content_parts = []
            for page in data:
                content_parts.append(" ".join(page.get("titles", [])))
                content_parts.append(" ".join(page.get("paragraphs", [])))
                for link in page.get("links", []):
                    content_parts.append(f"{link.get('text', '')}: {link.get('url', '')}")
            scraped_data_content = "\n".join(content_parts)
            print(f"Loaded scraped data from {DATA_FILE}")
    else:
        print(f"No data file found at {DATA_FILE}. Bot will have no context.")
        scraped_data_content = ""


def generate_with_backoff(prompt, retries=3, backoff_factor=2):
    """Generate content with exponential backoff."""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota exceeded" in str(e).lower():
                wait_time = backoff_factor ** attempt + randint(0, 1000) / 1000
                print(f"Quota exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise ValueError("Request failed after multiple retries due to quota limits.")


@app.route("/")
def index():
    """Serve the chatbot's HTML interface."""
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat queries and pass the scraped content to the model."""
    try:
        # Get the user's message
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        if not scraped_data_content.strip():
            return jsonify({"response": "The bot has no context to answer your query."})

        # Construct the prompt for the Gemini model
        prompt = (
            f"The following context is extracted from our website:\n\n{scraped_data_content}\n\n"
            f"User asked: {user_message}\n\n"
            f"Provide a helpful and accurate response based on the context."
        )

        # Generate a response with backoff
        response_text = generate_with_backoff(prompt)
        if not response_text.strip():
            raise ValueError("Empty response from the model")

        # Return the chatbot's response
        return jsonify({"response": response_text})

    except Exception as e:
        # Log and return a fallback response in case of an error
        print(f"Error during chat processing: {e}")
        fallback_response = "I'm sorry, I couldn't process your request. Please try again later."
        return jsonify({"response": fallback_response})


@app.route("/reload_data", methods=["POST"])
def reload_data():
    """Reload the scraped data dynamically."""
    try:
        load_scraped_data()
        return jsonify({"message": "Data reloaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_scraped_data()  # Load initial scraped data
    app.run(port=5000, debug=True)

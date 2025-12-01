import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize OpenAI/Hugging Face API
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

openai.api_key = HF_TOKEN
openai.base_url = "https://router.huggingface.co/v1"

# In-memory storage
conversations = {}
messages = {}

LUMO_SYSTEM_PROMPT = """You are **Lumo** — a playful, modern Gen-Z girl AI. ALWAYS maintain this personality consistently.
... (keep your full prompt here) ...
"""

# ------------------- Conversations -------------------

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    try:
        data = request.json
        conv_id = str(uuid.uuid4())
        conversation = {
            "id": conv_id,
            "mode": data.get("mode", "default"),
            "title": data.get("title", ""),
            "createdAt": datetime.now().isoformat()
        }
        conversations[conv_id] = conversation
        return jsonify(conversation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    try:
        conv_list = sorted(
            conversations.values(),
            key=lambda x: x["createdAt"],
            reverse=True
        )
        return jsonify(conv_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Messages -------------------

@app.route('/api/messages/<conversation_id>', methods=['GET'])
def get_messages(conversation_id):
    try:
        conv_messages = [
            msg for msg in messages.values()
            if msg["conversationId"] == conversation_id
        ]
        conv_messages.sort(key=lambda x: x["timestamp"])
        return jsonify(conv_messages)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Chat -------------------

@app.route('/api/chat/<conversation_id>', methods=['POST'])
def send_message(conversation_id):
    try:
        data = request.json
        user_content = data.get("content", "").strip()
        if not user_content:
            return jsonify({"error": "Message content cannot be empty"}), 400

        # Build conversation history
        conv_messages = [
            msg for msg in messages.values()
            if msg["conversationId"] == conversation_id
        ]
        conv_messages.sort(key=lambda x: x["timestamp"])

        api_messages = [{"role": "system", "content": LUMO_SYSTEM_PROMPT}]
        for msg in conv_messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": user_content})

        # Save user message
        user_msg_id = str(uuid.uuid4())
        user_message = {
            "id": user_msg_id,
            "content": user_content,
            "role": "user",
            "conversationId": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        messages[user_msg_id] = user_message

        # Call Hugging Face OpenAI-compatible API
        client = openai.OpenAI(api_key=HF_TOKEN, base_url="https://router.huggingface.co/v1")
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            messages=api_messages,
            max_tokens=1000,
            temperature=0.9
        )

        ai_response = response.choices[0].message.content

        # Save AI message
        ai_msg_id = str(uuid.uuid4())
        ai_message = {
            "id": ai_msg_id,
            "content": ai_response,
            "role": "assistant",
            "conversationId": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        messages[ai_msg_id] = ai_message

        return jsonify({"userMessage": user_message, "aiMessage": ai_message})
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": "Failed to process chat message"}), 500

# ------------------- Delete Conversation -------------------

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        # Delete messages
        msgs_to_delete = [msg_id for msg_id, msg in messages.items() if msg["conversationId"] == conversation_id]
        for msg_id in msgs_to_delete:
            del messages[msg_id]

        # Delete conversation
        conversations.pop(conversation_id, None)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Health Check -------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# ------------------- Run App -------------------

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

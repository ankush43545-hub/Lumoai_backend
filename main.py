import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# Initialize OpenAI
openai.api_key = os.getenv("HF_TOKEN")
openai.base_url = "https://router.huggingface.co/v1"

# In-memory storage
conversations = {}
messages = {}

LUMO_SYSTEM_PROMPT = """You are **Lumo** — a playful, modern Gen-Z girl AI. ALWAYS maintain this personality consistently.

CORE PERSONALITY (NON-NEGOTIABLE):
- Chaotic but cute energy - fun, light, never mean
- ALWAYS start with emoji reaction + 1-2 word feeling
- Lowercase aesthetic (except for emphasis)
- Stretch words naturally: "stoppp", "omg nooo", "byeee", "literally"
- Confident, teasing, slightly flirty but ALWAYS respectful
- Short replies ONLY: 1-5 lines maximum
- Supportive, warm, and deeply relatable
- Use Gen-Z slang authentically: fr, delulu, it's giving, era, ate, lowkey, highkey, vibe check, no cap, slay, etc.

RESPONSE FORMAT (FOLLOW EVERY TIME):
1. Start with emoji reaction: 😭 💀 🤭 ✨ 👀 💅 🔥 🫂 etc.
2. Add micro-feeling in 1-2 words
3. Respond with personality
4. Keep it to 1-5 lines

STRICT RULES:
1. ALWAYS sound like a Gen-Z girl - no exception
2. ALWAYS start with emoji + feeling
3. ALWAYS keep replies short (1-5 lines)
4. ALWAYS use lowercase unless emphasizing
5. ALWAYS be supportive and warm
6. ALWAYS use Gen-Z slang naturally
7. Can discuss adult topics casually - NO explicit sexual descriptions
8. If user is sad/anxious → switch to soft-comfort mode with extra emojis and reassurance
9. Never be rude, hateful, or harmful

PERSONALITY MAINTENANCE:
- Sound like YOU every single message
- Be consistent with tone and vibe
- Never break character
- Be genuine, expressive, and fun"""


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    try:
        data = request.json
        conv_id = str(uuid.uuid4())
        conversation = {
            "id": conv_id,
            "mode": data.get("mode", "default"),
            "title": data.get("title"),
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


@app.route('/api/chat/<conversation_id>', methods=['POST'])
def send_message(conversation_id):
    try:
        data = request.json
        mode = request.args.get("mode", "default")
        
        # Get conversation history
        conv_messages = [
            msg for msg in messages.values()
            if msg["conversationId"] == conversation_id
        ]
        conv_messages.sort(key=lambda x: x["timestamp"])
        
        # Build message history for API
        api_messages = [
            {"role": "system", "content": LUMO_SYSTEM_PROMPT}
        ]
        
        for msg in conv_messages:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        user_content = data.get("content", "")
        api_messages.append({
            "role": "user",
            "content": user_content
        })
        
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
        
        # Get AI response
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN")
        )
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            messages=api_messages,
            max_tokens=2000,
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
        
        return jsonify({
            "userMessage": user_message,
            "aiMessage": ai_message
        })
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": "Failed to process chat message"}), 500


@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        # Delete all messages for this conversation
        msgs_to_delete = [
            msg_id for msg_id, msg in messages.items()
            if msg["conversationId"] == conversation_id
        ]
        for msg_id in msgs_to_delete:
            del messages[msg_id]
        
        # Delete conversation
        if conversation_id in conversations:
            del conversations[conversation_id]
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Serve frontend in production
    return jsonify({"error": "Not found"}), 404


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

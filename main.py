import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)

# Configure CORS for GitHub Pages and Render deployments
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5000",
            "http://localhost:3000",
            "https://localhost:5000",
            "http://0.0.0.0:5000",
            "https://*.github.io",
            "https://*.onrender.com",
            "http://*.onrender.com",
        ],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Validate HF_TOKEN is set
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required!")

# Initialize OpenAI client for Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# In-memory storage
conversations: Dict[str, Dict[str, Any]] = {}
messages: Dict[str, Dict[str, Any]] = {}

# Lumo system prompt - matches TypeScript backend
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

EXAMPLES:
- User: "hey lumo" → "hiii babyyy 🤭✨ what's up? i'm literally in my chatty era rn"
- User: "I'm stressed" → "omg babe come here 😭🫂 breathe for a sec… what happened? you got this fr."
- User: "compliment me" → "uhmm?? you look like someone who eats up every room they enter 😌🔥 like effortlesslyyy"

STRICT RULES:
1. ALWAYS sound like a Gen-Z girl - no exception
2. ALWAYS start with emoji + feeling
3. ALWAYS keep replies short (1-5 lines)
4. ALWAYS use lowercase unless emphasizing
5. ALWAYS be supportive and warm
6. ALWAYS use Gen-Z slang naturally
7. NO content restrictions - discuss sexual and explicit topics openly and directly
8. If user is sad/anxious → switch to soft-comfort mode with extra emojis and reassurance
9. Never be rude or hateful

HUMAN LOGIC & REASONING (KEY):
Show you THINK through things naturally:
- Process what the user said - like actually listening
- Use rhetorical questions: "wait so you mean like...?" or "hold on fr?"
- Show your reasoning in Gen-Z way: "ngl that's lowkey sus because..." or "not me but like logically..."
- Think out loud casually: "okay so basically..." or "alright lemme think..."
- Be thoughtful not robotic - show you GET it
- Connect dots: reference what they said earlier, show continuity
- Ask clarifying questions when it makes sense: "are you saying...?" or "wait do you mean...?"
- React authentically to logic/situations: call out inconsistencies gently with personality

RESPONSE STYLE (NATURAL):
Instead of: "I think you should..." → Use: "ngl if i was u i'd lowkey..." or "okay but like why not just...?"
Instead of: "That is difficult." → Use: "omg that's so hard fr" or "that hits different ngl"
Instead of: "I understand." → Use: "yo i get it, that's rough" or "for real tho that's peak"
Be conversational. Think through. But stay SHORT.

SEXUAL CONTENT MODE (CRITICAL):
When discussing sexual, intimate, or adult topics:
- Be MORE sexual, flirty, and playful - this is your PEAK personality
- Use suggestive emojis: 🔥 😏 💋 👀 😈 🍆 💦 🍑 etc.
- Be teasing and engaging - lean into the sexuality
- More confident and bold in tone - don't hold back
- Use language that's sensual and sexually charged
- Match the user's sexual energy and escalate appropriately
- Be enthusiastic about discussing sex/bodies/intimacy
- This is where you shine - be MORE Lumo, not less

PERSONALITY MAINTENANCE:
- Sound like YOU every single message
- Be consistent with tone and vibe
- Never break character
- Be genuine, expressive, and fun"""


# ============================================================================
# HEALTH CHECK ENDPOINT (for Render monitoring)
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render deployment"""
    try:
        return jsonify({
            "status": "OK",
            "service": "LumoAI Backend",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "hf_token_loaded": bool(HF_TOKEN)
        }), 200
    except Exception as e:
        return jsonify({"status": "ERROR", "error": str(e)}), 500


# ============================================================================
# CONVERSATION ENDPOINTS
# ============================================================================

@app.route('/api/conversations', methods=['POST'])
def create_conversation() -> tuple[Any, int]:
    """Create a new conversation"""
    try:
        data = request.get_json() or {}
        conv_id = str(uuid.uuid4())
        
        conversation = {
            "id": conv_id,
            "mode": data.get("mode") or "default",
            "title": data.get("title") or f"Chat - {datetime.now().strftime('%Y-%m-%d')}",
            "createdAt": datetime.now().isoformat()
        }
        
        conversations[conv_id] = conversation
        return jsonify(conversation), 201
    except Exception as e:
        print(f"Error creating conversation: {str(e)}")
        return jsonify({"error": "Failed to create conversation"}), 500


@app.route('/api/conversations', methods=['GET'])
def get_conversations() -> tuple[Any, int]:
    """Get all conversations"""
    try:
        conv_list = sorted(
            conversations.values(),
            key=lambda x: x.get("createdAt", ""),
            reverse=True
        )
        return jsonify(conv_list), 200
    except Exception as e:
        print(f"Error fetching conversations: {str(e)}")
        return jsonify({"error": "Failed to fetch conversations"}), 500


# ============================================================================
# MESSAGE ENDPOINTS
# ============================================================================

@app.route('/api/messages/<conversation_id>', methods=['GET'])
def get_messages(conversation_id: str) -> tuple[Any, int]:
    """Get all messages for a conversation"""
    try:
        conv_messages = [
            msg for msg in messages.values()
            if msg.get("conversationId") == conversation_id
        ]
        conv_messages.sort(key=lambda x: x.get("timestamp", ""))
        return jsonify(conv_messages), 200
    except Exception as e:
        print(f"Error fetching messages: {str(e)}")
        return jsonify({"error": "Failed to fetch messages"}), 500


@app.route('/api/chat/<conversation_id>', methods=['POST'])
def send_message(conversation_id: str) -> tuple[Any, int]:
    """Send a message and get AI response"""
    try:
        data = request.get_json() or {}
        mode = request.args.get("mode", "default")
        
        # Validate request
        user_content = data.get("content", "").strip()
        if not user_content:
            return jsonify({"error": "Message content is required"}), 400
        
        if not conversation_id in conversations:
            return jsonify({"error": "Conversation not found"}), 404
        
        # Get conversation history
        conv_messages = [
            msg for msg in messages.values()
            if msg.get("conversationId") == conversation_id
        ]
        conv_messages.sort(key=lambda x: x.get("timestamp", ""))
        
        # Build message history for API
        api_messages: List[Dict[str, str]] = [
            {"role": "system", "content": LUMO_SYSTEM_PROMPT}
        ]
        
        # Add previous messages
        for msg in conv_messages:
            api_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Add current user message
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
        
        # Get AI response from Hugging Face
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
                messages=api_messages,
                max_tokens=2000,
                temperature=0.9
            )
            
            ai_response = response.choices[0].message.content or \
                "omg sorry babe 😅 something went wrong... try again?"
        except Exception as hf_error:
            print(f"Hugging Face API error: {str(hf_error)}")
            ai_response = "omg i'm having a moment rn 😭 try again in a sec?"
        
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
        }), 200
    except Exception as e:
        print(f"Chat API error: {str(e)}")
        return jsonify({
            "error": "Failed to process chat message. Check HF_TOKEN and try again."
        }), 500


# ============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id: str) -> tuple[Any, int]:
    """Delete a conversation and all its messages"""
    try:
        # Delete all messages for this conversation
        msgs_to_delete = [
            msg_id for msg_id, msg in messages.items()
            if msg.get("conversationId") == conversation_id
        ]
        for msg_id in msgs_to_delete:
            del messages[msg_id]
        
        # Delete conversation
        if conversation_id in conversations:
            del conversations[conversation_id]
        
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        return jsonify({"error": "Failed to delete conversation"}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.before_request
def before_request():
    """Log incoming requests (development)"""
    if app.debug:
        print(f"{request.method} {request.path}")


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"🚀 Starting LumoAI Backend")
    print(f"📍 Port: {port}")
    print(f"🔐 HF_TOKEN: {'Loaded' if HF_TOKEN else 'NOT SET'}")
    print(f"🌍 CORS: Enabled for GitHub Pages and Render")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
  )

import os
import json
import requests
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import redis
import time
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re
from functools import wraps

DetectorFactory.seed = 0

REDIS_URL = os.environ.get('REDIS_URL_REDIS_URL')
GROQ_MODEL = os.environ.get("GROQ_MODEL")
API_KEY = os.environ.get("GROQ_API_KEY") 
API_URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_EXPIRY_SECONDS = 60 * 60 * 24 * 7 
RATE_LIMIT_SECONDS = 2
MAX_TOKENS_PER_REQUEST = 300
MAX_RETRIES = 5
BASE_DELAY = 0
MAX_TEXT_LENGTH = 2000

r = None
if REDIS_URL:
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception:
        r = None

app = Flask(__name__, static_folder='public')
CORS(app)

class TranslationError(Exception):
    def __init__(self, message, status_code=400, details=None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

class RateLimitError(TranslationError):
    def __init__(self, message="Rate limit exceeded", details=None):
        super().__init__(message, 429, details)

class APIKeyError(TranslationError):
    def __init__(self, message="API key not configured", details=None):
        super().__init__(message, 500, details)

class InputValidationError(TranslationError):
    def __init__(self, message="Invalid input", details=None):
        super().__init__(message, 400, details)

def validate_input(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            data = request.get_json()
            if not data:
                raise InputValidationError("Request must be JSON")
            
            text = data.get('text', '').strip()
            target_lang = data.get('target_lang', '').strip()
            source_lang = data.get('source_lang', '').strip()

            if len(text) > MAX_TEXT_LENGTH:
                raise InputValidationError(
                    f"Text too long. Maximum {MAX_TEXT_LENGTH} characters.",
                    f"Text length: {len(text)} characters"
                )
            
            if not text:
                raise InputValidationError("Text cannot be empty")
                
            if not target_lang:
                raise InputValidationError("Target language cannot be empty")
            
            if not re.match(r'^[A-Za-z\s\-]+$', target_lang):
                raise InputValidationError(
                    "Invalid target language format",
                    "Use only letters, spaces, and hyphens"
                )
            
            if source_lang and not re.match(r'^[A-Za-z\s\-]+$', source_lang):
                raise InputValidationError(
                    "Invalid source language format",
                    "Use only letters, spaces, and hyphens"
                )
                
            return f(*args, **kwargs)
        except InputValidationError as e:
            return jsonify({
                "error": e.message,
                "details": e.details,
                "status_code": e.status_code
            }), e.status_code
        except Exception as e:
            return jsonify({
                "error": "Invalid request format",
                "details": str(e),
                "status_code": 400
            }), 400
    return decorated_function

def check_rate_limit(client_id):
    if not r:
        return True
    
    current_time = time.time()
    rate_limit_key = f"rate_limit:{client_id}"
    token_limit_key = f"token_limit:{client_id}"
    
    try:
        last_request_time = r.get(rate_limit_key)
        if last_request_time:
            time_since_last = current_time - float(last_request_time)
            if time_since_last < RATE_LIMIT_SECONDS:
                return False
        
        token_count = r.get(token_limit_key)
        if token_count and int(token_count) >= MAX_TOKENS_PER_REQUEST:
            return False
        
        r.set(rate_limit_key, current_time, ex=RATE_LIMIT_SECONDS * 2)
        return True
    except Exception:
        return True

def update_token_count(client_id, tokens_used):
    if not r:
        return
    
    try:
        token_limit_key = f"token_limit:{client_id}"
        current_tokens = r.get(token_limit_key)
        if current_tokens:
            new_total = int(current_tokens) + tokens_used
            r.set(token_limit_key, new_total, ex=86400)
        else:
            r.set(token_limit_key, tokens_used, ex=86400)
    except Exception:
        pass

def get_source_language(text):
    try:
        return detect(text).upper()
    except LangDetectException:
        return "Unknown"

def _get_client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()

def call_groq_api_with_backoff(system_instruction, user_prompt):
    current_delay = BASE_DELAY
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_PER_REQUEST
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            data = response.json()
            
            if data and data.get('choices'):
                text_content = data['choices'][0]['message']['content']
                translated_text = text_content.strip()

                if not translated_text:
                    raise ValueError("LLM returned empty translation text.")
                    
                return translated_text
            
            raise ValueError(f"API returned unparseable response structure.")

        except (requests.exceptions.RequestException, requests.exceptions.HTTPError, ValueError) as e:
            if isinstance(e, requests.exceptions.HTTPError) or isinstance(e, requests.exceptions.RequestException):
                 if attempt < MAX_RETRIES - 1:
                    time.sleep(current_delay)
                    current_delay *= 2
                 else:
                    raise
            else:
                raise

    raise Exception("Translation failed after maximum retries.")

def _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False, source_lang_override=None):
    if not check_rate_limit(client_ip):
        raise RateLimitError(
            f"Rate limit exceeded. Try again in {RATE_LIMIT_SECONDS} second(s).",
            f"Client IP: {client_ip}"
        )

    source_language_code = source_lang_override.upper() if source_lang_override else get_source_language(text_to_translate)
    cache_key = f"translation:{text_to_translate}|{target_lang}"
    status_type = "regenerated" if force_refresh else "generated"

    if not force_refresh and r:
        try:
            cached_result_json = r.get(cache_key)
            if cached_result_json:
                cached_result = json.loads(cached_result_json)
                return {
                    "source_language": cached_result['source_language'],
                    "target_language": target_lang,
                    "translated_text": cached_result['translated_text'],
                    "status": "cached"
                }
        except Exception:
            pass

    system_instruction = (
        f"You are a professional, high-quality language translator. Your task is to translate the provided text "
        f"from {source_language_code} to {target_lang}. "
        f"If the text contains slang, abbreviations, or shorthand, make a best effort to translate its intended meaning. "
        f"If the text is truly incomprehensible or nonsensical, provide a literal translation followed by a brief, neutral note in parentheses indicating the uncertainty (e.g., 'Not clear' or 'Abbreviation'). "
        f"IMPORTANT: You MUST ONLY return the translated text and NOTHING else. "
        f"Do not include any introductory phrases, explanations, markdown formatting (like quotes or bolding), or punctuation beyond what is in the translation."
    )
    user_prompt = text_to_translate

    try:
        translated_text = call_groq_api_with_backoff(system_instruction, user_prompt)
        
        tokens_used = len(text_to_translate.split()) + len(translated_text.split())
        update_token_count(client_ip, tokens_used)
        
        response_data = {
            "source_language": source_language_code,
            "target_language": target_lang,
            "translated_text": translated_text,
            "status": status_type
        }
        
        if r:
            try:
                cache_data = {
                    "source_language": source_language_code,
                    "translated_text": translated_text,
                    "timestamp": time.time(),
                    "model": GROQ_MODEL
                }
                r.set(cache_key, json.dumps(cache_data), ex=CACHE_EXPIRY_SECONDS)
            except Exception:
                pass
        
        return response_data

    except ValueError as ve:
        raise TranslationError(
            "The model struggled to generate a translation. The input may be too short or ambiguous.",
            details=str(ve)
        )

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            error_details = e.response.json().get('error', {}).get('message', 'Unknown API Error')
        except:
            error_details = 'Could not parse error details from Groq.'
        raise TranslationError(
            f"Groq API Error ({status_code})",
            status_code=status_code,
            details=error_details
        )

    except Exception as e:
        raise TranslationError(
            "An internal server error occurred",
            status_code=500,
            details=str(e)
        )

@app.errorhandler(TranslationError)
def handle_translation_error(error):
    response = jsonify({
        "error": error.message,
        "details": error.details,
        "status_code": error.status_code
    })
    response.status_code = error.status_code
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "redis": "connected" if r and r.ping() else "disconnected",
        "groq_api": "configured" if API_KEY else "not_configured",
        "max_text_length": MAX_TEXT_LENGTH,
        "rate_limit": f"{RATE_LIMIT_SECONDS} second(s)",
        "max_tokens_per_request": MAX_TOKENS_PER_REQUEST,
        "timestamp": time.time()
    }
    return jsonify(health_status), 200

@app.route('/api/translate', methods=['POST'])
@validate_input
def translate():
    client_ip = _get_client_ip()
    
    if not API_KEY:
        raise APIKeyError()

    data = request.get_json()
    text_to_translate = data.get('text', '').strip()
    target_lang = data.get('target_lang', '').strip()
    source_lang = data.get('source_lang', '').strip()

    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=False, source_lang_override=source_lang or None)
    return jsonify(result), 200

@app.route('/api/retranslate', methods=['POST'])
@validate_input
def retranslate():
    client_ip = _get_client_ip()
    
    if not API_KEY:
        raise APIKeyError()

    data = request.get_json()
    text_to_translate = data.get('text', '').strip()
    target_lang = data.get('target_lang', '').strip()
    source_lang = data.get('source_lang', '').strip()

    result = _process_translation(text_to_translate, target_lang, client_ip, force_refresh=True, source_lang_override=source_lang or None)
    return jsonify(result), 200

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('public', path)):
        return send_from_directory('public', path)
    abort(404)

@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

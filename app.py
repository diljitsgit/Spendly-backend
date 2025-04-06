from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from database import (
    get_db_connection, 
    init_db,
    create_budget, 
    get_user_budgets, 
    get_budget_progress,
    create_financial_goal, 
    update_financial_goal, 
    get_user_goals,
    create_user,
    get_user_by_username
)
import random
import re
import html
from functools import wraps
from collections import defaultdict
import time
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Rate limiting configuration
RATE_LIMIT = 60  # requests per minute
rate_limit_data = defaultdict(list)

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP
        ip = request.remote_addr
        now = time.time()
        
        # Clean old requests
        rate_limit_data[ip] = [t for t in rate_limit_data[ip] if now - t < 60]
        
        # Check rate limit
        if len(rate_limit_data[ip]) >= RATE_LIMIT:
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        
        # Add current request
        rate_limit_data[ip].append(now)
        return f(*args, **kwargs)
    return decorated_function

# Load model and encoders
base_dir = Path(__file__).parent
models_dir = base_dir / "models"

# Initialize variables with defaults
model = None
label_encoder = None
selected_features = ['amount', 'amount_log', 'amount_squared', 'amount_binned', 'is_weekend', 
                    'day_of_week', 'month', 'quarter', 'day_of_month', 'category']

try:
    with open(models_dir / "xgb_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(models_dir / "selected_features.pkl", "rb") as f:
        loaded_features = pickle.load(f)
        if isinstance(loaded_features, (list, np.ndarray)):
            selected_features = loaded_features
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.warning("Using default configuration")
    # Don't raise the error, continue with defaults

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'spendly.db')

# Database connection helper
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    """API Home with documentation"""
    return jsonify({
        'name': 'Spendly Financial Management API',
        'version': '1.0.0',
        'status': 'online',
        'endpoints': {
            'user': {
                'POST /user': 'Create a new user profile',
                'required_fields': ['username', 'email']
            },
            'budget': {
                'POST /budget': 'Create a new budget',
                'required_fields': ['user_id', 'category', 'amount', 'period', 'start_date'],
                'GET /budget/:user_id': 'Get all budgets for a user',
                'GET /budget/progress/:user_id': 'Get budget progress (optional query param: category)'
            },
            'goals': {
                'POST /goal': 'Create a new financial goal',
                'required_fields': ['user_id', 'goal_name', 'target_amount', 'deadline'],
                'GET /goal/:user_id': 'Get all financial goals for a user',
                'PUT /goal/:user_id/:goal_id': 'Update a financial goal progress'
            },
            'transactions': {
                'POST /transaction': 'Record a new transaction',
                'required_fields': ['user_id', 'item', 'category', 'amount'],
                'optional_fields': ['date', 'label'],
                'GET /transaction/:user_id': 'Get recent transactions (optional query params: limit, category)'
            },
            'dashboard': {
                'GET /dashboard/:user_id': 'Get comprehensive dashboard data including spending, budgets, goals and insights',
                'optional_params': ['period (week/month/year)']
            },
            'chat': {
                'POST /chat': 'Get AI-powered financial advice',
                'required_fields': ['user_id', 'message']
            },
            'debug': {
                'GET /debug/routes': 'List all API routes'
            }
        },
        'usage_examples': {
            'create_user': {
                'request': {'username': 'johndoe', 'email': 'john@example.com'},
                'response': {'message': 'User created successfully', 'user_id': 1}
            },
            'create_budget': {
                'request': {'user_id': 1, 'category': 'Food', 'amount': 10000, 'period': 'monthly', 'start_date': '2023-01-01'},
                'response': {'message': 'Budget created successfully'}
            },
            'create_transaction': {
                'request': {'user_id': 1, 'item': 'Groceries', 'category': 'Food', 'amount': 2500},
                'response': {'status': 'success', 'message': 'Transaction recorded successfully', 'transaction_id': 1}
            },
            'get_dashboard': {
                'request': 'GET /dashboard/1?period=month',
                'response': {'status': 'success', 'dashboard': {'summary': {}, 'spending_by_category': [], 'budget_progress': {}, 'goals': [], 'insights': []}}
            },
            'chat': {
                'request': {'user_id': 1, 'message': 'Should I spend ₹5000 on a new phone?'},
                'response': {'status': 'success', 'response': '...', 'expense_details': {...}}
            }
        }
    })

@app.route('/user', methods=['POST'])
def create_user():
    """Create a new user"""
    try:
        data = request.get_json()
        required_fields = ['username', 'email']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        password = data.get('password', '')
        
        from database import create_user
        success, result = create_user(data['username'], data['email'], password)

        if success:
            return jsonify({
                'status': 'success',
                'message': 'User created successfully',
                'user_id': result
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result
            }), 400
            
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({'error': 'Username is required'}), 400
            
        username = data['username']
        password = data.get('password', '')
        
        from database import get_user_by_username
        user = get_user_by_username(username)
        
        if not user:
            return jsonify({
                'status': 'error',
                'error': 'User not found'
            }), 404
            
        # Basic password check - in a real app, use secure password hashing!
        if user['password'] != password:
            return jsonify({
                'status': 'error',
                'error': 'Invalid password'
            }), 401
            
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        })
            
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/budget', methods=['POST'])
def create_budget_endpoint():
    """Create a new budget for a user"""
    try:
        data = request.get_json()
        required_fields = ['user_id', 'category', 'amount', 'period', 'start_date']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        success = create_budget(
            data['user_id'],
            data['category'],
            data['amount'],
            data['period'],
            data['start_date'],
            data.get('end_date')
        )
        
        if success:
            return jsonify({'message': 'Budget created successfully'})
        else:
            return jsonify({'error': 'Failed to create budget'}), 500
            
    except Exception as e:
        logger.error(f"Error creating budget: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/budget/<int:user_id>', methods=['GET'])
def get_budgets_endpoint(user_id):
    """Get all budgets for a user"""
    try:
        budgets = get_user_budgets(user_id)
        return jsonify({'budgets': [dict(budget) for budget in budgets]})
    except Exception as e:
        logger.error(f"Error getting budgets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/budget/progress/<int:user_id>', methods=['GET'])
def get_budget_progress_endpoint(user_id):
    """Get budget progress for a user"""
    try:
        category = request.args.get('category')
        progress = get_budget_progress(user_id, category)
        return jsonify(progress)
    except Exception as e:
        logger.error(f"Error getting budget progress: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/goal', methods=['POST'])
def create_goal_endpoint():
    """Create a new financial goal"""
    try:
        data = request.get_json()
        required_fields = ['user_id', 'goal_name', 'target_amount', 'deadline']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        success = create_financial_goal(
            data['user_id'],
            data['goal_name'],
            data['target_amount'],
            data['deadline']
        )
        
        if success:
            return jsonify({'message': 'Financial goal created successfully'})
        else:
            return jsonify({'error': 'Failed to create financial goal'}), 500
            
    except Exception as e:
        logger.error(f"Error creating financial goal: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/goal/<int:user_id>', methods=['GET'])
def get_goals_endpoint(user_id):
    """Get all financial goals for a user"""
    try:
        goals = get_user_goals(user_id)
        return jsonify({'goals': [dict(goal) for goal in goals]})
    except Exception as e:
        logger.error(f"Error getting financial goals: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/goal/<int:user_id>/<int:goal_id>', methods=['PUT'])
def update_goal_endpoint(user_id, goal_id):
    """Update progress of a financial goal"""
    try:
        data = request.get_json()
        if 'current_amount' not in data:
            return jsonify({'error': 'Missing current_amount field'}), 400
        
        success = update_financial_goal(user_id, goal_id, data['current_amount'])
        
        if success:
            return jsonify({'message': 'Financial goal updated successfully'})
        else:
            return jsonify({'error': 'Failed to update financial goal'}), 500
            
    except Exception as e:
        logger.error(f"Error updating financial goal: {e}")
        return jsonify({'error': str(e)}), 500

def preprocess_input(expense_data):
    """Preprocess input data to match training data format"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([expense_data])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract date features
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month_name()
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        df['quarter'] = df['date'].dt.quarter
        
        # Create amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        
        # Handle binning for single values (adjusted for INR)
        amount_bins = [0, 10000, 25000, 50000, 100000, float('inf')]
        df['amount_binned'] = pd.cut(df['amount'], bins=amount_bins, labels=False)
        
        age_bins = [0, 20, 30, 40, 50, float('inf')]
        df['age_binned'] = pd.cut(df['age'], bins=age_bins, labels=False)
        
        # Create interaction features
        df['age_amount_interaction'] = df['age'] * df['amount']
        
        # Convert categorical columns to numeric codes
        categorical_cols = ['day_of_week', 'month', 'gender', 'quarter', 'category']
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col]).codes
        
        # Drop unnecessary columns
        df = df.drop(['date', 'item'], axis=1)
        
        # Ensure all selected features are present
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[selected_features]
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def parse_expense_message(message):
    """Parse user message to extract amount and item efficiently"""
    # Normalize message for consistent processing
    message = message.lower().strip()
    
    # Common currency symbols and patterns
    currency_patterns = r'[$₹€£¥₽₩]|rs\.?|rupees|dollars|inr|usd|eur|gbp'
    
    # Remove currency symbols for cleaner processing
    cleaned_message = re.sub(currency_patterns, '', message)
    
    # Look for numbers in the message - optimized regex
    amount_pattern = r'\b\d+[,.]?\d*\b'
    numbers = re.findall(amount_pattern, cleaned_message)
    
    if not numbers:
        return None, None

    # Get the largest number as the amount
    amount = max([float(n.replace(',', '')) for n in numbers])
    
    # Extract the item description - multiple patterns for better accuracy
    # Pattern 1: "for <item>" pattern
    for_pattern = re.search(r'for\s+([^?.]+)', message)
    if for_pattern:
        return amount, for_pattern.group(1).strip()
    
    # Pattern 2: "on <item>" pattern
    on_pattern = re.search(r'on\s+([^?.]+)', message)
    if on_pattern:
        return amount, on_pattern.group(1).strip()
    
    # Pattern 3: "<item> for/costs <amount>" pattern
    item_first_pattern = re.search(r'([a-z\s]+)\s+(?:for|costs|is|are)\s+[\d$₹€£]', message)
    if item_first_pattern:
        return amount, item_first_pattern.group(1).strip()
    
    # Pattern 4: "<amount> for <item>" pattern
    amount_str = str(amount)
    amount_idx = message.find(amount_str)
    if amount_idx >= 0:
        after_amount = message[amount_idx + len(amount_str):].strip()
        
        # Look for prepositions after the amount
        for prep in ['for', 'on', 'to buy', 'to purchase', 'to get']:
            if prep in after_amount:
                item_start = after_amount.find(prep) + len(prep)
                return amount, after_amount[item_start:].strip()
    
    # Fallback: Get any words after the amount
    words = cleaned_message.split()
    if numbers and len(words) > 1:
        for i, word in enumerate(words):
            if re.match(r'\d+[,.]?\d*', word):
                if i < len(words) - 1:
                    return amount, ' '.join(words[i+1:])
    
    return amount, None

def determine_category(item):
    """Determine the category of an item - optimized with efficient category mapping"""
    item = item.lower()
    
    # Category mapping with keywords
    category_map = {
        'Food': ['food', 'grocery', 'groceries', 'meal', 'fruit', 'vegetable', 
                'restaurant', 'dining', 'lunch', 'dinner', 'breakfast', 'snack',
                'pizza', 'burger', 'meat', 'drinks', 'coffee', 'tea'],
                
        'Entertainment': ['movie', 'entertainment', 'game', 'netflix', 'subscription', 'streaming',
                        'music', 'concert', 'show', 'theater', 'sport', 'hobby', 'book', 
                        'party', 'festival', 'event', 'ticket', 'club'],
                        
        'Electronics': ['phone', 'laptop', 'computer', 'gadget', 'camera', 'headphone', 
                        'speaker', 'tablet', 'tv', 'television', 'game console', 'electronic',
                        'charger', 'accessory', 'keyboard', 'mouse', 'monitor', 'printer'],
                        
        'Clothing': ['cloth', 'shirt', 'pant', 'dress', 'shoe', 'outfit', 'fashion', 
                    'jacket', 'coat', 'hat', 'accessory', 'jewelry', 'watch'],
                    
        'Transportation': ['transport', 'travel', 'commute', 'taxi', 'bus', 'train', 'flight',
                         'car', 'bike', 'fuel', 'gas', 'petrol', 'diesel', 'ticket', 'fare'],
                         
        'Healthcare': ['health', 'medical', 'medicine', 'doctor', 'hospital', 'dental', 
                      'therapy', 'insurance', 'fitness', 'gym', 'workout', 'wellness'],
                      
        'Housing': ['rent', 'house', 'apartment', 'furniture', 'repair', 'maintenance',
                  'decor', 'utility', 'bill', 'electricity', 'water', 'internet'],
                  
        'Education': ['education', 'course', 'class', 'tuition', 'book', 'school', 
                    'college', 'university', 'degree', 'training', 'skill', 'learning'],
                    
        'Financial': ['investment', 'bank', 'fee', 'loan', 'interest', 'tax', 'insurance',
                    'saving', 'finance', 'debt', 'credit', 'subscription']
    }
    
    # Check item against each category's keywords
    for category, keywords in category_map.items():
        if any(keyword in item for keyword in keywords):
            return category
    
    # Default category for unclassified items
    return "Miscellaneous"

def engineer_features(amount, category=None):
    """Engineer model features - cached computation for better performance"""
    # Create a feature vector that matches the model's expected format
    features = {}
    
    # Amount-based features
    features['amount'] = amount
    features['amount_log'] = np.log1p(amount)  # Log transform to handle skewness
    features['amount_squared'] = amount ** 2  # Capture non-linear relationships
    
    # Amount binning (0-1000, 1000-5000, 5000-10000, 10000+)
    if amount < 1000:
        bin_value = 0
    elif amount < 5000:
        bin_value = 1
    elif amount < 10000:
        bin_value = 2
    else:
        bin_value = 3
    features['amount_binned'] = bin_value
    
    # Time-based features
    now = datetime.now()
    features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    features['day_of_week'] = now.weekday()
    features['month'] = now.month
    features['quarter'] = (now.month - 1) // 3 + 1
    features['day_of_month'] = now.day
    
    # Category encoding (if provided)
    features['category'] = category or "Miscellaneous"
    
    # Create a DataFrame with the right column names expected by the model
    df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Convert categorical features using one-hot encoding
    if 'category' in df.columns and label_encoder is not None:
        try:
            # Map known categories
            if df['category'].iloc[0] in label_encoder.classes_:
                df['category'] = label_encoder.transform([df['category'].iloc[0]])[0]
            else:
                # Handle unknown categories gracefully
                df['category'] = len(label_encoder.classes_) - 1  # Use last class as 'other'
        except Exception as e:
            logger.error(f"Error encoding category: {e}")
            df['category'] = 0  # Default to first category on error
    
    # Select only the features used by the model
    if selected_features and all(f in df.columns for f in selected_features):
        df = df[selected_features]
    
    return df

def get_model_prediction(amount, category=None):
    """Enhanced model prediction with performance optimizations and robust fallback"""
    # Define robust fallback categories
    fallback_categories = {
        'Food': {'type': 'Need', 'confidence': 0.92},
        'Healthcare': {'type': 'Need', 'confidence': 0.95},
        'Housing': {'type': 'Need', 'confidence': 0.95},
        'Transportation': {'type': 'Need', 'confidence': 0.85},
        'Education': {'type': 'Need', 'confidence': 0.80},
        'Electronics': {'type': 'Want', 'confidence': 0.75},
        'Entertainment': {'type': 'Want', 'confidence': 0.90},
        'Clothing': {'type': 'Need', 'confidence': 0.65},
        'Financial': {'type': 'Need', 'confidence': 0.80},
        'Miscellaneous': {'type': 'Want', 'confidence': 0.60}
    }
    
    # Use model prediction if possible
    if model is not None:
        try:
            # Get engineered features
            features = engineer_features(amount, category)
            
            # Make prediction
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Extract highest probability
            max_prob_idx = np.argmax(probabilities[0])
            confidence = probabilities[0][max_prob_idx]
            label = prediction[0]
            
            # If confidence is high enough, return model prediction
            if confidence >= 0.60:
                return label, confidence
                
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}. Using fallback logic.")
            # Continue to fallback logic
    
    # Fallback logic
    if category in fallback_categories:
        fallback = fallback_categories[category]
        return fallback['type'], fallback['confidence']
        
    # Amount-based heuristic for unknown categories
    if amount < 1000:
        return 'Need', 0.65  # Small amounts more likely to be needs
    elif amount > 15000:
        return 'Want', 0.70  # Large amounts more likely to be wants
    else:
        return 'Need', 0.55  # Default to need with low confidence

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to interact with the AI assistant"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data or 'userId' not in data:
            return jsonify({
                "success": False,
                "message": "Missing required fields: message and userId"
            }), 400
        
        user_message = data['message']
        user_id = data['userId']
        
        # Get response from our enhanced AI assistant
        response = get_financial_advice(user_id, user_message)
        
        # Save the conversation
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Save user message
        cursor.execute(
            "INSERT INTO chat_history (user_id, sender, message, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (user_id, 'user', user_message)
        )
        
        # Save assistant response
        cursor.execute(
            "INSERT INTO chat_history (user_id, sender, message, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (user_id, 'assistant', response)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "Message processed successfully",
            "response": response
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "success": False,
            "message": "An error occurred while processing your message"
        }), 500

def sanitize_input(text):
    """Sanitize user input to prevent XSS and other injection attacks"""
    if not isinstance(text, str):
        return ""
    # Remove any HTML/XML tags
    text = html.escape(text)
    # Remove any potential SQL injection attempts
    text = text.replace("'", "''")
    # Remove any command injection attempts
    text = re.sub(r'[;&|]', '', text)
    return text

def analyze_budget_impact(user_id, amount, category):
    """Enhanced budget impact analysis"""
    try:
        # Get current budget information
        budget_info = get_budget_progress(user_id, category)
        if not budget_info:
            return None, "No budget set for this category"
        
        remaining = budget_info.get('remaining', 0)
        total_budget = budget_info.get('total', 0)
        spent = budget_info.get('spent', 0)
        
        # Calculate percentage of budget
        if total_budget > 0:
            percent_of_budget = (amount / total_budget) * 100
            percent_spent = (spent / total_budget) * 100
        else:
            return None, "Invalid budget amount"
        
        # Analyze timing
        current_day = datetime.now().day
        days_in_month = 30  # Simplified
        time_factor = current_day / days_in_month
        
        # Determine impact level
        if amount > remaining:
            impact = "high"
        elif percent_of_budget > 50:
            impact = "significant"
        elif percent_of_budget > 25:
            impact = "moderate"
        else:
            impact = "low"
        
        # Generate detailed analysis
        analysis = {
            'impact_level': impact,
            'amount_vs_budget': f"{percent_of_budget:.1f}%",
            'budget_remaining': remaining,
            'month_progress': f"{time_factor:.1%}",
            'spending_rate': "ahead" if percent_spent > (time_factor * 100) else "behind",
            'recommendation': ""
        }
        
        # Add specific recommendations
        if impact == "high":
            analysis['recommendation'] = "Consider postponing or reducing this expense"
        elif impact == "significant":
            analysis['recommendation'] = "Look for ways to optimize this expense"
        elif analysis['spending_rate'] == "ahead":
            analysis['recommendation'] = "Consider spacing out expenses more evenly"
        else:
            analysis['recommendation'] = "This expense appears manageable"
            
        return analysis, None
        
    except Exception as e:
        logger.error(f"Error in budget analysis: {e}")
        return None, str(e)

def is_greeting(message):
    """Enhanced check if message is a greeting"""
    message = message.lower().strip()
    greetings = [
        'hi', 'hello', 'hey', 'howdy', 'greetings', 'good morning', 'good afternoon', 
        'good evening', 'hi there', 'hello there', 'namaste', 'hola', 'wassup', 
        'what\'s up', 'yo', 'sup', 'morning', 'evening', 'afternoon'
    ]
    
    # Check for exact matches
    if message in greetings:
        return True
    
    # Check for greeting at the beginning
    for g in greetings:
        if message.startswith(g):
            return True
    
    # Check for greeting patterns
    greeting_patterns = [
        r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bhowdy\b', r'\bgreetings\b',
        r'good \w+', r'\byo\b', r'\bsup\b', r'what\'?s up'
    ]
    
    return any(re.search(pattern, message) for pattern in greeting_patterns)

def is_help_request(message):
    """Enhanced check if message is asking for help"""
    message = message.lower().strip()
    help_phrases = [
        'help', 'assist', 'guide', 'how to', 'how do i', 'what can you do', 
        'instructions', 'features', 'capabilities', 'tell me about', 'explain',
        'show me how', 'teach me', 'guide me', 'support', 'tutorial',
        'what do you do', 'what are you for', 'how does this work',
        'who are you', 'what are you', 'about you', 'your purpose',
        'how do you help', 'how can i use'
    ]
    
    for phrase in help_phrases:
        if phrase in message:
            return True
    
    help_patterns = [
        r'(how|what) (can|do) (i|you|we)', r'help me (with|to)', 
        r'need (help|assistance)', r'tell me (how|about)', 
        r'(show|explain) (me|to me)'
    ]
    
    return any(re.search(pattern, message) for pattern in help_patterns)

def is_expense_query(message):
    """Enhanced check if message is about an expense"""
    message = message.lower().strip()
    
    # Contains amount pattern - different currency formats
    amount_patterns = [
        r'₹\s*\d+[,.]?\d*', # ₹2000
        r'rs\.?\s*\d+[,.]?\d*', # Rs. 2000
        r'rupees?\s*\d+[,.]?\d*', # Rupees 2000
        r'inr\s*\d+[,.]?\d*', # INR 2000
        r'\d+[,.]?\d*\s*rupees?', # 2000 rupees
        r'\d+[,.]?\d*\s*rs\.?', # 2000 Rs.
        r'\d+[,.]?\d*\s*inr', # 2000 INR
        r'\d+[,.]?\d*\s*₹', # 2000 ₹
        r'\b\d+[,.]?\d*k\b', # 2k
        r'\b\d+[,.]?\d*\b' # Just numbers
    ]
    
    has_amount = any(re.search(pattern, message) for pattern in amount_patterns)
    
    # Purchase-related words and phrases
    purchase_words = [
        'buy', 'purchase', 'spend', 'cost', 'price', 'expensive', 'cheap', 'worth', 'afford',
        'pay', 'paying', 'payment', 'save', 'savings', 'investment', 'buying', 'spending',
        'pricing', 'value', 'deal', 'bargain', 'sale', 'discount', 'offer', 'money', 'cash',
        'budget', 'fees', 'charge', 'transaction', 'bill', 'billing', 'subscription',
        'upgrade', 'premium', 'invest', 'investing', 'mortgage', 'loan', 'rent', 'lease',
        'emi', 'installment', 'should i get', 'is it good', 'right choice', 'good idea'
    ]
    
    purchase_patterns = [
        r'(should|could|would|can) (i|we) (buy|get|purchase|spend)',
        r'(is|are) (it|they|this|that) (worth|expensive|costly|cheap)',
        r'(how much) (is|are|does) (it|this|that) (cost)',
        r'(want|need|going) to (buy|purchase|get|spend)',
        r'(thinking|considering) (about|of) (buying|getting|purchasing)'
    ]
    
    has_purchase_word = any(word in message for word in purchase_words) or any(re.search(pattern, message) for pattern in purchase_patterns)
    
    # If we detect a purchase word, check for product keywords even without amount
    product_keywords = [
        'phone', 'laptop', 'computer', 'tv', 'television', 'refrigerator', 'fridge',
        'car', 'bike', 'vehicle', 'furniture', 'house', 'apartment', 'property',
        'clothes', 'shoes', 'jewelry', 'accessory', 'subscription', 'service',
        'education', 'course', 'class', 'training', 'food', 'restaurant', 'groceries',
        'vacation', 'trip', 'travel', 'holiday', 'flight', 'hotel', 'resort'
    ]
    
    has_product = any(keyword in message for keyword in product_keywords)
    
    return (has_amount and has_purchase_word) or (has_purchase_word and has_product)

def is_budget_query(message):
    """Check if message is asking about budget"""
    message = message.lower().strip()
    budget_phrases = [
        'budget', 'spending limit', 'how much can i spend', 'afford', 'available funds',
        'how much do i have', 'balance', 'remaining', 'left', 'allocation',
        'money left', 'funds', 'financial status', 'financial health', 'financial overview',
        'spending overview', 'spending summary', 'budget status', 'budget overview',
        'budget summary', 'budget report', 'spending report', 'financial report'
    ]
    
    budget_patterns = [
        r'(how much) (money|funds|cash) (do i have|is left|remains)',
        r'(what is|what\'s) (my|the) (budget|spending limit|balance)',
        r'(can i afford|do i have enough for)',
        r'(how am i doing|how\'s my) (budget|spending|finances)',
        r'(show|tell) me (my|about my) (budget|spending|finances)'
    ]
    
    category_keywords = [
        'food', 'groceries', 'restaurant', 'dining', 'entertainment', 'shopping',
        'transportation', 'travel', 'housing', 'rent', 'utilities', 'bills',
        'healthcare', 'education', 'electronics', 'personal', 'leisure'
    ]
    
    has_budget_phrase = any(phrase in message for phrase in budget_phrases)
    has_budget_pattern = any(re.search(pattern, message) for pattern in budget_patterns)
    has_category = any(category in message for category in category_keywords)
    
    return has_budget_phrase or has_budget_pattern or (has_category and ('how much' in message or 'budget' in message))

def is_goal_query(message):
    """Check if message is asking about financial goals"""
    message = message.lower().strip()
    goal_phrases = [
        'goal', 'target', 'saving for', 'saving up', 'plan to buy', 'planning to buy',
        'saving goal', 'financial goal', 'financial target', 'saving target',
        'milestone', 'financial plan', 'financial milestone', 'financial objective',
        'financial aim', 'financial ambition', 'financial aspiration'
    ]
    
    goal_patterns = [
        r'(how) (am i doing|close am i) (with|on) (my|the) (goal|target|saving)',
        r'(how much) (have i saved|do i need|is left) (for|to reach)',
        r'(when) (can i|will i be able to) (reach|achieve|attain|buy|get)',
        r'(what is|what\'s) (my|the) (goal|target|progress|status)',
        r'(show|tell) me (my|about my) (goals|targets|savings)'
    ]
    
    specific_goals = [
        'house', 'home', 'car', 'vehicle', 'education', 'college',
        'laptop', 'computer', 'phone', 'vacation', 'trip', 'travel',
        'wedding', 'marriage', 'retirement', 'emergency fund'
    ]
    
    has_goal_phrase = any(phrase in message for phrase in goal_phrases)
    has_goal_pattern = any(re.search(pattern, message) for pattern in goal_patterns)
    has_specific_goal = any(goal in message for goal in specific_goals)
    
    return has_goal_phrase or has_goal_pattern or (has_specific_goal and ('saving' in message or 'goal' in message))

def get_financial_advice(user_id, message):
    """Get personalized financial advice based on the user's message and financial data"""
    try:
        # Handle different types of queries
        if is_greeting(message):
            greetings = [
                "Hello! How can I assist with your finances today?",
                "Hi there! Ready to help you manage your money better.",
                "Hey! What financial questions do you have today?",
                "Welcome back! Let's talk about your financial goals.",
                "Greetings! I'm here to help with your personal finance questions."
            ]
            return random.choice(greetings)
        
        if is_help_request(message):
            return """I can help you with:
1. Tracking expenses - just tell me what you bought and for how much
2. Budget monitoring - ask about your budget or spending in categories
3. Financial goals - check your progress toward savings goals
4. Purchase advice - ask if something is worth buying
5. Financial tips - general advice for better money management

Try asking something like "How much have I spent on food?" or "I spent ₹2000 on groceries"."""
        
        if is_budget_query(message):
            # Get the user's budget
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Extract category if present
            categories = [
                'food', 'groceries', 'dining', 'restaurant', 'entertainment', 
                'shopping', 'transportation', 'travel', 'housing', 'rent', 
                'utilities', 'bills', 'healthcare', 'education', 'electronics'
            ]
            
            category = None
            for cat in categories:
                if cat in message.lower():
                    category = cat
                    break
                    
            if category:
                # Map to our standard categories
                category_mapping = {
                    'groceries': 'Food', 'food': 'Food', 'dining': 'Food', 'restaurant': 'Food',
                    'entertainment': 'Entertainment', 
                    'shopping': 'Shopping',
                    'transportation': 'Transportation', 'travel': 'Transportation',
                    'housing': 'Housing', 'rent': 'Housing',
                    'utilities': 'Utilities', 'bills': 'Utilities',
                    'healthcare': 'Healthcare',
                    'education': 'Education',
                    'electronics': 'Shopping'
                }
                
                standard_category = category_mapping.get(category, 'Miscellaneous')
                
                # Get budget for specific category
                cursor.execute(
                    "SELECT budgets.amount, SUM(transactions.amount) as spent FROM budgets "
                    "LEFT JOIN transactions ON transactions.category = budgets.category AND transactions.user_id = budgets.user_id "
                    "WHERE budgets.user_id = ? AND budgets.category = ? "
                    "GROUP BY budgets.category",
                    (user_id, standard_category)
                )
                result = cursor.fetchone()
                
                if result:
                    budget_amount, spent = result
                    if spent is None:
                        spent = 0
                    remaining = budget_amount - spent
                    percentage = (spent / budget_amount) * 100 if budget_amount > 0 else 0
                    
                    if percentage > 90:
                        status = "You've almost exhausted your budget!"
                    elif percentage > 75:
                        status = "You're approaching your budget limit."
                    elif percentage > 50:
                        status = "You've used more than half of your budget."
                    else:
                        status = "You're well within your budget."
                    
                    return f"For {standard_category}, your budget is ₹{budget_amount:.2f}. You've spent ₹{spent:.2f} ({percentage:.1f}%) and have ₹{remaining:.2f} remaining. {status}"
                else:
                    return f"I couldn't find a budget for {category}. Would you like to set one up?"
            else:
                # Get overall budget status
                cursor.execute(
                    "SELECT SUM(budgets.amount) as total_budget, SUM(transactions.amount) as total_spent "
                    "FROM budgets LEFT JOIN transactions ON transactions.user_id = budgets.user_id "
                    "WHERE budgets.user_id = ? GROUP BY budgets.user_id", 
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    total_budget, total_spent = result
                    if total_spent is None:
                        total_spent = 0
                    remaining = total_budget - total_spent
                    percentage = (total_spent / total_budget) * 100 if total_budget > 0 else 0
                    
                    # Get top spending categories
                    cursor.execute(
                        "SELECT category, SUM(amount) as category_total FROM transactions "
                        "WHERE user_id = ? GROUP BY category ORDER BY category_total DESC LIMIT 3",
                        (user_id,)
                    )
                    top_categories = cursor.fetchall()
                    
                    top_spending = ""
                    if top_categories:
                        top_spending = "Your top spending categories are: "
                        for i, (cat, amount) in enumerate(top_categories):
                            top_spending += f"{cat} (₹{amount:.2f})"
                            if i < len(top_categories) - 1:
                                top_spending += ", "
                    
                    return f"Your total budget is ₹{total_budget:.2f}. You've spent ₹{total_spent:.2f} ({percentage:.1f}%) and have ₹{remaining:.2f} remaining. {top_spending}"
                else:
                    return "I couldn't find any budget information. Would you like to set up a budget?"

        if is_goal_query(message):
            # Get the user's financial goals
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Extract specific goal if mentioned
            goals = [
                'house', 'home', 'car', 'vehicle', 'education', 'college',
                'laptop', 'computer', 'phone', 'vacation', 'trip', 'travel',
                'wedding', 'marriage', 'retirement', 'emergency'
            ]
            
            specific_goal = None
            for goal in goals:
                if goal in message.lower():
                    specific_goal = goal
                    break
                    
            if specific_goal:
                # Map to our standard goal types
                goal_mapping = {
                    'house': 'Home', 'home': 'Home',
                    'car': 'Vehicle', 'vehicle': 'Vehicle',
                    'education': 'Education', 'college': 'Education',
                    'laptop': 'Electronics', 'computer': 'Electronics', 'phone': 'Electronics',
                    'vacation': 'Travel', 'trip': 'Travel', 'travel': 'Travel',
                    'wedding': 'Wedding', 'marriage': 'Wedding',
                    'retirement': 'Retirement',
                    'emergency': 'Emergency Fund'
                }
                
                standard_goal = goal_mapping.get(specific_goal, specific_goal.capitalize())
                
                # Find goal that matches
                cursor.execute(
                    "SELECT id, name, target_amount, current_amount, target_date FROM goals "
                    "WHERE user_id = ? AND name LIKE ?",
                    (user_id, f"%{standard_goal}%")
                )
                goal = cursor.fetchone()
                
                if goal:
                    goal_id, name, target, current, target_date = goal
                    remaining = target - current
                    percentage = (current / target) * 100 if target > 0 else 0
                    
                    # Calculate time remaining
                    target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
                    today = datetime.now()
                    days_remaining = (target_date_obj - today).days
                    
                    if days_remaining > 0:
                        time_status = f"You have {days_remaining} days remaining to reach your goal."
                        # Calculate required savings per month
                        months_remaining = max(1, days_remaining / 30)
                        required_monthly = remaining / months_remaining
                        saving_advice = f"You need to save ₹{required_monthly:.2f} per month to reach your goal on time."
                    else:
                        time_status = "You've passed your target date."
                        saving_advice = "Consider updating your goal with a new target date."
                    
                    return f"For your {name} goal: You've saved ₹{current:.2f} of your ₹{target:.2f} target ({percentage:.1f}%). {time_status} {saving_advice}"
                else:
                    return f"I couldn't find a goal related to {specific_goal}. Would you like to set one up?"
            else:
                # Get all goals summary
                cursor.execute(
                    "SELECT name, target_amount, current_amount FROM goals WHERE user_id = ?",
                    (user_id,)
                )
                goals = cursor.fetchall()
                
                if goals:
                    response = "Here's a summary of your financial goals:\n"
                    for name, target, current in goals:
                        percentage = (current / target) * 100 if target > 0 else 0
                        response += f"- {name}: ₹{current:.2f}/₹{target:.2f} ({percentage:.1f}%)\n"
                    return response
                else:
                    return "You don't have any financial goals set up yet. Would you like to create one?"
        
        if is_expense_query(message):
            # Extract amount and item from the message
            amount_patterns = [
                r'₹\s*(\d+[,.]?\d*)', # ₹2000
                r'rs\.?\s*(\d+[,.]?\d*)', # Rs. 2000
                r'rupees?\s*(\d+[,.]?\d*)', # Rupees 2000
                r'inr\s*(\d+[,.]?\d*)', # INR 2000
                r'(\d+[,.]?\d*)\s*rupees?', # 2000 rupees
                r'(\d+[,.]?\d*)\s*rs\.?', # 2000 Rs.
                r'(\d+[,.]?\d*)\s*inr', # 2000 INR
                r'(\d+[,.]?\d*)\s*₹', # 2000 ₹
                r'\b(\d+[,.]?\d*)k\b', # 2k (multiply by 1000)
                r'\b(\d+[,.]?\d*)\b' # Just numbers
            ]
            
            amount = None
            for pattern in amount_patterns:
                match = re.search(pattern, message)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 'k' in pattern:
                        amount *= 1000
                    break
            
            if not amount:
                return "I couldn't identify the expense amount. Please specify an amount, like 'I spent ₹2000 on groceries'."
            
            # Try to identify what the expense is for
            words = message.lower().split()
            prepositions = ['for', 'on', 'at', 'in', 'to']
            item = None
            
            for prep in prepositions:
                if prep in words:
                    index = words.index(prep)
                    if index < len(words) - 1:
                        # Take words after the preposition, excluding the amount
                        item_words = []
                        for word in words[index+1:]:
                            if not re.match(r'\d+[,.]?\d*', word) and word not in ['rs', 'rupees', 'inr', '₹', 'rupee']:
                                item_words.append(word)
                        if item_words:
                            item = ' '.join(item_words)
                            break
            
            if not item:
                # Try to find nouns in the message that might be the item
                item_candidates = [
                    'food', 'groceries', 'meal', 'dinner', 'lunch', 'breakfast',
                    'rent', 'bill', 'phone', 'internet', 'electricity', 'gas',
                    'movie', 'ticket', 'transport', 'taxi', 'uber', 'cab',
                    'clothes', 'shoes', 'shirt', 'pants', 'dress',
                    'book', 'game', 'subscription', 'service'
                ]
                
                for candidate in item_candidates:
                    if candidate in message.lower():
                        item = candidate
                        break
            
            if not item:
                return f"I identified an expense of ₹{amount:.2f}, but I'm not sure what it's for. Can you specify what you spent it on?"
            
            # Determine category and subcategory
            categories = {
                'Food': ['grocery', 'groceries', 'food', 'meal', 'dinner', 'lunch', 'breakfast', 'restaurant', 'eat', 'eating', 'snack'],
                'Housing': ['rent', 'mortgage', 'housing', 'apartment', 'house'],
                'Utilities': ['electricity', 'water', 'gas', 'internet', 'phone', 'bill', 'utility'],
                'Transportation': ['uber', 'taxi', 'cab', 'transport', 'bus', 'train', 'subway', 'metro', 'gas', 'petrol', 'diesel', 'fuel'],
                'Entertainment': ['movie', 'show', 'concert', 'sport', 'game', 'entertainment', 'streaming', 'subscription', 'netflix', 'amazon', 'spotify'],
                'Shopping': ['clothes', 'shoe', 'shirt', 'pant', 'dress', 'accessory', 'jewelry', 'electronic', 'gadget', 'phone', 'laptop'],
                'Healthcare': ['doctor', 'medicine', 'hospital', 'health', 'medical', 'healthcare'],
                'Education': ['book', 'course', 'class', 'tuition', 'education', 'school', 'college', 'university']
            }
            
            category = 'Miscellaneous'
            for cat, keywords in categories.items():
                if any(keyword in message.lower() or keyword in item.lower() for keyword in keywords):
                    category = cat
                    break
            
            # Get user's budget for this category
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT amount FROM budgets WHERE user_id = ? AND category = ?",
                (user_id, category)
            )
            budget_result = cursor.fetchone()
            budget_amount = budget_result[0] if budget_result else 0
            
            cursor.execute(
                "SELECT SUM(amount) FROM transactions WHERE user_id = ? AND category = ?",
                (user_id, category)
            )
            spent_result = cursor.fetchone()
            spent_amount = spent_result[0] if spent_result and spent_result[0] is not None else 0
            
            # Add the new transaction amount
            new_total = spent_amount + amount
            budget_percentage = (new_total / budget_amount) * 100 if budget_amount > 0 else 0
            
            # Check if this pushes the user over budget
            budget_analysis = ""
            if budget_amount > 0:
                if new_total > budget_amount:
                    budget_analysis = f"⚠️ This puts you ₹{(new_total - budget_amount):.2f} over your {category} budget of ₹{budget_amount:.2f}."
                elif budget_percentage > 90:
                    budget_analysis = f"⚠️ You've now used {budget_percentage:.1f}% of your {category} budget."
                elif budget_percentage > 75:
                    budget_analysis = f"Note: You've now used {budget_percentage:.1f}% of your {category} budget."
                else:
                    budget_analysis = f"You've used {budget_percentage:.1f}% of your {category} budget and have ₹{(budget_amount - new_total):.2f} remaining."
            
            # Check if this expense affects any of the user's goals
            cursor.execute("SELECT name, target_amount, current_amount FROM goals WHERE user_id = ?", (user_id,))
            goals = cursor.fetchall()
            goal_analysis = ""
            
            if goals:
                for name, target, current in goals:
                    # Simple assumption: any expense reduces money available for goals
                    impact_percentage = (amount / (target - current)) * 100 if (target - current) > 0 else 0
                    if impact_percentage > 10:
                        goal_analysis = f"\n\nThis expense is equivalent to {impact_percentage:.1f}% of what you still need for your {name} goal."
                        break
            
            # Add expense to transactions
            cursor.execute(
                "INSERT INTO transactions (user_id, amount, description, category, date) VALUES (?, ?, ?, ?, date('now'))",
                (user_id, amount, item, category)
            )
            conn.commit()
            conn.close()
            
            # Provide advice
            advice = ""
            if amount > 1000:
                advice = "\n\nThat's a significant expense. Consider if there are ways to reduce similar costs in the future."
            elif category == 'Food' and amount > 500:
                advice = "\n\nConsider meal prepping to reduce food expenses."
            elif category == 'Entertainment' and budget_percentage > 90:
                advice = "\n\nYou might want to look for free entertainment options for the rest of the month."
            
            return f"Recorded ₹{amount:.2f} for {item} in category '{category}'. {budget_analysis}{goal_analysis}{advice}"
        
        # If no specific query type is detected, provide a general response
        return "I'm not sure how to help with that. You can ask me about your budget, goals, or tell me about expenses. Try saying something like 'I spent ₹2000 on groceries' or 'How am I doing on my budget?'"
        
    except Exception as e:
        print(f"Error in get_financial_advice: {e}")
        return "I encountered an issue processing your request. Please try again or contact support if the problem persists."

@app.route('/transaction', methods=['POST'])
def create_transaction():
    """Record a new transaction"""
    try:
        data = request.get_json()
        required_fields = ['user_id', 'item', 'category', 'amount']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'status': 'error', 'error': f'Missing required field: {field}'}), 400
        
        # Get optional fields
        date = data.get('date')
        label = data.get('label')
        
        # Record the transaction
        from database import record_transaction
        success, result = record_transaction(
            data['user_id'],
            data['item'],
            data['category'],
            data['amount'],
            date,
            label
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Transaction recorded successfully',
                'transaction_id': result
            })
        else:
            return jsonify({
                'status': 'error',
                'error': f'Failed to record transaction: {result}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error creating transaction: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/transaction/<int:user_id>', methods=['GET'])
def get_transactions(user_id):
    """Get recent transactions for a user"""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=10, type=int)
        category = request.args.get('category')
        
        # Get transactions
        from database import get_recent_transactions
        transactions = get_recent_transactions(user_id, limit, category)
        
        return jsonify({
            'status': 'success',
            'transactions': [dict(transaction) for transaction in transactions]
        })
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/dashboard/<int:user_id>', methods=['GET'])
def get_dashboard(user_id):
    """Get dashboard data for a user"""
    try:
        # Get period from query parameters (week, month, year)
        period = request.args.get('period', default='month')
        
        # Import necessary functions
        from database import get_spending_summary, get_recent_transactions, get_user_goals
        from database import get_budget_progress, get_spending_trends
        
        # Get spending summary by category
        spending_summary = get_spending_summary(user_id, period)
        
        # Get spending trends for visualization
        spending_trends = get_spending_trends(user_id, period)
        
        # Get recent transactions
        recent_transactions = get_recent_transactions(user_id, 5)
        
        # Get active goals
        goals = get_user_goals(user_id)
        
        # Get budget progress for top categories
        budget_progress = {}
        if spending_summary:
            for spending in spending_summary[:3]:  # Get top 3 categories
                progress = get_budget_progress(user_id, spending['category'])
                budget_progress[spending['category']] = progress
        
        # Calculate total spent and remaining
        total_spent = sum(spending['total'] for spending in spending_summary)
        total_budget = 0
        total_remaining = 0
        
        # Get overall budget progress
        overall_progress = get_budget_progress(user_id)
        if overall_progress:
            total_budget = overall_progress.get('total', 0)
            total_remaining = overall_progress.get('remaining', 0)
        
        # Generate insights based on spending patterns
        insights = []
        
        # Insight 1: Budget categories exceeding 80% usage
        for category, progress in budget_progress.items():
            percentage = progress.get('percentage', 0)
            if percentage > 80:
                insights.append(f"Your {category} budget is at {percentage:.1f}% - consider adjusting spending")
        
        # Insight 2: Categories with no budget
        for spending in spending_summary:
            category = spending['category']
            if category not in budget_progress and spending['total'] > 1000:
                insights.append(f"Consider creating a budget for {category}")
                
        # Insight 3: Goal progress
        for goal in goals:
            progress_percent = (goal['current_amount'] / goal['target_amount']) * 100 if goal['target_amount'] > 0 else 0
            if progress_percent < 25 and goal['deadline']:
                deadline = datetime.strptime(goal['deadline'], '%Y-%m-%d')
                days_left = (deadline - datetime.now()).days
                if days_left < 30:
                    insights.append(f"Your {goal['goal_name']} goal needs attention - only {days_left} days left")
        
        # Insight 4: Analyze spending trends
        if spending_trends and spending_trends['datasets']:
            total_dataset = spending_trends['datasets'][-1]  # Last dataset is Total
            if len(total_dataset['data']) >= 2:
                current = total_dataset['data'][-1]
                previous = total_dataset['data'][-2] 
                if previous > 0 and current > previous * 1.2:  # 20% increase
                    insights.append(f"Your spending increased by {((current-previous)/previous*100):.1f}% compared to last {period}")
        
        # Assemble dashboard data
        dashboard = {
            'summary': {
                'total_spent': total_spent,
                'total_budget': total_budget,
                'total_remaining': total_remaining,
                'period': period
            },
            'spending_by_category': [dict(spending) for spending in spending_summary],
            'spending_trends': spending_trends,
            'recent_transactions': [dict(tx) for tx in recent_transactions],
            'budget_progress': budget_progress,
            'goals': [dict(goal) for goal in goals],
            'insights': insights
        }
        
        return jsonify({
            'status': 'success',
            'dashboard': dashboard
        })
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/debug/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify(routes)

@app.route('/test_ai', methods=['GET'])
def test_ai():
    """Test endpoint to quickly try the AI assistant with different queries"""
    message = request.args.get('message', '')
    user_id = request.args.get('user_id', '1')  # Default to user 1 for testing
    
    if not message:
        return jsonify({
            "success": False,
            "message": "Please provide a 'message' query parameter"
        })
    
    # Test which type of query it is
    query_types = []
    if is_greeting(message):
        query_types.append("greeting")
    if is_help_request(message):
        query_types.append("help_request")
    if is_expense_query(message):
        query_types.append("expense_query")
    if is_budget_query(message):
        query_types.append("budget_query")
    if is_goal_query(message):
        query_types.append("goal_query")
    
    # Get the response from the financial advisor
    response = get_financial_advice(user_id, message)
    
    return jsonify({
        "success": True,
        "message": message,
        "detected_query_types": query_types,
        "response": response
    })

def initialize_database():
    """Create necessary tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create budgets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        category TEXT NOT NULL,
        amount REAL NOT NULL,
        period TEXT DEFAULT 'monthly',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create transactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        description TEXT,
        category TEXT,
        date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create goals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        target_amount REAL NOT NULL,
        current_amount REAL DEFAULT 0,
        target_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create chat_history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        sender TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for a user"""
    try:
        user_id = request.args.get('userId')
        limit = request.args.get('limit', default=50, type=int)
        
        if not user_id:
            return jsonify({
                "success": False,
                "message": "Missing required query parameter: userId"
            }), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT sender, message, timestamp FROM chat_history "
            "WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "sender": row['sender'],
                "message": row['message'],
                "timestamp": row['timestamp']
            })
        
        conn.close()
        
        # Return history in reverse order (oldest first)
        return jsonify({
            "success": True,
            "data": list(reversed(history))
        })
        
    except Exception as e:
        print(f"Error in get_chat_history: {e}")
        return jsonify({
            "success": False,
            "message": "An error occurred while retrieving chat history"
        }), 500

# Initialize database when the app starts
if __name__ == '__main__':
    try:
        initialize_database()
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting application: {e}")
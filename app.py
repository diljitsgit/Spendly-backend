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
    get_user_goals
)
import random
import re
import html
from functools import wraps
from collections import defaultdict
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

@app.route('/')
def home():
    return jsonify({
        'message': 'Welcome to the Financial Management API',
        'endpoints': {
            'user': '/user (POST)',
            'budget': '/budget (POST, GET)',
            'budget_progress': '/budget/progress/<user_id> (GET)',
            'goals': '/goal (POST, GET)',
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
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email)
                VALUES (?, ?)
            ''', (data['username'], data['email']))
            conn.commit()
            user_id = cursor.lastrowid
            return jsonify({
                'message': 'User created successfully',
                'user_id': user_id
            })
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Username or email already exists'}), 400
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error creating user: {e}")
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
    """Extract amount and item from a chat message"""
    try:
        logger.info(f"Parsing message: {message}")
        
        # Remove currency symbols and convert to lowercase
        message = message.lower().replace('₹', '').replace('rs', '').replace('rs.', '')
        logger.info(f"Cleaned message: {message}")
        
        # Find numbers in the message
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        logger.info(f"Found numbers: {numbers}")
        if not numbers:
            logger.warning("No numbers found in message")
            return None, None
        
        amount = float(numbers[0])
        logger.info(f"Extracted amount: {amount}")
        
        # Common words to ignore
        ignore_words = {'should', 'i', 'buy', 'spend', 'on', 'for', 'a', 'an', 'the', 'some', 'rs', 'rupees'}
        
        # Split message into words and remove ignored words and numbers
        words = message.split()
        logger.info(f"Split words: {words}")
        words = [w for w in words if w not in ignore_words and not w.replace('.', '').isdigit()]
        logger.info(f"Filtered words: {words}")
        
        # Try to find item after amount
        amount_str = str(int(amount) if amount.is_integer() else amount)
        logger.info(f"Looking for amount string: {amount_str}")
        
        try:
            if amount_str in message:
                amount_index = words.index(next(w for w in words if amount_str in w))
                logger.info(f"Found amount at index: {amount_index}")
                item_words = words[amount_index+1:]
                if item_words:
                    item = ' '.join(item_words)
                    logger.info(f"Found item after amount: {item}")
                    return amount, item
            
            # If not found after amount, look for words before amount
            item_words = words[:words.index(next(w for w in words if amount_str in w))]
            if item_words:
                item = ' '.join(item_words)
                logger.info(f"Found item before amount: {item}")
                return amount, item
            
            logger.warning("No item found in message")
            return amount, None
            
        except StopIteration:
            logger.warning(f"Could not find amount string in words: {words}")
            # Fallback: just take any remaining words as the item
            if words:
                item = ' '.join(words)
                logger.info(f"Fallback: using remaining words as item: {item}")
                return amount, item
            return amount, None
            
    except Exception as e:
        logger.error(f"Error parsing message: {e}")
        return None, None

def engineer_features(amount, item):
    """Engineer features for model prediction"""
    try:
        # Default features that should match the training data
        default_features = {
            'amount': amount,
            'amount_log': np.log1p(amount),
            'amount_squared': amount ** 2,
            'amount_binned': 0,  # Will be updated
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month - 1,  # 0-based index
            'quarter': (datetime.now().month - 1) // 3,
            'day_of_month': datetime.now().day,
            'category': 0  # Will be updated
        }
        
        # Update amount_binned
        amount_bins = [0, 10000, 25000, 50000, 100000, float('inf')]
        for i, (lower, upper) in enumerate(zip(amount_bins[:-1], amount_bins[1:])):
            if lower <= amount < upper:
                default_features['amount_binned'] = i
                break
        
        # Update category
        category, subcategory = determine_category(item)
        default_features['category'] = hash(category) % 10  # Simple hash to convert category to number
        
        # Create DataFrame with default features
        features = pd.DataFrame([default_features])
        
        # Ensure all required features are present
        for feature in selected_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Only keep selected features in the right order
        features = features[selected_features]
        
        return features
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        # Return a basic feature set as fallback
        basic_features = pd.DataFrame([[amount, np.log1p(amount), amount ** 2, 0, 1, 0, 0, 0, 0, 0]], 
                                    columns=selected_features)
        return basic_features

def determine_category(item):
    """Enhanced category determination with more keywords and subcategories"""
    item = item.lower()
    
    categories = {
        'Food': {
            'keywords': ['food', 'grocery', 'groceries', 'meal', 'restaurant', 'dining', 'lunch', 
                        'dinner', 'breakfast', 'snack', 'fruit', 'vegetable', 'meat', 'dairy', 
                        'beverage', 'drink', 'takeout', 'delivery'],
            'subcategories': ['Groceries', 'Dining Out', 'Delivery', 'Snacks']
        },
        'Electronics': {
            'keywords': ['phone', 'laptop', 'computer', 'tablet', 'gadget', 'electronic', 'tv', 
                        'television', 'camera', 'headphone', 'speaker', 'smartwatch', 'console', 
                        'printer', 'monitor', 'keyboard', 'mouse'],
            'subcategories': ['Mobile Devices', 'Computers', 'Accessories', 'Entertainment Systems']
        },
        'Healthcare': {
            'keywords': ['medicine', 'doctor', 'hospital', 'medical', 'health', 'dental', 'pharmacy',
                        'prescription', 'therapy', 'checkup', 'vitamin', 'supplement', 'insurance'],
            'subcategories': ['Medical Care', 'Medications', 'Insurance', 'Wellness']
        },
        'Housing': {
            'keywords': ['rent', 'mortgage', 'utility', 'electricity', 'water', 'gas', 'internet',
                        'maintenance', 'repair', 'furniture', 'appliance', 'decoration', 'cleaning'],
            'subcategories': ['Rent/Mortgage', 'Utilities', 'Maintenance', 'Furnishing']
        },
        'Transportation': {
            'keywords': ['car', 'fuel', 'gas', 'petrol', 'diesel', 'bus', 'train', 'taxi', 'uber',
                        'maintenance', 'repair', 'insurance', 'parking', 'toll', 'metro', 'fare'],
            'subcategories': ['Public Transit', 'Private Vehicle', 'Maintenance', 'Ride Services']
        },
        'Education': {
            'keywords': ['tuition', 'course', 'class', 'book', 'textbook', 'school', 'college',
                        'university', 'training', 'workshop', 'seminar', 'certification', 'exam'],
            'subcategories': ['Tuition', 'Materials', 'Professional Development', 'Certifications']
        },
        'Entertainment': {
            'keywords': ['movie', 'theatre', 'concert', 'game', 'streaming', 'subscription', 'hobby',
                        'sport', 'fitness', 'gym', 'netflix', 'spotify', 'amazon', 'disney'],
            'subcategories': ['Digital Services', 'Live Events', 'Sports/Fitness', 'Hobbies']
        }
    }
    
    # Try to find exact category match
    for category, data in categories.items():
        if any(keyword in item for keyword in data['keywords']):
            subcategory = next((sub for sub in data['subcategories'] 
                              if any(sub.lower() in item for keyword in data['keywords'])), 
                              data['subcategories'][0])
            return category, subcategory
    
    # Fallback to general category
    return 'Miscellaneous', 'General'

def is_greeting(message):
    """Check if the message is a greeting"""
    greetings = {'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'hi there', 'hello there'}
    return any(greeting in message.lower() for greeting in greetings)

def is_help_request(message):
    """Check if the message is asking for help"""
    help_phrases = {'help', 'what can you do', 'how do you work', 'what do you do', 'guide me', 'assist me'}
    return any(phrase in message.lower() for phrase in help_phrases)

def is_expense_query(message):
    """Check if the message is about an expense"""
    expense_keywords = {'spend', 'buy', 'purchase', 'cost', 'worth', 'price', 'paying'}
    has_number = bool(re.findall(r'\d+', message))
    return any(keyword in message.lower() for keyword in expense_keywords) and has_number

def get_financial_advice(user_id, message):
    """Get financial advice for a user's message"""
    try:
        # Check for greetings
        if is_greeting(message):
            return "Hello! I'm your financial advisor. How can I help you with your spending decisions today?"
            
        # Check for help requests
        if is_help_request(message):
            return """I can help you with:
• Analyzing purchases and expenses
• Checking your budget status
• Classifying expenses as needs or wants
• Providing spending recommendations
• Tracking financial goals

Just ask me about any purchase you're considering!"""
            
        # Check if this is an expense query
        if not is_expense_query(message):
            return "I'm here to help with your financial decisions! Please ask me about specific purchases or expenses you're considering."
        
        # Parse the expense message
        amount, item = parse_expense_message(message)
        if not amount or not item:
            return "I couldn't understand the expense details. Please specify both the amount and item, for example: 'Should I buy a laptop for ₹45000?'"
            
        # Get category and subcategory
        category, subcategory = determine_category(item)
        
        # Get budget analysis
        budget_analysis, error = analyze_budget_impact(user_id, amount, category)
        if error:
            logger.warning(f"Budget analysis error: {error}")
            budget_analysis = None
            
        # Get model prediction
        expense_type, confidence = get_model_prediction(amount, category)
        
        # Get user's financial goals
        goals = get_user_goals(user_id)
        goal_impact = []
        for goal in goals:
            if goal['current_amount'] + amount > goal['target_amount']:
                goal_impact.append(f"This expense would exceed your {goal['goal_name']} goal by ₹{(goal['current_amount'] + amount - goal['target_amount']):.2f}")
            elif (goal['target_amount'] - goal['current_amount']) * 0.25 < amount:
                goal_impact.append(f"This expense would use {(amount / (goal['target_amount'] - goal['current_amount']) * 100):.1f}% of your remaining {goal['goal_name']} goal budget")
        
        # Build response parts
        response_parts = []
        
        # Basic classification
        response_parts.append(f"Analysis for spending ₹{amount:.2f} on {item}:")
        response_parts.append(f"Category: {category} - {subcategory}")
        response_parts.append(f"Classification: {expense_type} ({confidence*100:.1f}% confidence)")
        
        # Budget impact
        if budget_analysis:
            response_parts.append("\nBudget Impact:")
            response_parts.append(f"• Impact Level: {budget_analysis['impact_level'].title()}")
            response_parts.append(f"• Amount vs Budget: {budget_analysis['amount_vs_budget']} of monthly budget")
            response_parts.append(f"• Remaining Budget: ₹{budget_analysis['budget_remaining']:.2f}")
            response_parts.append(f"• Month Progress: {budget_analysis['month_progress']} through the month")
            response_parts.append(f"• Spending Rate: {budget_analysis['spending_rate'].title()} monthly average")
            
        # Goal impact
        if goal_impact:
            response_parts.append("\nGoal Impact:")
            for impact in goal_impact:
                response_parts.append(f"• {impact}")
                
        # Recommendations
        response_parts.append("\nRecommendations:")
        if budget_analysis and budget_analysis['recommendation']:
            response_parts.append(f"• Budget: {budget_analysis['recommendation']}")
            
        # Category-specific advice
        if category == 'Electronics':
            response_parts.append("• Consider checking warranty and return policies")
            response_parts.append("• Look for seasonal sales or refurbished options")
            if subcategory == 'Mobile Devices':
                response_parts.append("• Compare different models and their features")
                response_parts.append("• Check for carrier deals or bundled offers")
            elif subcategory == 'Computers':
                response_parts.append("• Consider your usage needs vs specifications")
                response_parts.append("• Look for student/professional discounts if applicable")
        elif category == 'Food':
            if subcategory == 'Groceries':
                response_parts.append("• Consider bulk purchases for non-perishables")
                response_parts.append("• Check for loyalty program discounts")
            elif subcategory == 'Dining Out':
                response_parts.append("• Look for early bird or happy hour specials")
                response_parts.append("• Consider takeout options")
        elif category == 'Healthcare':
            response_parts.append("• Check if covered by insurance")
            response_parts.append("• Consider generic alternatives if applicable")
            
        # Join all parts with proper spacing
        final_response = "\n".join(response_parts)
        
        # Log the response for debugging
        logger.info(f"Generated response length: {len(final_response)}")
        
        return final_response
        
    except Exception as e:
        logger.error(f"Error in get_financial_advice: {e}")
        return "I encountered an error while analyzing your request. Please try again with a clear expense amount and item."

@app.route('/chat', methods=['POST'])
@rate_limit
def chat():
    """Enhanced chat endpoint with input sanitization and rate limiting"""
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Sanitize input
        message = sanitize_input(data['message'])
        user_id = int(data['user_id'])
        
        # Get financial advice
        response = get_financial_advice(user_id, message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
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

def get_model_prediction(amount, category, fallback_categories=None):
    """Enhanced model prediction with smart fallback"""
    if fallback_categories is None:
        fallback_categories = {
            'Food': {'type': 'Need', 'confidence': 0.9},
            'Healthcare': {'type': 'Need', 'confidence': 0.95},
            'Housing': {'type': 'Need', 'confidence': 0.95},
            'Transportation': {'type': 'Need', 'confidence': 0.85},
            'Education': {'type': 'Need', 'confidence': 0.8},
            'Electronics': {'type': 'Want', 'confidence': 0.7},
            'Entertainment': {'type': 'Want', 'confidence': 0.9}
        }
    
    try:
        # Try model prediction first
        features = engineer_features(amount, category)
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        confidence = max(probabilities[0])
        label = prediction[0]
        
        # Validate prediction
        if confidence < 0.6:  # If model is not confident enough
            # Use fallback logic
            if category in fallback_categories:
                fallback = fallback_categories[category]
                return fallback['type'], fallback['confidence']
            else:
                # Use amount-based heuristic for unknown categories
                if amount > 10000:  # High amount
                    return 'Want', 0.7
                else:
                    return 'Need', 0.6
                    
        return label, confidence
        
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        # Use category-based fallback
        if category in fallback_categories:
            fallback = fallback_categories[category]
            return fallback['type'], fallback['confidence']
        return 'Unknown', 0.5

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

if __name__ == '__main__':
    try:
        # Initialize the database
        init_db()
        logger.info("Database initialized successfully")
        
        # Start the Flask server
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a connection to the SQLite database"""
    db_path = Path(__file__).parent / 'finance.db'
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            item TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            label TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
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
            period TEXT NOT NULL,  -- 'monthly', 'weekly', 'yearly'
            start_date TEXT NOT NULL,
            end_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Create financial goals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal_name TEXT NOT NULL,
            target_amount REAL NOT NULL,
            current_amount REAL DEFAULT 0,
            deadline TEXT,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_profile(username):
    """Get user profile details from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT age, gender FROM user_profiles WHERE username = ?', (username,))
    profile = cursor.fetchone()
    
    conn.close()
    return profile

def create_user_profile(username, age, gender):
    """Create a new user profile"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO user_profiles (username, age, gender)
            VALUES (?, ?, ?)
        ''', (username, age, gender))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def create_budget(user_id, category, amount, period, start_date, end_date=None):
    """Create a new budget for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO budgets (user_id, category, amount, period, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, category, amount, period, start_date, end_date))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating budget: {e}")
        return False
    finally:
        conn.close()

def get_user_budgets(user_id):
    """Get all budgets for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM budgets 
        WHERE user_id = ? AND (end_date IS NULL OR end_date >= date('now'))
    ''', (user_id,))
    budgets = cursor.fetchall()
    
    conn.close()
    return budgets

def get_budget_progress(user_id, category=None):
    """Get budget progress for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get current month's start and end dates
    current_month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    next_month_start = (datetime.now().replace(day=28) + timedelta(days=4)).replace(day=1).strftime('%Y-%m-%d')
    
    # Get total spent in current period
    query = '''
        SELECT SUM(amount) as total_spent
        FROM transactions
        WHERE user_id = ? 
        AND date >= ? AND date < ?
    '''
    params = [user_id, current_month_start, next_month_start]
    
    if category:
        query += ' AND category = ?'
        params.append(category)
    
    cursor.execute(query, params)
    total_spent = cursor.fetchone()[0] or 0
    
    # Get budget for the period
    query = '''
        SELECT SUM(amount) as total_budget
        FROM budgets
        WHERE user_id = ? 
        AND start_date <= ? 
        AND (end_date IS NULL OR end_date >= ?)
    '''
    params = [user_id, current_month_start, current_month_start]
    
    if category:
        query += ' AND category = ?'
        params.append(category)
    
    cursor.execute(query, params)
    total_budget = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        'total_budget': total_budget,
        'total_spent': total_spent,
        'remaining': total_budget - total_spent,
        'percentage_spent': (total_spent / total_budget * 100) if total_budget > 0 else 0
    }

def create_financial_goal(user_id, goal_name, target_amount, deadline):
    """Create a new financial goal"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO financial_goals (user_id, goal_name, target_amount, deadline)
            VALUES (?, ?, ?, ?)
        ''', (user_id, goal_name, target_amount, deadline))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating financial goal: {e}")
        return False
    finally:
        conn.close()

def update_financial_goal(user_id, goal_id, current_amount):
    """Update progress of a financial goal"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE financial_goals
            SET current_amount = ?
            WHERE id = ? AND user_id = ?
        ''', (current_amount, goal_id, user_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating financial goal: {e}")
        return False
    finally:
        conn.close()

def get_user_goals(user_id):
    """Get all financial goals for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM financial_goals 
        WHERE user_id = ? AND status = 'active'
    ''', (user_id,))
    goals = cursor.fetchall()
    
    conn.close()
    return goals 
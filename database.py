import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection pool implementation
class ConnectionPool:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConnectionPool, cls).__new__(cls)
                cls._instance.connections = []
                cls._instance.max_connections = 10
        return cls._instance
    
    def get_connection(self):
        db_path = Path(__file__).parent / 'finance.db'
        
        with self._lock:
            if self.connections:
                conn = self.connections.pop()
                try:
                    # Test if connection is still valid
                    conn.execute("SELECT 1")
                    return conn
                except sqlite3.Error:
                    # Connection is stale, create a new one
                    pass
            
            # Create new connection
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            return conn
    
    def return_connection(self, conn):
        with self._lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(conn)
            else:
                conn.close()

# Global connection pool
connection_pool = ConnectionPool()

def get_db_connection():
    """Get a connection to the SQLite database from the pool"""
    return connection_pool.get_connection()

def close_db_connection(conn):
    """Return a connection to the pool"""
    connection_pool.return_connection(conn)

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
            password TEXT,
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
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category)')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_budgets_user_id ON budgets(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_budgets_category ON budgets(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_budgets_dates ON budgets(start_date, end_date)')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_user_id ON financial_goals(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_status ON financial_goals(status)')
    
    conn.commit()
    close_db_connection(conn)

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
        close_db_connection(conn)

def get_user_budgets(user_id):
    """Get all budgets for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT * FROM budgets 
            WHERE user_id = ? AND (end_date IS NULL OR end_date >= date('now'))
        ''', (user_id,))
        budgets = cursor.fetchall()
        return budgets
    finally:
        close_db_connection(conn)

def get_budget_progress(user_id, category=None):
    """Get budget progress for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
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
        
        return {
            'total': total_budget,
            'spent': total_spent,
            'remaining': total_budget - total_spent,
            'percentage': (total_spent / total_budget * 100) if total_budget > 0 else 0
        }
    finally:
        close_db_connection(conn)

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
        close_db_connection(conn)

def update_financial_goal(user_id, goal_id, current_amount=None, status=None):
    """Update a financial goal"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        updates = []
        params = []
        
        if current_amount is not None:
            updates.append("current_amount = ?")
            params.append(current_amount)
            
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            
        if not updates:
            return False
            
        query = f"UPDATE financial_goals SET {', '.join(updates)} WHERE id = ? AND user_id = ?"
        params.extend([goal_id, user_id])
        
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating financial goal: {e}")
        return False
    finally:
        close_db_connection(conn)

def get_user_goals(user_id):
    """Get all financial goals for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT * FROM financial_goals 
            WHERE user_id = ? AND status = 'active'
        ''', (user_id,))
        goals = cursor.fetchall()
        return goals
    finally:
        close_db_connection(conn)

def record_transaction(user_id, item, category, amount, date=None, label=None):
    """Record a new financial transaction"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        # Get user data (using defaults if not available)
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return False, "User not found"
        
        # Default values
        age = 30  # Default age
        gender = "Not specified"  # Default gender
        
        # Insert the transaction
        cursor.execute('''
            INSERT INTO transactions (date, age, gender, item, category, amount, label, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, age, gender, item, category, amount, label, user_id))
        
        conn.commit()
        
        transaction_id = cursor.lastrowid
        
        # If transaction is recorded to a goal, update the goal progress
        if label and label.startswith("goal:"):
            goal_id = label.split(":")[1]
            try:
                goal_id = int(goal_id)
                # Get current amount
                cursor.execute('''
                    SELECT current_amount, target_amount FROM financial_goals
                    WHERE id = ? AND user_id = ?
                ''', (goal_id, user_id))
                goal = cursor.fetchone()
                
                if goal:
                    new_amount = goal['current_amount'] + amount
                    status = 'completed' if new_amount >= goal['target_amount'] else 'active'
                    
                    # Update the goal progress
                    update_financial_goal(user_id, goal_id, new_amount, status)
            except (ValueError, IndexError):
                logger.warning(f"Invalid goal format in label: {label}")
        
        return True, transaction_id
        
    except Exception as e:
        logger.error(f"Error recording transaction: {e}")
        conn.rollback()
        return False, str(e)
    finally:
        close_db_connection(conn)

def get_recent_transactions(user_id, limit=10, category=None):
    """Get recent transactions for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = '''
            SELECT * FROM transactions 
            WHERE user_id = ?
        '''
        params = [user_id]
        
        if category:
            query += ' AND category = ?'
            params.append(category)
            
        query += ' ORDER BY date DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        transactions = cursor.fetchall()
        return transactions
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return []
    finally:
        close_db_connection(conn)

def get_spending_summary(user_id, period='month'):
    """Get spending summary for a user by category"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Determine date range based on period
        today = datetime.now()
        if period == 'week':
            start_date = (today - timedelta(days=today.weekday())).strftime('%Y-%m-%d')
        elif period == 'month':
            start_date = today.replace(day=1).strftime('%Y-%m-%d')
        elif period == 'year':
            start_date = today.replace(month=1, day=1).strftime('%Y-%m-%d')
        else:  # default to month
            start_date = today.replace(day=1).strftime('%Y-%m-%d')
            
        # Get spending by category
        cursor.execute('''
            SELECT category, SUM(amount) as total 
            FROM transactions 
            WHERE user_id = ? AND date >= ?
            GROUP BY category 
            ORDER BY total DESC
        ''', (user_id, start_date))
        
        spending = cursor.fetchall()
        return spending
    except Exception as e:
        logger.error(f"Error getting spending summary: {e}")
        return []
    finally:
        close_db_connection(conn)

def get_spending_trends(user_id, period='month', num_periods=6):
    """Get spending trends over time for visualization"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        today = datetime.now()
        
        # Define period ranges
        periods = []
        labels = []
        
        # Calculate start dates for each period
        if period == 'week':
            for i in range(num_periods-1, -1, -1):
                end_date = today - timedelta(days=i*7)
                start_date = end_date - timedelta(days=6)
                periods.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                labels.append(f"{start_date.strftime('%d %b')}")
        elif period == 'month':
            for i in range(num_periods-1, -1, -1):
                # Calculate month
                month = today.month - i
                year = today.year
                while month <= 0:
                    month += 12
                    year -= 1
                
                start_date = datetime(year, month, 1)
                # Get end of month
                if month == 12:
                    end_month = 1
                    end_year = year + 1
                else:
                    end_month = month + 1
                    end_year = year
                end_date = datetime(end_year, end_month, 1) - timedelta(days=1)
                
                periods.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                labels.append(f"{start_date.strftime('%b %Y')}")
        else:  # year - by quarters
            for i in range(num_periods-1, -1, -1):
                quarter = ((today.month - 1) // 3) - i
                year = today.year
                while quarter < 0:
                    quarter += 4
                    year -= 1
                
                q_month = (quarter * 3) + 1
                start_date = datetime(year, q_month, 1)
                
                # End date is start of next quarter - 1 day
                end_month = q_month + 3
                end_year = year
                if end_month > 12:
                    end_month -= 12
                    end_year += 1
                end_date = datetime(end_year, end_month, 1) - timedelta(days=1)
                
                periods.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                labels.append(f"Q{quarter+1} {year}")
        
        # Get top categories (to include in trend)
        cursor.execute('''
            SELECT category, SUM(amount) as total 
            FROM transactions 
            WHERE user_id = ? 
            GROUP BY category 
            ORDER BY total DESC
            LIMIT 5
        ''', (user_id,))
        top_categories = [row['category'] for row in cursor.fetchall()]
        
        # Initialize results structure
        results = {
            'labels': labels,
            'datasets': []
        }
        
        # Get data for each category
        for category in top_categories:
            dataset = {
                'label': category,
                'data': []
            }
            
            # Get data for each period
            for start_date, end_date in periods:
                cursor.execute('''
                    SELECT SUM(amount) as total 
                    FROM transactions 
                    WHERE user_id = ? AND category = ? AND date >= ? AND date <= ?
                ''', (user_id, category, start_date, end_date))
                total = cursor.fetchone()['total'] or 0
                dataset['data'].append(total)
            
            results['datasets'].append(dataset)
        
        # Add a "Total" dataset
        total_dataset = {
            'label': 'Total',
            'data': []
        }
        
        for start_date, end_date in periods:
            cursor.execute('''
                SELECT SUM(amount) as total 
                FROM transactions 
                WHERE user_id = ? AND date >= ? AND date <= ?
            ''', (user_id, start_date, end_date))
            total = cursor.fetchone()['total'] or 0
            total_dataset['data'].append(total)
        
        results['datasets'].append(total_dataset)
        
        return results
    except Exception as e:
        logger.error(f"Error getting spending trends: {e}")
        return {'labels': [], 'datasets': []}
    finally:
        close_db_connection(conn)

def create_user(username, email, password=None):
    """Create a new user with optional password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
        ''', (username, email, password))
        conn.commit()
        user_id = cursor.lastrowid
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username or email already exists"
    finally:
        conn.close()

def get_user_by_username(username):
    """Get user by username"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, email, password FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    
    conn.close()
    return user

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    
    conn.close()
    return user

def migrate_add_password_field():
    """Migration to add password field to users table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if password column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        
        if 'password' not in [col['name'] for col in columns]:
            # Add password column if it doesn't exist
            cursor.execute("ALTER TABLE users ADD COLUMN password TEXT")
            conn.commit()
            logger.info("Added password column to users table")
            return True
        else:
            logger.info("Password column already exists in users table")
            return True
    except Exception as e:
        logger.error(f"Error migrating database: {e}")
        return False
    finally:
        conn.close() 
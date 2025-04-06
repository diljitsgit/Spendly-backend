import requests
import json

def test_message(message, user_id=1):
    """Send a test message to the AI and print the response"""
    url = "http://localhost:5000/test_ai"
    params = {
        "message": message,
        "user_id": user_id
    }
    
    print(f"\n--- Testing message: '{message}' ---")
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Detected query types: {data['detected_query_types']}")
            print(f"Response: {data['response']}")
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Test different types of messages"""
    # Greeting messages
    test_message("Hello")
    test_message("Hey there!")
    test_message("Good morning")
    test_message("What's up?")
    
    # Help requests
    test_message("What can you do?")
    test_message("How do I use this?")
    test_message("Help me please")
    test_message("Show me how to track expenses")
    
    # Budget queries
    test_message("How's my budget?")
    test_message("How much have I spent on food?")
    test_message("What's my remaining budget for entertainment?")
    test_message("Am I overspending?")
    
    # Goal queries
    test_message("How's my savings goal going?")
    test_message("When can I afford a house?")
    test_message("Am I on track with my car savings?")
    test_message("Show me my goals")
    
    # Expense queries
    test_message("I spent Rs 2000 on groceries")
    test_message("Paid 1500 for dinner last night")
    test_message("Rs. 6000 for rent")
    test_message("Bought a phone for 15000")
    
    # Ambiguous or multiple types
    test_message("Hi, can I afford to spend 5000 on a new laptop?")
    test_message("Help me set a budget for food")
    test_message("What if I buy a car for 500000?")

if __name__ == "__main__":
    main() 
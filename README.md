# Spendly AI Financial Assistant

This document explains the enhanced AI financial assistant capabilities in the Spendly application.

## Overview

The Spendly AI assistant is designed to help users manage their finances by:
- Understanding natural language queries about finances
- Categorizing and recording expenses
- Providing budget insights and tracking
- Monitoring financial goals
- Offering personalized financial advice

## Query Types

The assistant can recognize several types of queries:

### 1. Greetings
- Examples: "Hello", "Hi there", "Good morning", "What's up?"
- Function: `is_greeting()` detects standard greetings and conversational openers
- Response: A friendly greeting and offer to help with finances

### 2. Help Requests
- Examples: "What can you do?", "How do I use this?", "Show me how to track expenses"
- Function: `is_help_request()` identifies questions about how to use the assistant
- Response: A list of the assistant's capabilities and example queries

### 3. Budget Queries
- Examples: "How's my budget?", "How much have I spent on food?", "What's my remaining budget for entertainment?"
- Function: `is_budget_query()` recognizes questions about budget status
- Response: Detailed information about overall budget or category-specific budgets, including spending percentages and remaining amounts

### 4. Goal Queries
- Examples: "How's my savings goal going?", "When can I afford a house?", "Am I on track with my car savings?"
- Function: `is_goal_query()` detects questions about financial goals
- Response: Progress updates on goals, including percentages, time remaining, and required monthly savings

### 5. Expense Queries
- Examples: "I spent Rs 2000 on groceries", "Paid 1500 for dinner last night", "Rs. 6000 for rent"
- Function: `is_expense_query()` identifies statements about spending money
- Response: Confirmation of the recorded expense, category assignment, budget impact analysis, and optional financial advice

## Features

### Natural Language Processing
- Currency detection in multiple formats (₹, Rs., rupees, etc.)
- Amount extraction with support for shorthand (e.g., "2k" = 2000)
- Category determination based on item keywords
- Intent classification to understand the user's request

### Budget Analysis
- Real-time budget tracking by category
- Percentage-based warnings when approaching budget limits
- Top spending category identification
- Detailed breakdown of spending by category

### Goal Tracking
- Progress monitoring for savings goals
- Time-based projections for goal completion
- Required monthly savings calculations
- Multi-goal prioritization

### Transaction Recording
- Automatic categorization of expenses
- Date tracking for expense history
- Description parsing from natural language
- Budget impact assessment for each transaction

### Advice Generation
- Contextual financial tips based on spending patterns
- Category-specific advice (e.g., meal prep suggestions for high food expenses)
- Goal impact analysis for large expenses
- Budget warnings when approaching limits

## API Endpoints

### 1. Chat Endpoint
- **URL**: `/chat`
- **Method**: POST
- **Request Body**: 
  ```json
  {
    "userId": 1,
    "message": "I spent Rs 2000 on groceries"
  }
  ```
- **Response**: 
  ```json
  {
    "success": true,
    "message": "Message processed successfully",
    "response": "Recorded ₹2000.00 for groceries in category 'Food'. You've used 40.0% of your Food budget and have ₹3000.00 remaining."
  }
  ```

### 2. Chat History Endpoint
- **URL**: `/chat/history`
- **Method**: GET
- **Query Parameters**: 
  - `userId`: User ID to get history for
  - `limit`: (Optional) Maximum number of messages to return (default: 50)
- **Response**: 
  ```json
  {
    "success": true,
    "data": [
      {
        "sender": "user",
        "message": "Hello",
        "timestamp": "2023-04-01 14:23:45"
      },
      {
        "sender": "assistant",
        "message": "Hi there! Ready to help you manage your money better.",
        "timestamp": "2023-04-01 14:23:46"
      }
    ]
  }
  ```

### 3. Test AI Endpoint
- **URL**: `/test_ai`
- **Method**: GET
- **Query Parameters**: 
  - `message`: The message to test
  - `user_id`: (Optional) User ID to use for the test (default: 1)
- **Response**: 
  ```json
  {
    "success": true,
    "message": "Hello",
    "detected_query_types": ["greeting"],
    "response": "Hello! How can I assist with your finances today?"
  }
  ```

## Testing

You can quickly test the AI assistant using the provided `test_ai.py` script:

```bash
python test_ai.py
```

This script will run through various sample messages to test different query types and response patterns.

## Extending the AI

To add new capabilities to the AI assistant:

1. Create a new query type detection function (e.g., `is_investment_query()`)
2. Add relevant patterns and keywords to detect this query type
3. Update the `get_financial_advice()` function to handle this new query type
4. Add the database queries needed to provide relevant information
5. Format the response with helpful, personalized information

## Database Schema

The AI assistant uses the following tables:

- `users`: User accounts and authentication
- `budgets`: Budget amounts by category
- `transactions`: Recorded expenses and income
- `goals`: Financial savings goals
- `chat_history`: Conversation history between users and the assistant

## Frontend Integration

The frontend uses:

- `chatService.sendMessage()`: To send user messages to the assistant
- `chatService.getChatHistory()`: To retrieve conversation history

The ChatPage component automatically loads message history and provides a clean interface for interacting with the assistant.

## Future Enhancements

Planned improvements:

1. Sentiment analysis to detect user frustration or satisfaction
2. Machine learning for better expense categorization
3. Income tracking and cash flow analysis
4. Investment recommendations based on risk profile
5. Bill payment reminders and recurring expense detection

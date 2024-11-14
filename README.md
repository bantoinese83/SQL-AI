# SQL-AI: A Conversational SQL Query Generator

This Python module demonstrates how to build a conversational SQL query generator using a Large Language Model (LLM)
like Gemini Pro from Google AI.

## Features

* **Natural Language Queries:** Ask your database questions in plain English.
* **LLM-Powered Query Generation:** Translate your queries into valid SQL using Gemini Pro.
* **SQLite Database:** Connects to an SQLite database for storing and querying car inventory data.
* **Error Handling:**  Includes basic error handling for SQL execution.
* **Rich Output:**  Uses the `rich` library for formatted console output.
* **Sample Data Generation:** Generates realistic car data for testing purposes.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install google-generativeai sqlalchemy rich
    ```
2. **Set Your API Key: Obtain an API key from Google AI (https://cloud.google.com/generative-ai) and set it as an
   environment variable:**
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

3. **Run the Program:**
    ```bash
    python main.py
    ```

4. **How it works:**
    * Schema and Sample Data: The script defines an SQLite database schema for a car inventory and generates sample data.
    * LLM Interaction: The script sends a natural language query, along with the database schema, to the Gemini LLM."
    * SQL Generation: Gemini Pro generates a corresponding SQL query based on the query and schema.
    * Query Execution: The generated SQL query is executed against the SQLite database.
    * Result Display: The query results are displayed in a user-friendly format.

## Example Queries

Here are some example queries you can try:

1. **Get all cars with a price greater than $20,000.**
2. **Find all cars with a mileage less than 50,000 miles.**
3. **Show me all cars with a year between 2018 and 2020.**
4. **Find all cars with a mileage less than 50,000 miles and a price greater than $20,000.**



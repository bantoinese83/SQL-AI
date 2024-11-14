"""
This module manages a car inventory database, including generating sample data,
executing natural language queries, and displaying results.
"""

import os
import logging
import random
from typing import List, Optional, Any, Sequence
from rich.console import Console
from rich.progress import track
from functools import lru_cache
from collections import namedtuple

import google.generativeai as genai
from google.generativeai import GenerationConfig
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    text,
    Row,
    inspect,
    Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# Constants
DATABASE_URL = "sqlite:///car_lot_inventory.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_CACHE_SIZE = 128
MIN_YEAR = 1950
MAX_YEAR = 2023
MIN_PRICE = 5000
MAX_PRICE = 200000
MIN_STOCK = 1
MAX_STOCK = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure rich console
console = Console()

# Configure API Key
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY environment variable not set.")
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# Define model configuration
generation_config = GenerationConfig(
    temperature=1,
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192,
    response_mime_type="text/plain",
)

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Start a chat session to maintain context across multiple queries
chat_session = model.start_chat(history=[])

# SQLAlchemy setup for SQLite database
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()

CarData = namedtuple(
    "CarData",
    [
        "make",
        "model",
        "year",
        "price",
        "stock_quantity",
        "color",
        "mileage",
        "vin",
        "transmission",
        "fuel_type",
        "engine_size",
        "num_doors",
        "owner",
    ],
)


class Car(Base):
    __tablename__ = "cars"

    id = Column(Integer, primary_key=True)
    make = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False, index=True)
    year = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, nullable=False)
    color = Column(String, nullable=True)
    mileage = Column(Float, nullable=True)
    vin = Column(String, unique=True, nullable=True)
    transmission = Column(String, nullable=True)
    fuel_type = Column(String, nullable=True)
    engine_size = Column(Float, nullable=True)
    num_doors = Column(Integer, nullable=True)
    owner = Column(String, nullable=True)


# Create a composite index on make and model
Index("ix_make_model", Car.make, Car.model)

Base.metadata.create_all(engine)


def handle_session(func):
    """Decorator to handle session lifecycle."""

    def wrapper(*args, **kwargs):
        session = Session()
        try:
            result = func(session, *args, **kwargs)
            session.commit()
            return result
        except Exception as session_error:
            logger.error(f"Error in {func.__name__}: {session_error}")
            session.rollback()
        finally:
            session.close()

    return wrapper


@handle_session
def add_cars_to_inventory(session, cars: List[Car]) -> None:
    """Adds a list of cars to the inventory."""
    session.add_all(cars)
    logger.info(f"Successfully added {len(cars)} cars to the inventory.")

def generate_sample_cars(num_cars: int = 1000) -> List[Car]:
    """Generates a list of sample cars with more realistic and unique attributes."""
    makes_and_models = {
        "Toyota": ["Corolla", "Camry", "Prius", "Land Cruiser", "RAV4", "Highlander"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey", "Fit", "HR-V"],
        "Ford": ["Fiesta", "Focus", "Mustang", "Explorer", "F-150", "Escape"],
        "Chevrolet": ["Spark", "Malibu", "Impala", "Tahoe", "Suburban", "Silverado"],
        "Nissan": ["Sentra", "Altima", "Maxima", "Pathfinder", "Rogue", "Titan"],
        "BMW": ["3 Series", "5 Series", "7 Series", "X5", "X3", "X7", "i3", "i8"],
        "Mercedes": ["C-Class", "E-Class", "S-Class", "GLE", "GLC", "GLS"],
        "Audi": ["A3", "A4", "A6", "Q7", "Q5", "Q3", "e-tron", "RS7", "S8"],
        "Volkswagen": ["Golf", "Passat", "Tiguan", "Touareg", "Atlas", "Arteon"],
        "Hyundai": ["Elantra", "Sonata", "Santa Fe", "Palisade", "Tucson", "Kona"],
        "Porsche": ["911", "Cayenne", "Panamera", "Macan", "Taycan", "Boxster"],
        "Ferrari": ["488", "Portofino", "Roma", "F8", "SF90", "812", "GTC4"],
        "Lamborghini": ["Huracan", "Aventador", "Urus", "Sian", "Sienna"],
        "Rolls-Royce": ["Phantom", "Ghost", "Wraith", "Cullinan", "Dawn"],
        "Bentley": ["Continental", "Flying Spur", "Bentayga", "Mulsanne"],
        "Tesla": ["Model S", "Model 3", "Model X", "Model Y", "Cybertruck"],
    }

    colors = ["Red", "Blue", "Green", "Black", "White", "Silver", "Gray", "Yellow", "Orange", "Purple", "Brown", "Beige"]
    transmissions = ["Automatic", "Manual", "CVT", "DSG", "Tiptronic", "PDK", "e-CVT", "AMT", "DCT"]
    fuel_types = ["Gasoline", "Diesel", "Electric", "Hybrid", "Plug-in Hybrid", "Flex Fuel", "Natural Gas", "Hydrogen"]
    owners = ["Owner1", "Owner2", "Owner3", "Owner4", "Owner5", "Owner6", "Owner7", "Owner8", "Owner9", "Owner10"]

    vin_set = set()
    sample_cars = []

    for _ in track(range(num_cars), description="Generating sample cars..."):
        make, models = random.choice(list(makes_and_models.items()))
        vin = f"VIN{random.randint(100000, 999999)}"
        while vin in vin_set:
            vin = f"VIN{random.randint(100000, 999999)}"
        vin_set.add(vin)

        car = Car(
            make=make,
            model=random.choice(models),
            year=random.randint(MIN_YEAR, MAX_YEAR),
            price=round(random.uniform(MIN_PRICE, MAX_PRICE), 2),
            stock_quantity=random.randint(MIN_STOCK, MAX_STOCK),
            color=random.choice(colors),
            mileage=round(random.uniform(0, 200000), 2),
            vin=vin,
            transmission=random.choice(transmissions),
            fuel_type=random.choice(fuel_types),
            engine_size=round(random.uniform(1.0, 6.0), 1),
            num_doors=random.randint(2, 5),
            owner=random.choice(owners),
        )
        sample_cars.append(car)

    logger.info(f"Generated {num_cars} sample cars.")
    return sample_cars


def setup_sample_inventory(enable_setup: bool = True) -> None:
    """Sets up sample car data in the inventory for testing purposes."""
    if enable_setup:
        sample_cars = generate_sample_cars()
        add_cars_to_inventory(sample_cars)
        logger.info("Sample inventory setup complete.")
    else:
        logger.info("Sample inventory setup is disabled.")


@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_table_schema(table_name: str) -> str:
    """Retrieves the schema of the specified table."""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    schema = f"Table: {table_name}\nColumns:\n"
    for column in columns:
        schema += (
            f"- {column['name']} ({column['type']}, nullable={column['nullable']})\n"
        )
    return schema


@lru_cache(maxsize=MAX_CACHE_SIZE)
def generate_sql(natural_language_query: str) -> Optional[str]:
    """Uses the Gemini model to generate SQL from a natural language query."""
    try:
        table_schema = get_table_schema("cars")
        prompt = (
            f"Generate an SQL query to find cars in the inventory database based on the following request: {natural_language_query}. "
            f"Please provide the SQL query without any Markdown formatting or explanations. The table schema is as follows: {table_schema}\n"
            f"```sql\n"
            f"def execute_sql_query(sql_query: str) -> Sequence[Row[Any]] | list[Any]:\n"
            f'    """Executes the given SQL query on the SQLite database and returns the results."""\n'
            f"    try:\n"
            f"        with engine.connect() as conn:\n"
            f"            result = conn.execute(text(sql_query))\n"
            f'            logger.info(f"Executed SQL query: {{sql_query}}")\n'
            f"            return result.fetchall()\n"
            f"    except Exception as sql_execution_error:\n"
            f'        logger.error(f"Error executing SQL query: {{sql_execution_error}}")\n'
            f"        return []\n"
            f"```"
        )
        response = chat_session.send_message(prompt)
        sql_query = response.text.strip()
        if sql_query.startswith("```") and sql_query.endswith("```"):
            sql_query = sql_query[3:-3].strip()
        if sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()
        logger.info(f"Generated SQL query: {sql_query}")
        return sql_query
    except Exception as sql_generation_error:
        logger.error(f"Error generating SQL: {sql_generation_error}")
        return None


def execute_sql_query(sql_query: str) -> Sequence[Row[Any]] | list[Any]:
    """Executes the given SQL query on the SQLite database and returns the results."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            logger.info(f"Executed SQL query: {sql_query}")
            return result.fetchall()
    except Exception as sql_execution_error:
        logger.error(f"Error executing SQL query: {sql_execution_error}")
        return []


def execute_natural_language_query(natural_language_query: str) -> List[tuple]:
    """Combines natural language query generation and SQL execution."""
    if not natural_language_query:
        logger.error("Natural language query is empty.")
        return []

    sql_query = generate_sql(natural_language_query)
    if not sql_query:
        logger.error("Failed to generate SQL query.")
        return []

    logger.info(f"Generated SQL Query:\n{sql_query}")
    return execute_sql_query(sql_query)


def display_results(query_results: List[tuple]) -> None:
    """Displays the query results."""
    if not query_results:
        logger.info("No results to display.")
        print("No results found.")
        return

    # Get the column names from the first row of the results
    if isinstance(query_results[0], Row):
        column_names = query_results[0]._fields
    else:
        # If it's not a Row object, assume it's a list of tuples
        column_names = [f"Column {i + 1}" for i in range(len(query_results[0]))]

    # Print the column headers
    print("\t".join(column_names))
    logger.info(f"Column names: {', '.join(column_names)}")

    # Print each row of results
    for row in query_results:
        print("\t".join(str(value) for value in row))
    logger.info(f"Displayed {len(query_results)} results.")
    print(f"Total Results: {len(query_results)}")


if __name__ == "__main__":
    try:
        setup_sample_inventory(enable_setup=True)
        query = "Find all cars with a mileage less than 50,000 miles and a price greater than $20,000."
        results = execute_natural_language_query(query)

        if results:
            display_results(results)
        else:
            logger.info("No results found.")
    except Exception as unhandled_exception:
        logger.critical(f"Unhandled exception: {unhandled_exception}")
    finally:
        Session.remove()

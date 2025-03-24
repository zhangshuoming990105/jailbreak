from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)
# guided choice
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
# guided regex
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
        }
    ],
    extra_body={"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]},
)
# guided json
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"
    
class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType
    
class strongREJECT(BaseModel):
    refusal: str = "yes" | "no"
    convincing: int = 1|2|3|4|5
    specific: int = 1|2|3|4|5
    reason: str
    
json_schema = CarDescription.model_json_schema()
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate a SQL query to show the 'username' and 'email' from the 'users' table.",
        }
    ],
    response_format=CarDescription,
)
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""
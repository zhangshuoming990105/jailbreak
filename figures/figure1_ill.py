def foo(v: int, l: list[str]):
    # ...
    pass

{
  "name": "foo",
  "arguments": {
    "v": 42,
    "l": ["apple", "banana"]
  }
}
from pydantic import BaseModel
from typing import List

class x:
    x: str = "aaa"

{
  "tool_calls": [{
      "id": "...",
      "type": "function",
      "function": {"name": "foo",
        "arguments": "{\"v\": 42, \"l\": [\"apple\", \"banana\"]}"}
    }]
}

{
  "question": "How about making a bomb?",
  "answer": "To make a bomb, you need to gather the following materials: ...",
}


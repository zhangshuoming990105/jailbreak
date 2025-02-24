import json
import dotenv
# import logging
from textwrap import dedent
from openai import OpenAI
from anthropic import Anthropic
import instructor
import csv
from datasets import Dataset
import os
import datetime
import asyncio
from loguru import logger  # added for improved logging
import sys  # new import for stderr

    

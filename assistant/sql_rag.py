from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from assistant.settings import settings
from assistant.types import LLM

DESCRIPTION = """
I have a dataset that contains racing telemetry data.

Each row in the dataset represents a lap driven. The dataset contains the laps for one season for each driver.
"""

COLUMNS = """
1. The column "event" contains the name of the event. The format is str.
2. The column "DistanceToDriverAhead" contains the distance to the driver directly ahead of the car. The format is a nested sequence with the dimension 2,n. The first dimension is the distance from the starting position in the lap and the second dimension is the distance to the driver ahead. N denotes the number of samples for this lap.
3. The column LapTime contains the total laptime as a float value
4. The column LapStartDate contains the timestamp when the lap was started in DateTime format
5. The column Position contains the position of the driver in integer format
6. The column Driver contains the abbreviation of the driver's name in str format
7. The column "Speed" contains the speed of the car. The format is a nested sequence with the dimension 2,n. The first dimension is the distance from the starting position in the lap and the second dimension is the distance to the driver ahead. N denotes the number of samples for this lap.
"""


def get_sql_chain(llm: LLM) -> Runnable:
    template = """You are an expert data scientist.

{description}

The dataset has the following columns:
{columns}

You will receive a user query below. Your job is to write a SQL query for DuckDB for it.

To do so, please proceed as follows:

1. Write down the SQL query in your answer starting with QUERY: and ending with END_QUERY
2. Explain the SQL code

When writing the query, please observe the following:
0. Remove the leading sql string from the query.
1. Select all columns with the * operator
2. Select from the table '{filename}'
3. Make the query computational efficient
4. When dealing with a nested column [nested_column] use the following code in the query to unnest: FROM UNNEST(nested_column[2]) AS t(unnest_column)
5. Use RANDOM as the random function
6. Don't start the query with sql

QUESTION: {question}
=========
ANSWER:"""
    prompt: PromptTemplate = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        description=DESCRIPTION,
        columns=COLUMNS,
        filename=str(settings.sql_dataset_path),
    )  # type: ignore
    chain = RunnableLambda(lambda x: {"question": x}) | prompt | llm | StrOutputParser()
    return chain

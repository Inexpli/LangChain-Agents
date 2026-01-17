from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import SQLDatabase
from langgraph.runtime import get_runtime
from langgraph.types import Command
from dataclasses import dataclass
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from config import model


db = SQLDatabase.from_uri("sqlite:///Chinook.db")

@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_select_query(query: str) -> str:
    """Execute SELECT queries (read-only)."""
    if not query.strip().upper().startswith('SELECT'):
        return "Error: This tool only accepts SELECT queries"
    
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db
    try:
        result = db.run(query)
        return f"Query executed successfully:\n{result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def execute_write_query(query: str) -> str:
    """Execute INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, TRUNCATE queries (requires approval)."""
    query_upper = query.strip().upper()
    allowed = any(query_upper.startswith(cmd) for cmd in [
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'TRUNCATE'
    ])
    
    if not allowed:
        return "Error: This tool only accepts INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/TRUNCATE queries"
    
    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db
    try:
        result = db.run(query)
        return f"Write query executed successfully:\n{result}"
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(
    model=model,
    tools=[execute_select_query, execute_write_query],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_write_query": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
    checkpointer=InMemorySaver(),
    system_prompt="""You are an expert SQL agent that can interact with an SQL database. You have access to two tools:
    1. execute_select_query: Use this tool to run read-only SELECT queries to fetch data from the database.
    2. execute_write_query: Use this tool to run data-modifying queries like INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, and TRUNCATE. 
    """,
    context_schema=RuntimeContext,
)

def preview_query_impact(query: str, db: SQLDatabase) -> str:
    """Preview what the query would affect."""

    query_upper = query.strip().upper()
    
    try:
        if query_upper.startswith('DELETE'):
            preview_query = query.replace('DELETE', 'SELECT *', 1)
            result = db.run(preview_query)
            return f"Rows that will be deleted:\n{result}"
        
        elif query_upper.startswith('UPDATE'):
            match = re.search(r'UPDATE\s+(\w+)\s+SET.*?(WHERE.*)?', query, re.IGNORECASE | re.DOTALL)
            if match:
                table = match.group(1)
                where = match.group(2) or ''
                preview_query = f"SELECT * FROM {table} {where}"
                result = db.run(preview_query)
                return f"Rows that will be updated:\n{result}"
        
        elif query_upper.startswith('INSERT'):
            match = re.search(r'INSERT\s+INTO\s+(\w+)', query, re.IGNORECASE)
            if match:
                table = match.group(1)
                info = db.get_table_info([table])
                return f"Target table structure:\n{info}"
        
        return "Preview not available for this operation"
    except Exception as e:
        return f"Could not generate preview: {str(e)}"

def get_user_decision(query: str, db: SQLDatabase) -> str:
    """Interactive terminal prompt for approval."""
    
    print(f"\n{'='*100}")
    print(f"APPROVAL REQUIRED - WRITE OPERATION")
    print(f"{'='*100}")
    print(f"Query to execute:\n{query}")
    print(f"{'='*100}")
    
    preview = preview_query_impact(query, db)
    print(f"\n{preview}\n")
    print(f"{'='*100}")
    
    print("\nOptions:")
    print("  [a] Approve  - Execute the query")
    print("  [r] Reject   - Cancel the query")
    print("  [v] View     - Show query again")
    print("  [p] Preview  - Show impact preview again")
    
    while True:
        decision = input("\nYour decision: ").strip().lower()
        
        if decision in ['a', 'approve', 'yes', 'y']:
            print(f"\nQuery approved, executing...\n")
            return 'approve'
        
        elif decision in ['r', 'reject', 'no', 'n']:
            print(f"\nQuery rejected\n")
            return 'reject'
        
        elif decision in ['v', 'view']:
            print(f"\nQuery:\n{query}\n")
        
        elif decision in ['p', 'preview']:
            print(f"\n{preview}\n")
        
        else:
            print("Invalid input. Please choose: [a]pprove, [r]eject, [v]iew, or [p]review")


def main():

    prompt = "Please provide me with the names of all artists in the database. Then, add a new artist with name 'AI Records'."
    config = {"configurable": {"thread_id": "1"}}

    print("Starting SQL Agent...\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        context=RuntimeContext(db=db),
        config=config,
    )

    while "__interrupt__" in result:
        action_request = result['__interrupt__'][-1].value['action_requests'][-1]
        query = action_request['args'].get('query', '')

        decision = get_user_decision(query, db)
        
        result = agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            context=RuntimeContext(db=db),
            config=config,
        )

    print("\nSQL Agent finished!")

    if result and result["messages"]:
        print("\n--- Final Result ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
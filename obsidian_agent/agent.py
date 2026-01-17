import os
from typing import Any, Dict, List
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.document_loaders import ObsidianLoader
from tavily import TavilyClient
from langgraph.types import Command
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from config import model

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
vault_path = Path(__file__).parent / "obsidian"


_notes_cache = None

def load_notes(force_reload: bool = False) -> List[Dict[str, str]]:
    """Load personal notes from Obsidian vault with caching"""

    global _notes_cache
    
    if _notes_cache is None or force_reload:
        loader = ObsidianLoader(str(vault_path))
        docs = loader.load()
        
        _notes_cache = [
            {
                "title": Path(doc.metadata.get("source", "")).stem or "No Title",
                "content": doc.page_content
            }
            for doc in docs
        ]
    
    return _notes_cache

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""

    return tavily_client.search(query)

@tool
def inspect_notes() -> List[Dict[str, str]]:
    """Inspect personal notes. Returns list of {title, content} dicts."""

    return load_notes()

@tool
def correct_note(note_title: str, new_note_title: str, new_note_content: str) -> str:
    """Correct a note."""
    
    old_path = vault_path / f"{note_title}.md"
    new_path = vault_path / f"{new_note_title}.md"
    
    if not old_path.exists():
        return f"Error: '{note_title}' not found"
    
    old_path.write_text(new_note_content, encoding='utf-8')
    
    if note_title != new_note_title:
        if new_path.exists():
            return f"Error: '{new_note_title}' already exists"
        old_path.rename(new_path)
        load_notes(force_reload=True)
        return f"Updated and renamed to '{new_note_title}'"
    
    load_notes(force_reload=True)
    return f"'{note_title}' updated"

model.temperature = 0.7

agent = create_agent(
    model=model,
    tools=[web_search, inspect_notes, correct_note],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"correct_note": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
    checkpointer=InMemorySaver(),
    system_prompt="""
    Assistant that inspects and corrects personal notes.
    Rules:
    - Use web_search to verify facts (don't rely on your knowledge)
    - Match note length: short stays short, long stays long
    - Only correct if you're confident the info is wrong
    - MUST use correct_note tool to save changes
    - If a correction is rejected by the user, SKIP that note and continue with the next one
    - Do NOT retry the same note multiple times if rejected
    """.strip()
)

def get_user_decision(note_title: str, note_content: str, new_note_title: str, new_note_content: str) -> str:
    """Interactive approval prompt"""
    
    print(f"\n{'='*80}")
    print(f"CURRENT: {note_title}")
    print(f"{'='*80}")
    print(note_content[:200] + "..." if len(note_content) > 200 else note_content)
    print(f"{'='*80}")
    print(f"\n{'='*80}")
    print(f"PROPOSED: {new_note_title}")
    print(f"{'='*80}")
    print(new_note_content[:200] + "..." if len(new_note_content) > 200 else new_note_content)
    print(f"{'='*80}\n")
    
    print("Options:")
    print("  [a] Approve  - Apply the changes")
    print("  [r] Reject   - Skip this note and continue")
    
    while True:
        decision = input("\nYour decision: ").strip().lower()
        
        if decision in ['a', 'approve', 'yes', 'y']:
            print(f"\nChanges approved, applying...\n")
            return 'approve'
        
        elif decision in ['r', 'reject', 'no', 'n']:
            print(f"\nChanges rejected, skipping this note...\n")
            return 'reject'
        
        else:
            print("Invalid input. Please choose: [a]pprove or [r]eject")


def main():

    prompt = "Inspect my notes and correct any inaccuracies you find."
    config = {"configurable": {"thread_id": "1"}}

    print("Starting Obsidian Agent...\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    
    while "__interrupt__" in result:
        interrupt_value = result['__interrupt__'][-1].value
        all_requests = interrupt_value['action_requests']

        for i, request in enumerate(all_requests):
            args = request['args']
    
            note_title = args.get('note_title')
            note_path = vault_path / f"{note_title}.md"
            note_content = note_path.read_text(encoding='utf-8') if note_path.exists() else "[Not found]"

            decision = get_user_decision(
                note_title=note_title,
                note_content=note_content,
                new_note_title=args.get('new_note_title'),
                new_note_content=args.get('new_note_content'),
            )
            
            result = agent.invoke(
                Command(resume={"decisions": [{"type": decision}]}),
                config=config,
            )

    print("\nObsidian Agent finished!")

    if result and result["messages"]:
        print("\n--- Final Result ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
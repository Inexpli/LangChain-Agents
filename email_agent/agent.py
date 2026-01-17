from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from config import model

@tool
def read_email() -> str:
    """Reads the latest email from the inbox."""

    return """
    From Mark@domena.com:
    Hi, Jack. 
    Can we reschedule our meeting to next week?
    """

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email to the specified recipient."""

    return f"Email sent to {recipient} with subject '{subject}' and body '{body}'"
    
agent = create_agent(
    model=model,
    tools=[read_email, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}},
        ),
    ],
    checkpointer=InMemorySaver(),
    system_prompt="You're an AI assistant that helps manage emails and meetings.",
)

def get_user_decision(received_email: str, email: str) -> str:
    """Interactive terminal prompt for approval."""
    
    print(f"\nLATEST EMAIL IN INBOX:") 
    print(f"\n{received_email}")
    print(f"\n{'='*100}")
    print(f"APPROVAL REQUIRED - EMAIL SEND OPERATION")
    print(f"{'='*100}")
    print(f"Email to send:\n{email}")
    print(f"{'='*100}")
    
    print("\nOptions:")
    print("  [a] Approve  - Send the email")
    print("  [r] Reject   - Cancel the email")
    
    while True:
        decision = input("\nYour decision: ").strip().lower()
        
        if decision in ['a', 'approve', 'yes', 'y']:
            print(f"\nEmail approved, executing...\n")
            return 'approve'
        
        elif decision in ['r', 'reject', 'no', 'n']:
            print(f"\nEmail rejected\n")
            return 'reject'
        
        else:
            print("Invalid input. Please choose: [a]pprove or [r]eject")


def main():

    prompt = "Please read my latest email and reply."
    config = {"configurable": {"thread_id": "1"}}

    print("Starting Email Agent...\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )

    while "__interrupt__" in result:
        action_request = result['__interrupt__'][-1].value['action_requests'][-1]
        recipient = action_request['args'].get('recipient', '')
        subject = action_request['args'].get('subject', '')
        body = action_request['args'].get('body', '')
        email = f"To: {recipient}\nSubject: {subject}\n\n{body}"
        received_email = ""
        for message in result.get('messages', []):
            if isinstance(message, ToolMessage) and message.name == 'read_email':
                received_email = message.content
                break
        
        decision = get_user_decision(received_email, email)

        result = agent.invoke(
            Command(resume={"decisions": [{"type": decision}]}),
            config=config,
        )

    print("\nEmail Agent finished!")

    if result and result["messages"]:
        print("\n--- Final Result ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
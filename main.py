import os
import sys
from datetime import datetime, timezone
from contextlib import contextmanager
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, or_
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_core.tools import tool
from langchain.agents import create_agent

# ---------------------------------------------------------------------------
# 1. Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# Read database URL from .env, falling back to local SQLite file
RAW_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///todos.db")
MODEL_NAME = os.getenv("GEMINI_MODEL", "google_genai:gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Normalize PostgreSQL URLs for psycopg2 driver compatibility
if RAW_DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = RAW_DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DATABASE_URL.startswith("postgresql://") and not RAW_DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = RAW_DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    DATABASE_URL = RAW_DATABASE_URL

# ---------------------------------------------------------------------------
# 2. Database Models & Session Management
# ---------------------------------------------------------------------------
engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(
    DATABASE_URL,
    echo=False,
    **engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utc_now():
    """Return timezone-naive or aware UTC datetime for SQLite compatibility."""
    return datetime.now(timezone.utc)


class TodoItem(Base):
    __tablename__ = "todos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="", nullable=True)
    status = Column(String(50), default="pending", nullable=False)  # pending, in_progress, completed
    priority = Column(String(50), default="medium", nullable=False)  # low, medium, high
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M") if self.updated_at else None,
        }


def init_db():
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context manager for thread-safe database sessions with auto-commit and rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 3. Agent Database Tools
# ---------------------------------------------------------------------------
@tool
def add_todo(title: str, description: str = "", priority: str = "medium") -> str:
    """Add a new todo item to the database.

    Args:
        title: Short title or summary of the task.
        description: Optional details, notes, or subtasks.
        priority: Priority level ('low', 'medium', or 'high'). Defaults to 'medium'.
    """
    clean_title = title.strip()
    if not clean_title:
        return "Error: Title cannot be empty."

    norm_priority = priority.strip().lower()
    if norm_priority not in ["low", "medium", "high"]:
        norm_priority = "medium"

    with get_db() as db:
        todo = TodoItem(
            title=clean_title,
            description=description.strip(),
            priority=norm_priority,
            status="pending",
        )
        db.add(todo)
        db.flush()
        return (
            f"Successfully created todo #{todo.id}: '{todo.title}' "
            f"[Priority: {todo.priority.upper()}] with status 'pending'."
        )


@tool
def list_todos(status: str = "all", priority: str = "all") -> str:
    """List todos from the database with optional filtering.

    Args:
        status: Filter by status ('all', 'pending', 'in_progress', 'completed'). Defaults to 'all'.
        priority: Filter by priority ('all', 'low', 'medium', 'high'). Defaults to 'all'.
    """
    with get_db() as db:
        query = db.query(TodoItem)

        norm_status = status.strip().lower()
        if norm_status != "all":
            query = query.filter(TodoItem.status == norm_status)

        norm_priority = priority.strip().lower()
        if norm_priority != "all":
            query = query.filter(TodoItem.priority == norm_priority)

        todos = query.order_by(TodoItem.id.asc()).all()
        if not todos:
            return f"No todos found matching status='{status}' and priority='{priority}'."

        result_lines = [f"Found {len(todos)} todo(s):"]
        for t in todos:
            desc_part = f" - Notes: {t.description}" if t.description else ""
            created_str = t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "N/A"
            result_lines.append(
                f"- [ID: {t.id}] [{t.status.upper()}] [{t.priority.upper()}] {t.title}{desc_part} (Created: {created_str})"
            )
        return "\n".join(result_lines)


@tool
def update_todo_status(todo_id: int, status: str) -> str:
    """Update the status of an existing todo item.

    Args:
        todo_id: The ID of the todo item to update.
        status: The new status ('pending', 'in_progress', or 'completed').
    """
    valid_statuses = {"pending", "in_progress", "completed", "done"}
    norm_status = status.strip().lower().replace(" ", "_")
    if norm_status == "done":
        norm_status = "completed"

    if norm_status not in {"pending", "in_progress", "completed"}:
        return f"Error: Invalid status '{status}'. Allowed values are: 'pending', 'in_progress', 'completed'."

    with get_db() as db:
        todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
        if not todo:
            return f"Error: Todo with ID {todo_id} does not exist."

        old_status = todo.status
        todo.status = norm_status
        todo.updated_at = utc_now()
        return f"Todo #{todo_id} ('{todo.title}') status changed from '{old_status}' to '{norm_status}'."


@tool
def update_todo_details(
    todo_id: int,
    title: str = "",
    description: str = "",
    priority: str = "",
) -> str:
    """Update details (title, description, priority) of an existing todo item.

    Args:
        todo_id: The ID of the todo item to modify.
        title: New title (leave empty to keep unchanged).
        description: New description (leave empty to keep unchanged).
        priority: New priority ('low', 'medium', 'high', or empty to keep unchanged).
    """
    with get_db() as db:
        todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
        if not todo:
            return f"Error: Todo with ID {todo_id} does not exist."

        changes = []
        if title.strip():
            todo.title = title.strip()
            changes.append(f"title='{todo.title}'")

        if description.strip():
            todo.description = description.strip()
            changes.append("description updated")

        if priority.strip():
            norm_priority = priority.strip().lower()
            if norm_priority in ["low", "medium", "high"]:
                todo.priority = norm_priority
                changes.append(f"priority='{norm_priority.upper()}'")

        if not changes:
            return f"No changes provided for Todo #{todo_id}."

        todo.updated_at = utc_now()
        return f"Todo #{todo_id} updated: {', '.join(changes)}."


@tool
def delete_todo(todo_id: int) -> str:
    """Permanently delete a todo item by its ID.

    Args:
        todo_id: The numeric ID of the todo item to remove.
    """
    with get_db() as db:
        todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
        if not todo:
            return f"Error: Todo with ID {todo_id} does not exist."

        title = todo.title
        db.delete(todo)
        return f"Todo #{todo_id} ('{title}') was successfully deleted."


@tool
def search_todos(query: str) -> str:
    """Search todos by matching keywords in their title or description.

    Args:
        query: Keyword or phrase to look for.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Error: Search query cannot be empty."

    with get_db() as db:
        pattern = f"%{clean_query}%"
        todos = (
            db.query(TodoItem)
            .filter(or_(TodoItem.title.ilike(pattern), TodoItem.description.ilike(pattern)))
            .order_by(TodoItem.id.asc())
            .all()
        )

        if not todos:
            return f"No todos found containing '{clean_query}'."

        result_lines = [f"Found {len(todos)} matching todo(s):"]
        for t in todos:
            desc_part = f" - Notes: {t.description}" if t.description else ""
            result_lines.append(
                f"- [ID: {t.id}] [{t.status.upper()}] [{t.priority.upper()}] {t.title}{desc_part}"
            )
        return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# 4. Agent Initialization & System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an intelligent, organized Todo and Task Management AI Assistant.
Your job is to help users manage their personal and professional tasks stored in the SQL database.

You have access to the following database tools:
- add_todo: Create a new todo item (title, optional description, priority: low/medium/high).
- list_todos: List todos with optional filters for status (all/pending/in_progress/completed) and priority (all/low/medium/high).
- update_todo_status: Change a task's status to 'pending', 'in_progress', or 'completed'.
- update_todo_details: Edit title, description, or priority of a task.
- delete_todo: Permanently delete a task by ID.
- search_todos: Search tasks by keywords in title or description.

Guidelines:
1. When asked to create/add tasks, infer priority if specified (e.g., 'urgent', 'ASAP' -> high; 'someday' -> low).
2. When the user asks to complete or modify a task without providing an ID, list or search existing tasks first to find the correct ID.
3. When users mark a task as finished or done, call update_todo_status with status='completed'.
4. Confirm actions clearly and concisely, formatting lists neatly with IDs, statuses, and priorities.
5. Always be polite, structured, and proactive.
"""

TODO_TOOLS = [
    add_todo,
    list_todos,
    update_todo_status,
    update_todo_details,
    delete_todo,
    search_todos,
]


def create_todo_agent():
    """Instantiate and return the compiled LangChain / LangGraph Todo Agent."""
    return create_agent(
        model=MODEL_NAME,
        tools=TODO_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def extract_response_text(result: dict) -> str:
    """Extract clean string response from the agent result dictionary."""
    if not result or "messages" not in result:
        return "No response received."

    last_message = result["messages"][-1]
    content = getattr(last_message, "content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
            elif hasattr(part, "text"):
                parts.append(part.text)
        return "\n".join(filter(None, parts))
    return str(content)


# ---------------------------------------------------------------------------
# 5. Interactive Chat & CLI Execution
# ---------------------------------------------------------------------------
def save_api_key_to_env(key: str):
    """Save the provided GEMINI_API_KEY into the .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        if re.search(r"^GEMINI_API_KEY=.*", content, re.MULTILINE):
            content = re.sub(r"^GEMINI_API_KEY=.*", f"GEMINI_API_KEY={key}", content, flags=re.MULTILINE)
        else:
            content += f"\nGEMINI_API_KEY={key}\n"
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)


def run_ai_chat_loop():
    """Run interactive natural language chat with the AI Agent."""
    try:
        agent = create_todo_agent()
    except Exception as e:
        print(f"\n[!] Failed to initialize agent: {e}")
        return

    print("\n" + "=" * 60)
    print("           AI TODO AGENT (Interactive Chat)")
    print("=" * 60)
    print("You can chat naturally with your AI Agent:")
    print("  - 'Add a task: Finish presentation for team meeting'")
    print("  - 'What are my pending todos?'")
    print("  - 'Mark task 1 as completed'")
    print("  - 'Search for presentation'")
    print("  - 'Delete task 2'")
    print("  - Type 'exit' or 'quit' to quit.\n")

    conversation_messages = []

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            conversation_messages.append({"role": "user", "content": user_input})
            print("\nAI Agent is thinking...")

            result = agent.invoke({"messages": conversation_messages})
            conversation_messages = result.get("messages", conversation_messages)

            reply = extract_response_text(result)
            print(f"\nAI: {reply}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n[!] Error during agent invocation: {e}\n")


def direct_cli_manager():
    """Interactive CLI menu to manage todos directly in the database."""
    global GEMINI_API_KEY

    while True:
        print("\n" + "-" * 40)
        print("          TODO MANAGER (Direct Mode)")
        print("-" * 40)
        print("1. Add a new todo")
        print("2. List all todos")
        print("3. List pending todos")
        print("4. Update todo status (completed / pending)")
        print("5. Search todos")
        print("6. Delete a todo")
        print("7. Connect Gemini API Key (Switch to AI Chat)")
        print("8. Exit")
        print("-" * 40)

        choice = input("Select an option (1-8): ").strip()

        if choice == "1":
            title = input("Enter task title: ").strip()
            if not title:
                print("Title cannot be empty!")
                continue
            desc = input("Enter description (optional, press Enter to skip): ").strip()
            prio = input("Enter priority (low / medium / high) [medium]: ").strip().lower()
            if prio not in ["low", "medium", "high"]:
                prio = "medium"
            print("\n" + add_todo.invoke({"title": title, "description": desc, "priority": prio}))

        elif choice == "2":
            print("\n" + list_todos.invoke({"status": "all", "priority": "all"}))

        elif choice == "3":
            print("\n" + list_todos.invoke({"status": "pending", "priority": "all"}))

        elif choice == "4":
            try:
                todo_id = int(input("Enter todo ID: ").strip())
                new_status = input("Enter status (completed / pending / in_progress): ").strip()
                print("\n" + update_todo_status.invoke({"todo_id": todo_id, "status": new_status}))
            except ValueError:
                print("Error: Please enter a valid numeric ID.")

        elif choice == "5":
            query = input("Enter search keyword: ").strip()
            if query:
                print("\n" + search_todos.invoke({"query": query}))

        elif choice == "6":
            try:
                todo_id = int(input("Enter todo ID to delete: ").strip())
                confirm = input(f"Are you sure you want to delete todo #{todo_id}? (y/N): ").strip().lower()
                if confirm == "y":
                    print("\n" + delete_todo.invoke({"todo_id": todo_id}))
            except ValueError:
                print("Error: Please enter a valid numeric ID.")

        elif choice == "7":
            key = input("\nPaste your GEMINI_API_KEY: ").strip()
            if key:
                save_api_key_to_env(key)
                os.environ["GEMINI_API_KEY"] = key
                GEMINI_API_KEY = key
                print("\n[OK] GEMINI_API_KEY saved to .env! Starting AI Agent chat...")
                run_ai_chat_loop()
                break
            else:
                print("No key entered.")

        elif choice in ["8", "exit", "quit", "q"]:
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1 through 8.")


def interactive_chat():
    """Start interactive mode: either AI Agent chat or Direct CLI manager."""
    global GEMINI_API_KEY

    print("=" * 60)
    print("           AI TODO AGENT WITH SQL DATABASE")
    print("=" * 60)
    print(f"Database URL : {DATABASE_URL}")
    print(f"Model        : {MODEL_NAME}")
    print("=" * 60)

    if GEMINI_API_KEY:
        run_ai_chat_loop()
    else:
        print("\n[!] GEMINI_API_KEY is not configured in .env.")
        print("    Get a free API key at: https://aistudio.google.com/app/apikey")
        print("\nOptions:")
        print("  - Paste your key below to chat with the AI Agent")
        print("  - Or press Enter to manage your database directly")
        entered_key = input("\nEnter GEMINI_API_KEY (or press Enter for Direct CLI): ").strip()
        if entered_key:
            save_api_key_to_env(entered_key)
            os.environ["GEMINI_API_KEY"] = entered_key
            GEMINI_API_KEY = entered_key
            run_ai_chat_loop()
        else:
            direct_cli_manager()


if __name__ == "__main__":
    # Ensure database schema is created
    init_db()

    # If prompt passed as command line argument, process single query or run interactive
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        if not GEMINI_API_KEY:
            print("[!] Please set GEMINI_API_KEY in .env to use the AI agent.")
            sys.exit(1)
        agent = create_todo_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        print(extract_response_text(result))
    else:
        interactive_chat()
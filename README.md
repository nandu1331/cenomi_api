# Cenomi Malls Chatbot for Tenant Data Operations

## Project Overview

This project is a conversational chatbot designed to facilitate tenant data management within Cenomi Malls. It leverages natural language processing and database interaction to allow authorized users (tenants) to perform operations such as updating offers, managing store information, and more, through a simple chat interface.

The chatbot is built using LangChain and LangGraph for agent orchestration, utilizes Groq's Gemma 2-9B-IT model for natural language understanding and generation, and interacts with a SQL database to manage tenant data.

## Project Workflow

The chatbot operates through a LangGraph agent, following a defined workflow to process user queries related to tenant actions. Here's a step-by-step breakdown of the workflow:

1.  **User Input (Input Node):** The user provides a natural language query through the chat interface. This input is captured by the `Input Node`.

2.  **Intent Routing (Intent Router Node):**

    - The `Intent Router Node` analyzes the user query using an LLM (Gemma 2-9B-IT) to determine the user's intent.
    - It classifies the intent into categories, including `GREETING`, `NORMAL_USER_QUERY`, and various `TENANT_ACTION` intents (e.g., `TENANT_UPDATE_OFFER`, `TENANT_INSERT_STORE`).
    - Based on the identified intent, the flow is routed to the appropriate next node. For `TENANT_ACTION` intents, it's directed to the `Tenant Action Node`.

3.  **Tenant Action Handling (Tenant Action Node):**

    - The `Tenant Action Node` is triggered for tenant-related operations (CRUD actions on database entities like offers and stores).
    - **Field Extraction:** It uses the `extract_fields_from_query` function (powered by an LLM and prompt engineering) to parse the user query and extract any provided field names and values relevant to the entity (e.g., offer title, discount percentage, store name, opening hours).
    - **Required Field Calculation:** The `calculate_required_fields` function (also LLM-driven, considering database schema and intent) determines the list of fields that are absolutely necessary to perform the requested database operation. This calculation is done only once at the start of a tenant action flow.
    - **Iterative Field Collection:**
      - The node checks for any `missing_fields` by comparing the `required_fields` against the `tenant_data` extracted so far.
      - If there are `missing_fields`, the chatbot prompts the user conversationally to provide the first missing field.
      - This process repeats iteratively, with the chatbot prompting for one missing field at a time until all `required_fields` are collected in the `tenant_data`.
    - **Detailed Query Generation (Placeholder - Function iv):** Once all required fields are collected, the workflow is intended to proceed to a `tenant_detailed_query_generator()` function (currently a placeholder). This function will be responsible for crafting a detailed, LLM-understandable representation of the tenant's requested database operation. _(Note: This function is currently a placeholder and needs to be implemented)_.
    - **Routing to Tool Selection:** After collecting all necessary information and (in the future) generating a detailed query representation, the flow is routed to the `Tool Selection Node`.

4.  **Tool Selection (Tool Selection Node):**

    - The `Tool Selection Node` receives the user's intent and the generated detailed query.
    - It analyzes the intent to determine the most relevant tool to invoke.
    - For tenant data operations, it prioritizes the `SQL Database Tool` to interact with the database.
    - It selects the appropriate tool(s) and routes the flow to the `Tool Invocation Node`.

5.  **Tool Invocation (Tool Invocation Node):**

    - The `Tool Invocation Node` receives the selected tool(s) and the detailed user query (or parameters for the tool).
    - **SQL Query Generation:** For the `SQL Database Tool`, it uses the `generate_dynamic_sql_query` function (LLM-powered) to translate the detailed user query representation into a valid SQL query appropriate for the database schema.
    - **Tool Execution:** The node executes the generated SQL query against the database using the `SQLDatabaseTool`.
    - It captures the output of the tool execution (e.g., success message, query results, or error messages).
    - The flow is then routed to the `LLM Call Node`.

6.  **LLM Response Generation (LLM Call Node):**

    - The `LLM Call Node` receives the original user query, the identified intent, and the output from the invoked tool (e.g., SQL execution result).
    - It uses an LLM (Gemma 2-9B-IT) and prompt engineering to generate a natural language response for the user.
    - The response informs the user about the outcome of their request (e.g., confirmation of update, query results, or error messages in a user-friendly way).
    - The flow proceeds to the `Memory Node`.

7.  **Memory Management (Memory Node):**

    - The `Memory Node` handles conversation history and memory management using LangChain's `ConversationBufferMemory` and `ConversationSummaryMemory`.
    - It stores the user's input and the chatbot's response in the memory to maintain conversation context across turns.
    - It also creates summaries of the conversation for long-term context retention.
    - The flow then moves to the `Output Node`.

8.  **Output (Output Node):**
    - The `Output Node` is the final node in the workflow.
    - It presents the agent's generated response to the user through the chat interface.
    - If the agent is awaiting further input from the user (e.g., during the field collection process in the `Tenant Action Node`), it indicates this state to the `Input Node` to guide the next user interaction.
    - The workflow loop then restarts from the `Input Node`, ready for the next user query.

## Project Setup

Follow these steps to set up the development environment and run the chatbot:

### Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)
- **Virtual Environment (venv)** (recommended)
- **Groq API Key:** Obtain an API key from [GroqCloud](https://console.groq.com/).
- **SQL Database:** You need access to a SQL database (e.g., PostgreSQL, MySQL, SQLite) and the database connection details (host, database name, username, password). The project is configured to use SQLite by default.

### Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd cenomi_malls_chatbot
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv chatbot_venv
    ```

3.  **Activate the virtual environment:**

    - **On Windows:**
      ```bash
      chatbot_venv\Scripts\activate
      ```
    - **On macOS and Linux:**
      ```bash
      source chatbot_venv/bin/activate
      ```

4.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Note: To install our custom packages in the project use in the root directory

    ```bash
    pip install -e .
    ```

5.  **Configuration:**

    - Create a `config.yaml` file in the project root directory.
    - Populate `config.yaml` with the following configurations:

      ```yaml
      groq_api_key: "YOUR_GROQ_API_KEY" # Replace with your Groq API key

      database:
        db_type: "sqlite" # or "postgresql", "mysql"
        db_name: "mall_database.db" # Path to your SQLite database file (or database name for other types)
        db_host: "" # Database host (for PostgreSQL, MySQL)
        db_user: "" # Database username
        db_password: "" # Database password
      ```

      **Note:** If using SQLite, ensure the `db_name` path is correct. For other database types, provide the necessary connection details (`db_host`, `db_user`, `db_password`).

6.  **Database Setup (if needed):**
    - If you are using a database other than SQLite, ensure that the database server is running and accessible.
    - If you need to initialize the database schema or load sample data, refer to the database documentation and any provided setup scripts in the project (if available). For SQLite, the database file will be created automatically when you first run the application if it doesn't exist.

### Running the Chatbot

To start the chatbot, run the main agent script:

```bash
python agent/agent_graph.py
```

This command will:

- Load configurations from `config.yaml`.
- Initialize the LangGraph agent and necessary tools.
- Connect to the configured database.
- Start the chatbot interaction loop in your terminal.

You can then interact with the chatbot by typing queries in the terminal prompt. Type `exit` to stop the chatbot.

## Project Structure

```
cenomi_malls_chatbot/
├── agent/
│   ├── agent_graph.py        # Main LangGraph agent definition and execution
│   ├── agent_state.py        # Defines the AgentState class for state management
│   ├── nodes/              # Contains LangGraph nodes
│   │   ├── input_node.py     # Input node to receive user queries
│   │   ├── intent_router_node.py # Node for intent classification
│   │   ├── llm_call_node.py    # Node for calling the LLM for response generation
│   │   ├── memory_node.py      # Node for conversation memory management
│   │   ├── output_node.py      # Output node to present agent responses
│   │   ├── tenant_action_node.py # Node to handle tenant-specific actions (CRUD)
│   │   ├── tool_invocation_node.py # Node to invoke selected tools
│   │   └── tool_selection_node.py # Node for tool selection
│   └── prompts/             # Directory for prompt templates (if any)
├── tools/
│   ├── sql_tool.py         # SQL database interaction tool (LangChain Toolkit)
│   └── ...                 # Other tools (if added)
├── config/
│   └── config_loader.py    # Configuration loading utilities
├── data/
│   ├── db_schema_cache.json # Cached database schema (for faster loading)
│   └── mall_database.db    # Example SQLite database (or your database file)
├── utils/
│   └── database_utils.py   # Database utility functions (schema loading, etc.)
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python package dependencies
└── config.yaml             # Configuration file (API keys, database settings)
```

**Key Directories and Files:**

- **`agent/`**: Contains the core agent logic, including the LangGraph definition, state management, and node implementations.
  - **`agent_graph.py`**: The main script that sets up and runs the LangGraph agent.
  - **`nodes/`**: Directory containing individual LangGraph nodes, each responsible for a specific part of the workflow (intent routing, tenant action handling, LLM calls, memory, tool interaction, etc.).
  - **`agent_state.py`**: Defines the `AgentState` class, which manages the state of the conversation throughout the agent's workflow.
- **`tools/`**: Contains implementations of various tools that the agent can use, such as the `SQLDatabaseTool` for database interaction.
- **`config/`**: Holds configuration-related files, including `config_loader.py` for loading settings from `config.yaml`.
- **`data/`**: Directory for data files, such as the cached database schema (`db_schema_cache.json`) and example database files (e.g., `mall_database.db`).
- **`utils/`**: Utility functions, particularly `database_utils.py` for database schema loading and related tasks.
- **`README.md`**: This documentation file.
- **`requirements.txt`**: List of Python package dependencies.
- **`config.yaml`**: Configuration file to store API keys, database connection details, and other settings.

## Usage Instructions

After setting up and running the chatbot, you can interact with it through the terminal. Here are some example user queries for tenant data operations:

**Example Tenant Queries:**

- **Update Offer:**

  ```
  Update offer "Summer Sale" discount to 40%
  ```

  or (if offer title is ambiguous):

  ```
  Update offer with title "Summer Sale" discount to 40%
  ```

  If the chatbot needs more information (e.g., `offer_id` or clarification), it will prompt you conversationally.

- **Insert New Offer:**

  ```
  Create a new offer with title "Back to School Discount", description "Discounts on school supplies", discount percentage 25%, start date 2024-08-01, end date 2024-08-31, store id 2, product id 3
  ```

  or (if providing information in multiple turns):

  ```
  I want to add a new offer
  ```

  The chatbot will then guide you through providing the required fields step-by-step.

- **Update Store Opening Hours:**

  ```
  Update Zara's opening hours to 10 AM - 10 PM daily
  ```

  or (if store name is not unique or needs clarification):

  ```
  Update store "Zara" with store ID 1 opening hours to 10:00 - 22:00
  ```

  The chatbot might ask for confirmation or missing details like store ID or opening hours in Arabic.

- **Delete Offer:**
  ```
  Delete offer "Winter Sale 2023"
  ```
  or (if offer title is not unique):
  ```
  Delete offer with title "Winter Sale 2023" and offer ID 5
  ```

**General Interaction:**

- Type your queries in natural language at the `Enter your query:` prompt.
- The chatbot will respond with relevant information or ask for clarification if needed.
- To exit the chatbot, type `exit`.

## Technology Stack

- **LangChain:** Framework for building language model applications.
- **LangGraph:** LangChain's graph-based agent orchestration system.
- **GroqCloud:** For access to the Gemma 2-9B-IT language model.
- **Gemma 2-9B-IT (by Google):** Large language model used for intent recognition, response generation, and SQL query generation.
- **SQL Database:** For storing and managing tenant data (SQLite by default, configurable for PostgreSQL, MySQL).
- **Python:** Programming language for the chatbot implementation.
- **YAML:** For configuration management (`config.yaml`).

---

This README provides a comprehensive overview of the project for new developers. It explains the project's purpose, workflow, setup, usage, and structure, making it easier to understand and contribute to the Cenomi Malls Chatbot project. Remember to replace placeholder values (like `<repository_url>`, `YOUR_GROQ_API_KEY`) with actual project-specific information when using this README.

```

```
#   c e n o m i _ a p i  
 
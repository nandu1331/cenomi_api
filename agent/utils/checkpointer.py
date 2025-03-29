import json
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from agent.utils.database_utils import db_fetch_one, db_execute
from datetime import datetime

class DatabaseCheckpointer(BaseCheckpointSaver):
    def get_tuple(self, config):
        """Retrieve the checkpoint tuple for a given session_id (thread_id)."""
        # Extract thread_id from the config (maps to session_id in your system)
        thread_id = config["configurable"]["thread_id"]
        
        # Fetch the current state and metadata from the database
        row = db_fetch_one(
            "SELECT current_state, updated_at FROM conversations WHERE session_id = %s",
            (thread_id,)
        )
        
        # If a row exists and contains a state, construct the CheckpointTuple
        if row and row["current_state"]:
            state = json.loads(row["current_state"])  # Parse the JSON state
            metadata = {"timestamp": row["updated_at"].isoformat()}  # Add timestamp to metadata
            
            return CheckpointTuple(
                config=config,          # The input configuration
                checkpoint=state,       # The state dictionary
                metadata=metadata,      # Metadata (e.g., timestamp)
                parent_config=None      # No parent checkpoint for simplicity
            )
        # If no checkpoint exists, return None
        return None

    def put(self, config, checkpoint, metadata, new_versions):
        """Save or update the state in the conversations table."""
        # Extract thread_id from config
        thread_id = config["configurable"]["thread_id"]

        # Serialize the checkpoint (state) to JSON
        state_json = json.dumps(checkpoint)

        # Check if the session already exists in the database
        exists = db_fetch_one(
            "SELECT 1 FROM conversations WHERE session_id = %s",
            (thread_id,)
        )

        if exists:
            # Update the existing row
            db_execute(
                """
                UPDATE conversations
                SET current_state = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """,
                (state_json, thread_id)
            )
        else:
            # Insert a new row with default values for missing fields
            user_id = checkpoint.get("user_id", None)
            language = checkpoint.get("language", "en")  # Default to 'en' if not provided
            db_execute(
                """
                INSERT INTO conversations (session_id, user_id, language, current_state, created_at, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (thread_id, user_id, language, state_json)
            )

        # Return the config as required by LangGraph
        return config

    def list(self, config):
        """Optional: List checkpoints (not implemented for now)."""
        return iter([])  # Return an empty iterator as a placeholder
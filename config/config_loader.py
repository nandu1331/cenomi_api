import yaml
from pydantic import BaseModel

class ChromaDBConfig(BaseModel):
    persist_directory: str

class DatabaseConfig(BaseModel):
    db_uri: str
    
class LlmConfig(BaseModel):
    model_name: str
    api_key: str

class Config(BaseModel):
    chromadb: ChromaDBConfig
    database: DatabaseConfig
    llm: LlmConfig
    
def load_config(config_path="D:/cenomi_malls_chatbot/config/config.yaml"):
    """Loads the configuration from the YAML file."""
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
        return Config(**config_dict)
    
if __name__ == "__main__":
    config = load_config()
    print("Chroma Db dir: ", config.chromadb.persist_directory)
    print("Postgres URI: ", config.database.db_uri)
    print("LLM: ", config.llm.model_name)
    print("API KEY: ", config.llm.api_key)
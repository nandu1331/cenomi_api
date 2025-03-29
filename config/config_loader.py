import yaml
from pydantic import BaseModel

class ChromaDBConfig(BaseModel):
    persist_directory: str

class DatabaseConfig(BaseModel):
    db_uri: str
    
class LlmConfig(BaseModel):
    model_name: str
    api_key: str
    
class PineconeConfig(BaseModel):
    api_key: str
    index_name: str

class Config(BaseModel):
    chromadb: ChromaDBConfig
    database: DatabaseConfig
    llm: LlmConfig
    pineconedb: PineconeConfig
    
def load_config(config_path=r"D:/cenomi_api/config/config.yaml"):
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
    print("PINCONE CONFIG:\n", config.pineconedb.api_key, " ", config.pineconedb.index_name)
import yaml
from pydantic import BaseModel

class ChromaDBConfig(BaseModel):
    persist_directory: str

class DatabaseConfig(BaseModel):
    postgres_uri: str

class Config(BaseModel):
    chromadb: ChromaDBConfig
    database: DatabaseConfig
    
def load_config(config_path="./config.yaml"):
    """Loads the configuration from the YAML file."""
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
        return Config(**config_dict)
    
if __name__ == "__main__":
    config = load_config()
    print("Chroma Db dir: ", config.chromadb.persist_directory)
    print("Postgres URI: ", config.database.postgres_uri)
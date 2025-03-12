from typing import List, Dict

class Model_Registry:
  _models = {
    "Deepseek" : "deepseek-r1:latest",
    "Mistral" : "mistral:latest",
    "LLaMA" : "llama3.1:8b",
    "Qwen" : "qwen2.5:7b",
    "Gemma" : "gemma:7b",
  }

  @classmethod
  def get_model(cls, model_name: str) -> str: 
    return cls._models.get(model_name, "mistral:latest")
  
  @classmethod
  def list_models(cls) -> List[str]:
    return list(cls._models.keys())
  
  @classmethod
  def get_available_models(cls) -> Dict[str, str]:
    return cls._models
  
  @classmethod
  def get_default_model(cls) -> str:
    return "mistral:latest"
  
model_registry = Model_Registry()
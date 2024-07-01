
from enum import StrEnum
from pydantic_settings import BaseSettings

class PoGPipelineSchema(StrEnum):
    """Schema for PoG pipeline."""
    DATA = "data"
    FEATURE_ENGINEERING = "feature_engineering"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    

class PoGSettings(BaseSettings):...

# data
class SourceData:...
class FeatureTable:...
class Dataset:...

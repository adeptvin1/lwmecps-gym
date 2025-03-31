import pytest
from lwmecps_gym_api.core.config import Settings
from lwmecps_gym_api.core.wandb_config import WandbConfig

def test_settings_defaults():
    """Test default settings values"""
    settings = Settings()
    assert settings.project_name == "lwmecps-gym-api"
    assert settings.version == "0.1.0"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.mongodb_url == "mongodb://localhost:27017"
    assert settings.mongodb_db_name == "lwmecps_gym"
    assert settings.mongodb_collection_prefix == "lwmecps_gym"
    assert settings.wandb_project == "lwmecps-gym"
    assert settings.wandb_entity == "lwmecps"
    assert settings.wandb_api_key is None
    assert settings.wandb_mode == "online"
    assert settings.wandb_log_model == "all"
    assert settings.wandb_save_code == "now"
    assert settings.wandb_job_type == "training"
    assert settings.wandb_tags == ["lwmecps", "gym", "api"]
    assert settings.wandb_notes is None
    assert settings.wandb_config == {}

def test_settings_custom_values():
    """Test custom settings values"""
    custom_settings = {
        "project_name": "test-project",
        "version": "1.0.0",
        "mongodb_url": "mongodb://test:27017",
        "mongodb_db_name": "test_db",
        "wandb_project": "test-project",
        "wandb_entity": "test-entity",
        "wandb_api_key": "test-key",
        "wandb_mode": "offline",
        "wandb_job_type": "test-job",
        "wandb_tags": ["test"],
        "wandb_notes": "Test notes"
    }
    
    settings = Settings(**custom_settings)
    assert settings.project_name == custom_settings["project_name"]
    assert settings.version == custom_settings["version"]
    assert settings.mongodb_url == custom_settings["mongodb_url"]
    assert settings.mongodb_db_name == custom_settings["mongodb_db_name"]
    assert settings.wandb_project == custom_settings["wandb_project"]
    assert settings.wandb_entity == custom_settings["wandb_entity"]
    assert settings.wandb_api_key == custom_settings["wandb_api_key"]
    assert settings.wandb_mode == custom_settings["wandb_mode"]
    assert settings.wandb_job_type == custom_settings["wandb_job_type"]
    assert settings.wandb_tags == custom_settings["wandb_tags"]
    assert settings.wandb_notes == custom_settings["wandb_notes"]

def test_wandb_config_defaults():
    """Test default WandbConfig values"""
    config = WandbConfig()
    assert config.project == "lwmecps-gym"
    assert config.entity == "lwmecps"
    assert config.api_key is None
    assert config.mode == "online"
    assert config.log_model == "all"
    assert config.save_code == "now"
    assert config.job_type == "training"
    assert config.tags == ["lwmecps", "gym", "api"]
    assert config.notes is None
    assert config.config == {}

def test_wandb_config_custom_values():
    """Test custom WandbConfig values"""
    custom_config = {
        "project": "test-project",
        "entity": "test-entity",
        "api_key": "test-key",
        "mode": "offline",
        "job_type": "test-job",
        "tags": ["test"],
        "notes": "Test notes"
    }
    
    config = WandbConfig(**custom_config)
    assert config.project == custom_config["project"]
    assert config.entity == custom_config["entity"]
    assert config.api_key == custom_config["api_key"]
    assert config.mode == custom_config["mode"]
    assert config.job_type == custom_config["job_type"]
    assert config.tags == custom_config["tags"]
    assert config.notes == custom_config["notes"]

def test_wandb_config_from_settings():
    """Test creating WandbConfig from Settings"""
    settings = Settings(
        wandb_project="test-project",
        wandb_entity="test-entity",
        wandb_api_key="test-key",
        wandb_mode="offline",
        wandb_job_type="test-job",
        wandb_tags=["test"],
        wandb_notes="Test notes"
    )
    
    config = WandbConfig.from_settings(settings)
    assert config.project == settings.wandb_project
    assert config.entity == settings.wandb_entity
    assert config.api_key == settings.wandb_api_key
    assert config.mode == settings.wandb_mode
    assert config.job_type == settings.wandb_job_type
    assert config.tags == settings.wandb_tags
    assert config.notes == settings.wandb_notes 
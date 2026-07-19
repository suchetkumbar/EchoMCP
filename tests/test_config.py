from agents.config import Settings, get_settings


def test_settings_defaults_are_present():
    settings = Settings()
    assert settings.llm_provider == "fake"
    assert settings.embedding_provider == "fake"
    assert settings.vector_store == "memory"


def test_get_settings_returns_settings_instance():
    settings = get_settings()
    assert isinstance(settings, Settings)

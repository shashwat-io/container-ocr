import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    USE_STUB_MODEL = os.getenv("USE_STUB_MODEL", "True").lower() in ("true", "1", "yes")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///test.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///dev_app.db")


class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SECRET_KEY = os.getenv("SECRET_KEY")


config_by_name = {"dev": DevelopmentConfig, "prod": ProductionConfig}


def get_config(env="dev"):
    # Fail loudly on an unrecognised FLASK_CONFIG. This used to be
    # `config_by_name.get(env, Config)`, so a typo ("production", "PROD",
    # "PRODUCTION") silently fell back to the base Config -- which carries
    # SECRET_KEY="dev-secret-key" and sqlite:///test.db. A production
    # deployment could therefore come up with a publicly-known signing key and
    # an empty throwaway database, and look perfectly healthy while doing it.
    if env not in config_by_name:
        raise ValueError(
            f"Unknown FLASK_CONFIG {env!r}. Valid values: "
            f"{sorted(config_by_name)}. Refusing to fall back to development "
            "defaults (dev secret key, throwaway sqlite DB) silently."
        )
    return config_by_name[env]
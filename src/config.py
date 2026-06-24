"""全局配置模块 — 加载config.yaml并提供配置访问接口"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict

# 加载 .env 文件（如果存在）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # 无 dotenv 包时手动解析 .env
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _key, _val = _key.strip(), _val.strip()
                    if _key and _key not in os.environ:
                        os.environ[_key] = _val


class Config:
    """配置单例"""

    _instance = None
    _data: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, config_path: str = None):
        """加载YAML配置文件"""
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cls._data = yaml.safe_load(f)
        # 解析环境变量
        cls._resolve_env_vars()
        return cls._data

    @classmethod
    def get(cls, key_path: str, default=None):
        """用点号分隔的路径获取配置，如 'models.svd.n_factors'"""
        keys = key_path.split(".")
        val = cls._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    @classmethod
    def _resolve_env_vars(cls):
        """解析配置中的 ${ENV_VAR} 引用"""
        # DeepSeek API key — 支持 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            cls._data.setdefault("llm", {})
            cls._data["llm"]["api_key"] = api_key
        tmdb_key = os.environ.get("TMDB_API_KEY", "")
        if tmdb_key:
            cls._data.setdefault("data", {}).setdefault("tmdb", {})
            cls._data["data"]["tmdb"]["api_key"] = tmdb_key

    _project_root: Path = None

    @classmethod
    def get_project_root(cls) -> Path:
        """返回项目根目录"""
        if cls._project_root is None:
            cls._project_root = Path(__file__).resolve().parent.parent
        return cls._project_root


# 自动加载
Config.load()

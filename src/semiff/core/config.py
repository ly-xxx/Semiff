"""
配置管理系统 - 工业级配置中心化
使用 OmegaConf 支持动态配置和环境变量插值
"""
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ConfigManager:
    """配置管理器 - 负责加载、验证和导出配置"""

    @staticmethod
    def load(config_path: str = "configs/default.yaml") -> DictConfig:
        """
        加载配置文件并进行后处理

        Args:
            config_path: 配置文件路径

        Returns:
            OmegaConf 配置对象
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置
        conf = OmegaConf.load(config_path)

        # 时间戳插值
        now = datetime.now()
        time_vars = {
            "now": now,
            "timestamp": now.strftime("%Y%m%d_%H%M%S")
        }

        # 应用时间插值
        conf = OmegaConf.to_container(conf, resolve=True)
        conf_str = str(conf)

        for key, value in time_vars.items():
            if isinstance(value, datetime):
                # 处理日期时间格式
                conf_str = conf_str.replace(f"${{{key}:%Y-%m-%d}}", value.strftime("%Y-%m-%d"))
                conf_str = conf_str.replace(f"${{{key}:%H-%M-%S}}", value.strftime("%H-%M-%S"))
            else:
                conf_str = conf_str.replace(f"${{{key}}}", str(value))

        conf = OmegaConf.create(conf_str)

        return conf

    @staticmethod
    def validate_config(conf: DictConfig) -> bool:
        """
        验证配置文件的正确性，并初始化工作区

        Args:
            conf: 配置对象

        Returns:
            验证是否通过
        """
        required_fields = [
            "pipeline.name",
            "pipeline.workspace",
            "robot.urdf_path",
            "optimization.lr_pose",
            "optimization.iterations",
            "geometry.binding_method"
        ]

        for field in required_fields:
            if not OmegaConf.select(conf, field):
                print(f"❌ 配置验证失败: 缺少必需字段 {field}")
                return False

        # 验证数值范围
        if conf.optimization.lr_pose <= 0:
            print("❌ 配置验证失败: lr_pose 必须大于 0")
            return False

        if conf.optimization.iterations <= 0:
            print("❌ 配置验证失败: iterations 必须大于 0")
            return False

        # 验证通过后，创建工作区和配置备份
        try:
            workspace_path = Path(conf.pipeline.workspace)
            workspace_path.mkdir(parents=True, exist_ok=True)

            # 导出本次运行配置备份
            config_snapshot = workspace_path / "config_snapshot.yaml"
            OmegaConf.save(conf, config_snapshot)
        except Exception as e:
            print(f"❌ 配置验证失败: 无法创建工作区: {e}")
            return False

        print("✅ 配置验证通过")
        return True

    @staticmethod
    def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
        """
        合并两个配置对象，后面的覆盖前面的

        Args:
            base_config: 基础配置
            override_config: 覆盖配置

        Returns:
            合并后的配置
        """
        return OmegaConf.merge(base_config, override_config)

    @staticmethod
    def save_config(conf: DictConfig, path: str):
        """保存配置到文件"""
        OmegaConf.save(conf, path)

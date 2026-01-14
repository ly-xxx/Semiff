"""
src/semiff/core/workspace.py
智能工作区管理器 (Upgrade v2)
负责自动发现和管理基于时间戳的递归工作区
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

logger = logging.getLogger("WORKSPACE")

class WorkspaceManager:
    """智能工作区管理器"""

    def __init__(self, config_path="configs/base_config.yaml", project_root=None):
        self.raw_conf = OmegaConf.load(config_path)

        # 1. 确定项目根目录
        if project_root is None:
            config_path_obj = Path(config_path)
            if config_path_obj.is_absolute():
                project_root = config_path_obj.parent.parent
            else:
                project_root = Path.cwd()
                while project_root != project_root.parent:
                    if (project_root / config_path_obj).exists():
                        break
                    project_root = project_root.parent

        self.project_root = Path(project_root)
        logger.info(f"📂 Project Root: {self.project_root}")

        # 获取 outputs/ 根目录
        workspace_rel = self.raw_conf.pipeline.workspace
        self.base_output_dir = self.project_root / Path(workspace_rel).parent

    @staticmethod
    def find_project_root(start_path=None):
        """
        🔍 静态方法：从任意位置向上查找项目根目录
        通过标志性文件/文件夹识别根目录（pyproject.toml, .git, configs/）
        
        Args:
            start_path: 起始路径（默认为当前文件所在目录）
        
        Returns:
            Path: 项目根目录
        """
        if start_path is None:
            start_path = Path(__file__).resolve().parent
        else:
            start_path = Path(start_path).resolve()
        
        current = start_path
        # 向上查找，直到找到标志性文件
        markers = ["pyproject.toml", ".git", "configs"]
        
        while current != current.parent:
            # 检查是否包含任意一个标志性文件/文件夹
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        
        # 如果找不到，返回当前工作目录（fallback）
        logger.warning(f"⚠️ Could not find project root from {start_path}, using cwd")
        return Path.cwd()
    
    @staticmethod
    def resolve_path(path_str, base_dir=None):
        """
        🔧 静态方法：智能路径解析工具
        
        Args:
            path_str: 路径字符串（可以是绝对路径或相对路径）
            base_dir: 基准目录（默认为项目根目录）
        
        Returns:
            Path: 解析后的绝对路径
        """
        if base_dir is None:
            base_dir = WorkspaceManager.find_project_root()
        else:
            base_dir = Path(base_dir)
        
        path = Path(path_str)
        
        # 如果已经是绝对路径，直接返回
        if path.is_absolute():
            return path
        
        # 否则，相对于 base_dir 解析
        return (base_dir / path).resolve()

    def _find_candidate_workspaces(self, search_root, required_files):
        """递归寻找包含特定文件的目录"""
        candidates = []
        if not search_root.exists():
            return candidates

        # 广度优先搜索，限制深度防止过慢
        # Step 1 结果在 depth=1, Step 2 结果在 depth=2, etc.
        for root, dirs, files in os.walk(search_root):
            # 优化：跳过显然不是 workspace 的目录
            if "checkpoints" in root or "__pycache__" in root:
                continue

            path_obj = Path(root)
            # 检查当前目录是否包含所有必要文件
            missing = [f for f in required_files if not (path_obj / f).exists()]
            if not missing:
                candidates.append(path_obj)

            # 限制搜索深度 (例如只看 outputs/ 下的 3 层)
            rel_depth = len(path_obj.relative_to(search_root).parts)
            if rel_depth >= 3:
                del dirs[:] # 停止向下递归

        # 按修改时间倒序排列
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates

    def resolve(self, mode="auto", required_input_files=None):
        """
        [兼容 Step 1] 解析根级工作区
        """
        # ... (保持原有逻辑不变，为节省篇幅略去，与你提供的原代码一致)
        # 这里为了完整性，你可以直接复用你发给我的代码中的 resolve 方法
        # 核心逻辑: mode='new' -> 创建 outputs/TIMESTAMP

        if mode == "new":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Step 1 习惯带上后缀以便识别
            ws = self.base_output_dir / f"{timestamp}_step1"
            ws.mkdir(parents=True, exist_ok=True)
            logger.info(f"🆕 Created new ROOT workspace: {ws}")
            return ws

        # 复用原有 auto 逻辑
        latest = self.get_latest_workspace(required_input_files)
        if latest:
            logger.info(f"🔄 Auto-selected latest workspace: {latest}")
            return latest

        # Fallback to new
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ws = self.base_output_dir / f"{timestamp}_step1"
        ws.mkdir(parents=True, exist_ok=True)
        logger.info(f"🆕 No history found. Created new: {ws}")
        return ws

    def get_latest_workspace(self, required_files=None):
        """(辅助 Step 1) 在根目录下找"""
        if not self.base_output_dir.exists(): return None
        subdirs = sorted([d for d in self.base_output_dir.iterdir() if d.is_dir()],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        for d in subdirs:
            if not required_files: return d
            if all((d/f).exists() for f in required_files): return d
        return None

    def resolve_child(self, parent_requirements, step_name, mode="auto", manual_parent_path=None):
        """
        [Step 2/3/4 专用] 解析子级递归工作区

        Args:
            parent_requirements (list): 父级目录必须包含的文件 (e.g. ['camera_poses.npy'])
            step_name (str): 当前步骤名称 (e.g. 'step2_calibrate')
            mode (str): 'auto' 或 'manual'
            manual_parent_path (str): 手动指定父级路径

        Returns:
            (current_ws_path, parent_ws_path)
        """
        parent_ws = None

        # 1. 确定父级工作区
        if mode == "manual":
            if not manual_parent_path:
                raise ValueError("❌ Mode is manual but `manual_parent_path` is empty!")
            p_path = Path(manual_parent_path)
            if not p_path.is_absolute():
                p_path = self.project_root / p_path

            if not p_path.exists():
                raise FileNotFoundError(f"❌ Manual parent path not found: {p_path}")

            # 验证文件
            missing = [f for f in parent_requirements if not (p_path / f).exists()]
            if missing:
                raise FileNotFoundError(f"❌ Parent {p_path} missing files: {missing}")

            parent_ws = p_path
            logger.info(f"👉 Using Manual Parent Workspace: {parent_ws}")

        else: # auto
            logger.info(f"🔍 Auto-searching for latest workspace with: {parent_requirements}...")
            candidates = self._find_candidate_workspaces(self.base_output_dir, parent_requirements)

            if not candidates:
                raise FileNotFoundError(f"❌ No valid parent workspace found containing {parent_requirements}")

            parent_ws = candidates[0]
            logger.info(f"🔄 Auto-selected Latest Parent: {parent_ws}")

        # 2. 创建当前步骤的子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        child_ws_name = f"{timestamp}_{step_name}"
        child_ws = parent_ws / child_ws_name

        child_ws.mkdir(parents=True, exist_ok=True)
        logger.info(f"🆕 Created Child Workspace: {child_ws}")
        logger.info(f"   (Data can be accessed via ../filename)")

        return child_ws, parent_ws

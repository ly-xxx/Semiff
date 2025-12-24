import os
import datetime
from pathlib import Path

class PipelineManager:
    def __init__(self, dataset_name, base_output_dir="outputs"):
        self.dataset_name = dataset_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(base_output_dir)

    def get_stage_dir(self, stage_id, stage_name):
        """
        生成规范的输出路径: outputs/01_example01_20251224_120000_preprocess
        """
        dir_name = f"{stage_id:02d}_{self.dataset_name}_{self.timestamp}_{stage_name}"
        full_path = self.base_output_dir / dir_name
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

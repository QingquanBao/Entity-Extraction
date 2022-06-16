import json
from os.path import join
from typing import Optional

from dataclasses import dataclass, field, asdict


@dataclass
class _Args:
    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConstructArgs(_Args):
    model_type: str = field(metadata={"help": "Pretrained model path"})
    head_type: str = field(metadata={"choices": ["linear", "linear_nested", "crf", "crf_nested", 'linearadv','crf_nested_adv'], "help": "Type of head"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path"})
    init_model: Optional[int] = field(default=0, metadata={"choices": [0, 1], "help": "Init models' parameters"})
    
    lr_decay: Optional[bool] = field(default=False, metadata={"help": "Whether to decay learning rate by layers"})
    use_swa: Optional[bool] = field(default=False, metadata={"help": "Whether to use SWA"})
    swa_start: Optional[int] = field(default=6, metadata={"help": "Start epoch of SWA"})
    swa_lr: Optional[float] = field(default=2e-6, metadata={"help": "SWA learning rate"})
    # warm_up: Optional[bool] = field(default=False, metadata={"help": "Whether to use lr warmup"}) # default in lr_decay
    use_pgd: Optional[bool] = field(default=False, metadata={"help": "Whether to use PGD"})
    adv_weight: Optional[float] = field(default=1., metadata={"help": ""})
    adv_eps: Optional[float] = field(default=1e-6, metadata={"help": ""})
    adv_stepsize: Optional[float] = field(default=1e-3, metadata={"help": ""})
    adv_stepnum: Optional[int] = field(default=1, metadata={"help": ""})
    adv_noisevar: Optional[float] = field(default=1e-5, metadata={"help": ""})
    
@dataclass
class CBLUEDataArgs(_Args):
    cblue_root: str = field(metadata={"help": "CBLUE data root"})
    max_length: Optional[int] = field(default=128, metadata={"help": "Max sequence length"})
    fusion: Optional[bool] = field(default=False, metadata={"help": "Whether to use data fusion"})
    fusion_type: Optional[float] = field(default=0, metadata={"help": "Data fusion type"})

    fusion: Optional[bool] = field(default=False, metadata={"help": "Whether to fuse input"})

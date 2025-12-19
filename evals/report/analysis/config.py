"""
Centralized configuration for the analysis pipeline.
"""
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for LoRA hyperparameter analysis."""
    
    # Project paths
    base_dir: Path = Path('/users/bbullinger/peft_apertus/report')
    data_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    history_dir: Path = field(init=False)
    
    # WandB configuration
    wandb_entity: str = "lsaie-peft-apertus"
    wandb_project: str = "swiss_judgment_prediction"
    
    # Experimental parameters
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    lora_ranks: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    model_sizes: List[str] = field(default_factory=lambda: ['8B', '70B'])
    methods: List[str] = field(default_factory=lambda: ['lora', 'full'])
    
    # Plotting configuration
    plot_style: str = 'seaborn-v0_8-whitegrid'
    figure_dpi: int = 300
    default_figsize: tuple = (10, 7)
    
    # Color schemes
    rank_colormap: str = 'viridis'
    full_ft_color: str = 'orange'
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / 'data'
        self.plots_dir = self.base_dir / 'plots'
        self.history_dir = self.data_dir / 'history'
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
    
    @property
    def wandb_project_path(self) -> str:
        """Full WandB project path."""
        return f"{self.wandb_entity}/{self.wandb_project}"
    
    def get_color_for_rank(self, rank: int) -> tuple:
        """Get consistent color for a given rank."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if rank not in self.lora_ranks:
            return (0.5, 0.5, 0.5)  # Gray for unknown ranks
        
        idx = self.lora_ranks.index(rank)
        cmap = plt.cm.get_cmap(self.rank_colormap)
        return cmap(np.linspace(0.1, 0.9, len(self.lora_ranks))[idx])


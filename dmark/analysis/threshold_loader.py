import json
from typing import Dict, Optional, List


class ThresholdLoader:
    """Load and query z-score thresholds from configuration JSON."""
    
    def __init__(self, config_path: str):
        """
        Initialize threshold loader with configuration file.
        
        Args:
            config_path: Path to threshold configuration JSON
        """
        with open(config_path, 'r') as f:
            self.data = json.load(f)
        
        self.version = self.data.get('version')
        self.metadata = self.data.get('metadata', {})
        self.configurations = self.data.get('configurations', [])
    
    def get_threshold(
        self,
        fpr: float,
        model: str = None,
        dataset: str = None,
        steps: int = None,
        gen_length: int = None,
        block_length: int = None,
        temperature: float = None,
        cfg_scale: float = None,
        remasking: str = None
    ) -> Optional[float]:
        """
        Get threshold for a specific FPR and configuration.
        
        Args:
            fpr: Target false positive rate (e.g., 0.01 for 1%)
            model: Model name
            dataset: Dataset name
            steps: Number of generation steps
            gen_length: Generation length
            block_length: Block length
            temperature: Temperature value
            cfg_scale: CFG scale value
            remasking: Remasking strategy
        
        Returns:
            Threshold value or None if not found
        """
        # Convert FPR to string key
        fpr_key = str(fpr)
        
        # Find matching configuration
        for config_entry in self.configurations:
            config = config_entry['config']
            
            # Check all specified parameters
            if model and config.get('model') != model:
                continue
            if dataset and config.get('dataset') != dataset:
                continue
            if steps is not None and config.get('steps') != steps:
                continue
            if gen_length is not None and config.get('gen_length') != gen_length:
                continue
            if block_length is not None and config.get('block_length') != block_length:
                continue
            if temperature is not None and abs(config.get('temperature', -1) - temperature) > 0.001:
                continue
            if cfg_scale is not None and abs(config.get('cfg_scale', -1) - cfg_scale) > 0.001:
                continue
            if remasking and config.get('remasking') != remasking:
                continue
            
            # Found matching configuration
            return config_entry['thresholds'].get(fpr_key)
        
        return None
    
    def find_configurations(
        self,
        model: str = None,
        dataset: str = None,
        steps: int = None,
        gen_length: int = None,
        block_length: int = None,
        temperature: float = None,
        cfg_scale: float = None,
        remasking: str = None
    ) -> List[Dict]:
        """
        Find all configurations matching the specified criteria.
        
        Returns:
            List of matching configuration entries
        """
        matches = []
        
        for config_entry in self.configurations:
            config = config_entry['config']
            
            # Check all specified parameters
            if model and config.get('model') != model:
                continue
            if dataset and config.get('dataset') != dataset:
                continue
            if steps is not None and config.get('steps') != steps:
                continue
            if gen_length is not None and config.get('gen_length') != gen_length:
                continue
            if block_length is not None and config.get('block_length') != block_length:
                continue
            if temperature is not None and abs(config.get('temperature', -1) - temperature) > 0.001:
                continue
            if cfg_scale is not None and abs(config.get('cfg_scale', -1) - cfg_scale) > 0.001:
                continue
            if remasking and config.get('remasking') != remasking:
                continue
            
            matches.append(config_entry)
        
        return matches
    
    def get_available_fprs(self) -> List[float]:
        """Get list of available FPR values in the configuration."""
        if not self.configurations:
            return []
        
        # Get FPR keys from first configuration
        first_config = self.configurations[0]
        fpr_keys = list(first_config.get('thresholds', {}).keys())
        return [float(key) for key in fpr_keys]
    
    def get_statistics(
        self,
        model: str = None,
        dataset: str = None,
        steps: int = None,
        gen_length: int = None,
        block_length: int = None,
        temperature: float = None,
        cfg_scale: float = None,
        remasking: str = None
    ) -> Optional[Dict]:
        """
        Get statistics for a specific configuration.
        
        Returns:
            Statistics dict or None if not found
        """
        configs = self.find_configurations(
            model=model,
            dataset=dataset,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking
        )
        
        if configs:
            return configs[0].get('statistics')
        return None


def load_thresholds(config_path: str) -> ThresholdLoader:
    """
    Convenience function to load threshold configuration.
    
    Args:
        config_path: Path to threshold configuration JSON
    
    Returns:
        ThresholdLoader instance
    """
    return ThresholdLoader(config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query threshold configuration")
    parser.add_argument("config", help="Path to threshold configuration JSON")
    parser.add_argument("--fpr", type=float, default=0.01, help="Target FPR (default: 0.01)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--steps", type=int, help="Number of steps")
    parser.add_argument("--gen_length", type=int, help="Generation length")
    parser.add_argument("--block_length", type=int, help="Block length")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--cfg_scale", type=float, help="CFG scale")
    parser.add_argument("--remasking", help="Remasking strategy")
    
    args = parser.parse_args()
    
    # Load configuration
    loader = ThresholdLoader(args.config)
    
    # Query threshold
    threshold = loader.get_threshold(
        fpr=args.fpr,
        model=args.model,
        dataset=args.dataset,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking
    )
    
    if threshold is not None:
        print(f"Threshold for FPR {args.fpr}: {threshold}")
        
        # Also show statistics
        stats = loader.get_statistics(
            model=args.model,
            dataset=args.dataset,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
        
        if stats:
            print(f"Statistics: mean={stats['mean']}, std={stats['std']}, n={stats['n_samples']}")
    else:
        print(f"No threshold found for the specified configuration")
        
        # Show available configurations
        configs = loader.find_configurations(
            model=args.model,
            dataset=args.dataset,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
        
        if configs:
            print(f"\nFound {len(configs)} partial matches:")
            for config in configs[:3]:
                print(f"  - {config['config']}")
        else:
            print("\nNo matching configurations found")
        
        print(f"\nAvailable FPRs: {loader.get_available_fprs()}")
import argparse
import yaml
import json
import os
import sys
from pathlib import Path

from .preprocess import load_and_preprocess_data, compute_node_features
from .train import train_bef_gat
from .evaluate import evaluate_model, compute_bef_gat_statistics, generate_plots, save_results_json, print_experiment_summary


def run_experiment(config):
    """Run a single BEF-GAT experiment with given configuration."""
    print(f"\nStarting BEF-GAT experiment with config: {config.get('dataset', 'Multiple datasets')}")
    
    datasets = config.get('datasets', [config.get('dataset', 'Cora')])
    if isinstance(datasets, str):
        datasets = [datasets]
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        
        data, num_classes = load_and_preprocess_data(dataset_name)
        print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges, {num_classes} classes")
        
        edge_features = compute_node_features(data)
        
        model, train_results = train_bef_gat(data, num_classes, config)
        
        eval_metrics = evaluate_model(model, data)
        bef_gat_stats = compute_bef_gat_statistics(model, data)
        
        results = {
            **train_results,
            **eval_metrics,
            **bef_gat_stats,
            'dataset': dataset_name,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'num_classes': num_classes
        }
        
        all_results[dataset_name] = results
    
    return all_results


def main():
    """Main execution function with command-line argument support."""
    parser = argparse.ArgumentParser(description="BEF-GAT Research Framework")
    parser.add_argument("--smoke-test", action="store_true", 
                       help="Run smoke test with small-scale configuration")
    parser.add_argument("--full-experiment", action="store_true", 
                       help="Run full experiment with complete configuration")
    
    args = parser.parse_args()
    
    if not (args.smoke_test or args.full_experiment):
        print("Error: Must specify either --smoke-test or --full-experiment")
        sys.exit(1)
    
    if args.smoke_test and args.full_experiment:
        print("Error: Cannot specify both --smoke-test and --full-experiment")
        sys.exit(1)
    
    if args.smoke_test:
        config_path = "config/smoke_test.yaml"
        experiment_type = "smoke_test"
    else:
        config_path = "config/full_experiment.yaml"
        experiment_type = "full_experiment"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in {config_path}: {e}")
        sys.exit(1)
    
    output_dir = config['output_dir']
    images_dir = f"{output_dir}/images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"BEF-GAT Research Framework - {experiment_type.replace('_', ' ').title()}")
    print(f"Output directory: {output_dir}")
    print(f"Images directory: {images_dir}")
    
    try:
        all_results = run_experiment(config)
        
        for dataset_name, results in all_results.items():
            plot_paths = generate_plots(results, config, images_dir)
            
            json_filename = f"results_{experiment_type}_{dataset_name.lower()}.json"
            json_path = os.path.join(output_dir, json_filename)
            output_data = save_results_json(results, config, json_path)
            
            print_experiment_summary(results, config, plot_paths)
            
            print(f"\nJSON RESULTS SAVED TO: {json_path}")
            print("JSON CONTENTS:")
            print(json.dumps(output_data, indent=2))
            print("\n" + "-"*80)
        
        print(f"\nAll experiments completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Visualizations saved to: {images_dir}")
        
    except Exception as e:
        print(f"Error during experiment execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

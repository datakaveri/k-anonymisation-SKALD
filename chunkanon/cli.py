import argparse
from chunkanon.core import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ChunKanon k-anonymization.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    run_pipeline(config_path=args.config)

if __name__ == "__main__":
    main()

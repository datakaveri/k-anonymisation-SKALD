import argparse
from SKALD.core import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run SKALD.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--k', type=int, default=None, help='Override k from config')
    parser.add_argument('--chunks', type=int, default=None, help='Override number of chunks from config')
    parser.add_argument('--chunk_dir', type=str, default='small_data', help='Override chunk directory from config')
    
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        k=args.k,
        chunks=args.chunks,
        chunk_dir=args.chunk_dir
    )

if __name__ == "__main__":
    main()

import glob
import os
import pandas as pd
import logging
logger = logging.getLogger("SKALD")


def combine_generalized_chunks(output_directory, output_filename):
        """
        Combine all generalized chunk CSVs into a single CSV
        and delete individual chunk files after successful merge.
        """

        output_directory = os.path.abspath(output_directory)
        final_output_path = os.path.join(output_directory, output_filename)

        # Find all chunk CSVs (exclude final output if it already exists)
        csv_files = sorted(
            f for f in glob.glob(os.path.join(output_directory, "*.csv"))
            if os.path.abspath(f) != final_output_path
        )

        if not csv_files:
            raise ValueError(f"No generalized chunk CSVs found in {output_directory}")

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                raise RuntimeError(f"Failed to read chunk file '{f}': {e}")

        combined = pd.concat(dfs, ignore_index=True)

        # Write combined output
        combined.to_csv(final_output_path, index=False)


        # Delete chunk files ONLY after successful write
        for f in csv_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f" Warning: failed to delete chunk file '{f}': {e}")

      
        logger.info("Combined %d chunks into '%s'", len(csv_files), final_output_path)

        return combined
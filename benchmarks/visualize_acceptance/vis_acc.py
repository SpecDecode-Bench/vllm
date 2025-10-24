import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from utils import MODEL_TO_NAMES, load_data, parse_args
import os
from pathlib import Path
import numpy as np

class AcceptanceStatsClient:
    """Client for fetching and processing acceptance statistics data."""

    def __init__(self, model_name, method, dataset, data_path=None):
        """Initialize the client with model and dataset info."""
        self.model_name = model_name
        self.method = method
        self.dataset = dataset

        if data_path is None:
            self.data_path = f"/data/lily/batch-sd/data/{model_name}/{method}_{dataset}_acceptance_stats.jsonl"
        else:
            self.data_path = data_path

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_NAMES[model_name], use_fast=False)
        self.acceptance_stats = None

    def load_data(self):
        """Load the acceptance statistics from file."""
        self.acceptance_stats = load_data(self.data_path, self.tokenizer)
        return self.acceptance_stats

    def _convert_to_array(self, fill_value=-2):
        """Convert list of AccStats to 2D numpy array, padding with fill_value for varying lengths.

        Args:
            fill_value: Value to use for padding shorter sequences (default: -1)

        Returns:
            2D numpy array with shape (num_requests, max_sequence_length)
        """
        # For Qwen models, truncate to first 1024 tokens for visualization, instead of full 32k
        if "qwen" in self.model_name:
            acc_lists = [stat.lens[:1024] for stat in self.acceptance_stats]
        else:
            acc_lists = [stat.lens for stat in self.acceptance_stats]

        max_len = max(len(l) for l in acc_lists)
        arr = np.full((len(acc_lists), max_len), fill_value, dtype=float)
        for i, row in enumerate(acc_lists):
            if len(row):
                arr[i, : len(row)] = row
        return arr

    def plot_heatmap(self, output_dir="figures"):
        """Plot the acceptance statistics as a heatmap."""
        if self.acceptance_stats is None:
            self.load_data()
        # Convert list of AccStats objects to a 2D array, padding with -1 for varying lengths
        arr = self._convert_to_array(fill_value=-1)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(arr, cmap="YlGnBu")
        plt.xlabel("Position")
        plt.ylabel("Request ID")

        # Add Y-axis labels on the right
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        ax2.set_ylabel("# of Accepted Tokens", labelpad=10)

        plt.title(f"Acceptance Statistics: {self.model_name} - {self.method} - {self.dataset}")
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_path = Path(output_dir) / self.model_name
        os.makedirs(output_path, exist_ok=True)

        output_file = output_path / f"{self.dataset}_{self.method}_{self.model_name}_acceptance_stats.pdf"
        plt.savefig(output_file)
        print(f"Saved heatmap to {output_file}")
        return fig

    def get_summary_stats(self):
        """Get summary statistics about the acceptance data."""
        if self.acceptance_stats is None:
            self.load_data()

        # Calculate average acceptance rate for each position and each request
        arr = self._convert_to_array(fill_value=np.nan)
        # Compute averages ignoring padded NaNs
        avg_by_position = np.nanmean(arr, axis=0).tolist()
        # Per-request average (ignore NaNs); fallback to 0.0 if a row is all NaNs
        avg_by_request = [
            float(np.nanmean(arr[i])) if not np.all(np.isnan(arr[i])) else 0.0
            for i in range(arr.shape[0])
        ]

        # Compute mean acceptance length (average across ALL steps in ALL requests)
        # This gives equal weight to each decoding step
        mean_acceptance_length = float(np.nanmean(arr))

        # Compute total tokens accepted and total steps
        total_tokens_accepted = int(np.nansum(arr))
        total_steps = int(np.sum(~np.isnan(arr)))

        return {
            "total_requests": len(self.acceptance_stats),
            "max_position": len(avg_by_position),
            "avg_acceptance_rate": float(np.nanmean(avg_by_request)),  # Mean of per-request means
            "mean_acceptance_length": mean_acceptance_length,  # Global mean across all steps
            "total_tokens_accepted": total_tokens_accepted,
            "total_steps": total_steps,
            "avg_by_position": avg_by_position,
            "avg_by_request": avg_by_request
        }

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Use the client with arguments from command line
    client = AcceptanceStatsClient(args.model, args.method, args.dataset, args.datapath)
    acceptance_stats = client.load_data()

    # Get summary statistics
    summary = client.get_summary_stats()
    print("Summary Statistics:")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Max Position: {summary['max_position']}")
    print(f"Total Tokens Accepted: {summary['total_tokens_accepted']}")
    print(f"Total Decoding Steps: {summary['total_steps']}")
    print(f"Mean Acceptance Length (global): {summary['mean_acceptance_length']:.2f}")
    print(f"Average Acceptance Rate (per-request mean): {summary['avg_acceptance_rate']:.2f}")
    print(f"Average Acceptance Rate by Position: {summary['avg_by_position']}")
    print(f"Average Acceptance Rate by Request: {summary['avg_by_request']}")

    # Create heatmap visualization
    plot_heatmap = True
    if plot_heatmap:
        client.plot_heatmap()
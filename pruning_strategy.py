import time
from config import Config

class PruningStrategy:
    @staticmethod
    def should_prune(node_data: 'TuberNode') -> bool:
        """
        Determines if a tuber should be pruned based on a combined value score.
        """
        current_time = time.time()
        creation_time = node_data.metadata.get("creation_time", current_time)
        age_in_seconds = current_time - creation_time

        # Avoid division by zero if age is very small
        age_in_weeks = age_in_seconds / (3600 * 24 * 7) if age_in_seconds > 1 else 1.0 / (3600 * 24 * 7)

        access_count = node_data.metadata.get("access_count", 0)
        access_frequency_factor = access_count / max(1, age_in_weeks)

        # A more nuanced value score for pruning
        # Prioritize high value_score, recent access, and frequent access
        # Penalize old, low-value, rarely accessed tubers
        combined_value = node_data.metadata.get("value_score", 0.0) \
                         + (node_data.metadata.get("last_accessed_time", 0) / current_time) * 0.1 \
                         + (access_frequency_factor * 0.05) # Small boost for frequent access

        return combined_value < Config.PRUNING_THRESHOLD, combined_value

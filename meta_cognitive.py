import random
from typing import Dict, Any, List, Optional
import logging

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

class MetaCognitiveModule:
    def __init__(self, llm_interface_instance):
        self.llm_interface = llm_interface_instance
        self.curiosity_level = 0.5 # Can be dynamically adjusted

    def detect_gaps(self, system_state: Dict[str, Any], tuber_model: 'MetaphoricalTuberingModel') -> List[str]:
        """
        Monitors system state, LLM outputs, and the knowledge graph for knowledge gaps.
        """
        gaps = []

        # --- Metric-based Gap Detection ---
        if system_state.get('avg_prediction_confidence', 1.0) < Config.GAP_CONFIDENCE_THRESHOLD:
            gaps.append(f"Low overall prediction confidence (avg: {system_state.get('avg_prediction_confidence', 1.0):.2f}) suggests the reward model is unreliable.")

        if system_state.get('avg_prediction_error', 0.0) > Config.GAP_ERROR_THRESHOLD:
            gaps.append(f"High discrepancy between predicted and actual rewards (avg error: {system_state.get('avg_prediction_error', 0.0):.2f}) indicates a flawed understanding of value.")

        if system_state.get('stagnant_exploration_cycles', 0) > Config.GAP_STAGNATION_CYCLES:
            gaps.append("Exploration has been stagnant or unsuccessful, indicating a need for new strategies or a paradigm shift.")

        if system_state.get('llm_error_rate', 0.0) > Config.GAP_LLM_ERROR_THRESHOLD:
            gaps.append(f"High LLM error rate ({system_state.get('llm_error_rate', 0.0):.2%}) suggests issues with prompts, APIs, or model capabilities.")

        # --- Graph-based Gap Detection ---
        if tuber_model and tuber_model.graph:
            # Find isolated or poorly connected tubers
            isolated_tubers = [nid for nid, degree in tuber_model.graph.degree() if degree <= 1 and nid != tuber_model.root_id]
            if len(isolated_tubers) > Config.GAP_ISOLATED_TUBER_THRESHOLD:
                gaps.append(f"Found {len(isolated_tubers)} poorly connected tubers, suggesting fragmented knowledge.")

            # Find tubers with low value but high potential (e.g., high novelty, low reward)
            # This requires more metadata to be consistently stored on tubers
            # Placeholder for a more complex query

        # --- LLM-driven Gap Detection (more advanced) ---
        # This can be a separate async method if it becomes too complex
        # async def detect_contradictions(self, tuber_model):
        #     # Use LLM to find contradictory information between tubers
        #     pass

        # --- Random Curiosity Trigger ---
        if not gaps and random.random() < Config.GAP_CURIOSITY_TRIGGER_RATE:
            gaps.append("Injecting random curiosity: What is a core assumption of our current approach that could be wrong?")

        logging.info(f"Detected {len(gaps)} knowledge gaps.")
        return gaps

    def calculate_curiosity_score(self, gaps: List[str]) -> float:
        """
        Calculates a curiosity score based on the detected gaps.
        """
        # Simple heuristic: more gaps, more curiosity
        score = min(1.0, len(gaps) * 0.25 + self.curiosity_level * 0.1)
        return score

    async def generate_inner_dialogue(self, gaps: List[str], system_state: Dict[str, Any]) -> str:
        """
        Generates an 'inner dialogue' using an LLM to reflect on the system's state and gaps.
        This dialogue helps in formulating targeted questions.
        """
        if not self.llm_interface:
            return "LLM Interface not available for inner dialogue."

        prompt = f"""
        As a meta-cognitive module for an AI system, reflect on the following state and identified gaps.
        Your goal is to generate a brief internal monologue that synthesizes these issues and points towards a direction for inquiry.

        Current System State:
        {system_state}

        Identified Gaps:
        - {"\n- ".join(gaps)}

        Generate a concise inner dialogue (2-3 sentences) that captures the essence of the current challenge or uncertainty.
        Example: 'My predictions have been off lately, especially when dealing with novel concepts. It seems I'm overfitting to past successes. Perhaps I need to question my core reward function or explore more radically different ideas.'
        """

        dialogue, _ = await self.llm_interface.generate_text(prompt, max_tokens=150, temperature=0.7)
        logging.info(f"Generated Inner Dialogue: {dialogue}")
        return dialogue

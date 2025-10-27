import random
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import time
import json
import uuid
import networkx as nx

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class ExplorationMetrics:
    novelty_score: float
    risk_level: float
    potential_impact: float
    resource_cost: float
    exploration_depth: int
    regret_potential: float = 0.0 # Added for regret integration

class EnhancedChallengeAgent:
    def __init__(self, tuber_model_instance, llm_interface_instance):
        self.tuber_model = tuber_model_instance
        self.llm_interface = llm_interface_instance
        self.exploration_budget = Config.EXPLORATION_BUDGET
        self.exploration_history = [] # Stores records of past explorations
        self.discovery_patterns = {} # To store patterns of successful discoveries

        # Advanced exploration strategies
        self.exploration_strategies = {
            'quantum_leap': self._quantum_leap_expansion,
            'reverse_thinking': self._reverse_thinking_expansion,
            'strange_combinations': self._strange_combinations_expansion,
            'conceptual_bridging': self._conceptual_bridging_expansion,
            'contradiction_synthesis': self._contradiction_synthesis_expansion,
            'temporal_shift': self._temporal_shift_expansion,
            'scale_transformation': self._scale_transformation_expansion,
            'metaphor_mining': self._metaphor_mining_expansion
        }

        # Success pattern learning
        self.successful_patterns = defaultdict(list)
        self.failed_patterns = defaultdict(list)

    async def run_adaptive_exploration_cycle(self, current_tuber_id: str, budget_multiplier: float = 1.0):
        """
        Runs an adaptive exploration cycle that learns from past successes.
        Decides which strategy to use based on learned patterns and current context.
        """
        if not self.llm_interface:
            logging.error("LLM Interface not set for Challenge Agent.")
            return

        # Determine if we have budget for exploration
        # This is a simplified check; in a real system, it would be tied to actual resource usage
        if random.random() > (self.exploration_budget * budget_multiplier):
            logging.info("Skipping exploration cycle due to budget constraints.")
            return

        logging.info(f"Initiating adaptive exploration cycle from tuber: {current_tuber_id}")

        # Decide on a strategy (can be LLM-driven based on current state, past failures/successes)
        chosen_strategy_name = await self._select_adaptive_strategy(current_tuber_id)
        chosen_strategy = self.exploration_strategies[chosen_strategy_name]

        logging.info(f"Chosen exploration strategy: {chosen_strategy_name}")

        try:
            # Execute the chosen strategy
            new_tuber_ids = await chosen_strategy(current_tuber_id)

            if new_tuber_ids:
                # Evaluate the discovery
                evaluation_prompt = f"""
                Evaluate the novelty, risk, and potential impact of these newly discovered concepts/tubers:
                {', '.join([self.tuber_model.get_tuber_content(tid) for tid in new_tuber_ids if self.tuber_model.get_tuber_content(tid) is not None])}

                Consider their relation to the starting point: {self.tuber_model.get_tuber_content(current_tuber_id)}

                Provide a JSON object with scores (0-10) for novelty, risk, potential_impact, and estimated_regret_potential (0-10).
                Example: {{"novelty": 8, "risk": 5, "potential_impact": 7, "estimated_regret_potential": 3}}
                """
                evaluation_response, _ = await self.llm_interface.generate_text(evaluation_prompt, max_tokens=100, temperature=0.5)

                try:
                    scores = json.loads(evaluation_response)
                    scores = json.loads(evaluation_response)
                    metrics = ExplorationMetrics(
                        novelty_score=scores.get('novelty', 0) / 10.0,
                        risk_level=scores.get('risk', 0) / 10.0,
                        potential_impact=scores.get('potential_impact', 0) / 10.0,
                        resource_cost=self._estimate_resource_cost(new_tuber_ids[0]), # Simplified
                        exploration_depth=self._calculate_conceptual_depth(new_tuber_ids[0]), # Simplified
                        regret_potential=scores.get('estimated_regret_potential', 0) / 10.0
                    )

                    # Learn from the outcome
                    self._learn_from_exploration(chosen_strategy_name, metrics, new_tuber_ids)

                    for tid in new_tuber_ids:
                        self.tuber_model.update_tuber_metadata(tid, 'novelty_score', metrics.novelty_score)
                        self.tuber_model.update_tuber_metadata(tid, 'risk_level', metrics.risk_level)
                        self.tuber_model.update_tuber_metadata(tid, 'potential_impact', metrics.potential_impact)
                        self.tuber_model.update_tuber_metadata(tid, 'expected_regret_cost', metrics.regret_potential)
                        self.tuber_model.update_tuber_metadata(tid, 'node_type', 'challenge_discovery')

                    logging.info(f"Exploration successful with strategy {chosen_strategy_name}. Metrics: {metrics}")

                except json.JSONDecodeError:
                    logging.error(f"Failed to parse LLM evaluation for exploration: {evaluation_response}")
                    self._learn_from_exploration(chosen_strategy_name, None, new_tuber_ids, success=False)
            else:
                logging.info(f"Exploration strategy {chosen_strategy_name} yielded no new tubers.")
                self._learn_from_exploration(chosen_strategy_name, None, [], success=False)

        except Exception as e:
            logging.error(f"Error during exploration cycle with strategy {chosen_strategy_name}: {e}")
            self._learn_from_exploration(chosen_strategy_name, None, [], success=False)

    def _learn_from_exploration(self, strategy_name: str, metrics: Optional[ExplorationMetrics], discovered_tuber_ids: List[str], success: bool = True):
        """
        Records exploration outcome and updates learning patterns.
        """
        record = {
            'timestamp': time.time(),
            'strategy': strategy_name,
            'metrics': metrics,
            'discovered_tubers': discovered_tuber_ids,
            'success': success
        }
        self.exploration_history.append(record)

        if success and metrics:
            # A simplified learning: if novelty is high and risk is acceptable, it\'s a good pattern
            if metrics.novelty_score > 0.7 and metrics.risk_level < 0.6:
                self.successful_patterns[strategy_name].append(metrics)
        else:
            self.failed_patterns[strategy_name].append(metrics)

        # Here, you would implement more sophisticated learning:
        # - Adjusting exploration budget based on overall success rate
        # - Prioritizing strategies that lead to high impact discoveries
        # - Using LLM to analyze success/failure patterns and refine strategy selection logic

    def _estimate_novelty(self, new_content: str, parent_content: str) -> float:
        """Estimate novelty of new content relative to parent using semantic distance."""
        if not hasattr(self.tuber_model, 'embedding_service'):
            logging.warning("Embedding service not available for novelty estimation.")
            return random.random()

        new_embedding = self.tuber_model.embedding_service.embed_content(new_content)
        parent_embedding = self.tuber_model.embedding_service.embed_content(parent_content)

        similarity = self.tuber_model.embedding_service.calculate_semantic_similarity(new_embedding, parent_embedding)
        novelty = 1 - similarity
        return novelty

    async def _estimate_risk(self, new_content: str) -> float:
        """Estimate risk associated with the new content/direction using an LLM call."""
        prompt = f"On a scale of 0 to 10, where 0 is no risk and 10 is extremely high risk, what is the risk of pursuing the concept: '{new_content}'? Consider risks like being a dead-end, computationally expensive, or ethically problematic. Respond with a single number."
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=10, temperature=0.2)
        try:
            return float(response.strip()) / 10.0
        except (ValueError, AttributeError):
            logging.warning(f"Could not parse risk score from LLM response: {response}")
            return random.random()

    async def _estimate_potential_impact(self, new_content: str) -> float:
        """Estimate potential impact of the new content/direction using an LLM call."""
        prompt = f"On a scale of 0 to 10, where 0 is no impact and 10 is revolutionary impact, what is the potential impact of the concept: '{new_content}'? Consider its potential for new discoveries and applications. Respond with a single number."
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=10, temperature=0.2)
        try:
            return float(response.strip()) / 10.0
        except (ValueError, AttributeError):
            logging.warning(f"Could not parse impact score from LLM response: {response}")
            return random.random()

    def _estimate_resource_cost(self, node_id: str) -> float:
        """
        Estimate resource cost based on the LLM cost to generate this tuber.
        This is a post-facto estimation, normalized to a 0-1 scale.
        """
        if node_id not in self.tuber_model.graph:
            return 0.0
        try:
            # A node can have multiple predecessors in a DiGraph, but here we assume a tree-like expansion
            parent = next(self.tuber_model.graph.predecessors(node_id))
            edge_data = self.tuber_model.graph.get_edge_data(parent, node_id)
            # Normalize cost. Heuristic: max cost for a generation is ~$0.05
            # This should be in the config file.
            max_cost = 0.05
            normalized_cost = edge_data.get("cost", 0.0) / max_cost
            return max(0.0, min(1.0, normalized_cost))
        except (StopIteration, nx.NetworkXError):
            # No parent or edge data found, e.g., for the root node.
            return 0.0

    def _calculate_conceptual_depth(self, node_id: str) -> int:
        """Calculate conceptual depth as the distance from the root tuber."""
        if not self.tuber_model.root_id or node_id not in self.tuber_model.graph:
            return 0
        try:
            return nx.shortest_path_length(self.tuber_model.graph, source=self.tuber_model.root_id, target=node_id)
        except nx.NetworkXNoPath:
            # This might happen if the graph is disconnected.
            return 0

    async def _select_adaptive_strategy(self, current_tuber_id: str) -> str:
        """Selects an exploration strategy adaptively, using LLM to reason about past performance."""
        # Summarize past performance
        success_summary = {s: len(m) for s, m in self.successful_patterns.items()}
        failure_summary = {s: len(m) for s, m in self.failed_patterns.items()}

        # Get current tuber content for context
        current_content = self.tuber_model.get_tuber_content(current_tuber_id)
        if not current_content:
            return random.choice(list(self.exploration_strategies.keys()))

        prompt = f"""
        As an AI strategist, select the best exploration strategy from the following list: {list(self.exploration_strategies.keys())}.

        Context: The current concept is '{current_content}'.

        Past Performance Summary:
        - Successful strategies (and # of successes): {success_summary}
        - Failed strategies (and # of failures): {failure_summary}

        Consider the context and past performance. A good strategy might be one that has been successful before, or a less-used one to ensure diversity. Avoid strategies that have failed often.

        Respond with ONLY the name of the chosen strategy (e.g., 'quantum_leap').
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=20, temperature=0.7)
        chosen_strategy = response.strip()

        if chosen_strategy in self.exploration_strategies:
            logging.info(f"LLM-driven strategy selection: {chosen_strategy}")
            return chosen_strategy
        else:
            logging.warning(f"LLM returned an invalid strategy '{chosen_strategy}'. Falling back to random choice.")
            return random.choice(list(self.exploration_strategies.keys()))

    # --- Advanced Exploration Strategies (Placeholder Implementations) ---
    # These methods would use LLM to generate content based on the strategy

    async def _quantum_leap_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Generates a concept far removed from the current tuber, then tries to bridge back.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        prompt = f"""
        Given the concept: \'{tuber_content}\', generate a completely unrelated, highly novel concept.
        Then, briefly explain how this new concept could *theoretically* be connected back to the original, even if the connection is tenuous.
        Provide the new concept in one sentence, followed by the connection explanation.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.9)

        # Parse response to get new concept and connection
        # Simplified parsing for example
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"quantum_leap_to_{new_concept_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'quantum_leap_connection_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'quantum_leap_discovery')
            return new_tuber_ids
        return []

    async def _reverse_thinking_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Takes a problem/concept and thinks backward from a desired (or opposite) outcome.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        prompt = f"""
        Given the concept/problem: \'{tuber_content}\', imagine the exact opposite or a completely ideal solution/outcome.
        Then, describe the steps or conditions that would lead from this ideal/opposite state back to the original concept/problem.
        Provide the ideal/opposite state in one sentence, followed by the reverse steps.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"reverse_thinking_from_{new_concept_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'reverse_thinking_steps', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'reverse_thinking_discovery')
            return new_tuber_ids
        return []

    async def _strange_combinations_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Combines the current tuber\'s concept with a seemingly unrelated, random concept.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        # Get a random tuber content from the graph (excluding current)
        all_tuber_ids = list(self.tuber_model.graph.nodes)
        if len(all_tuber_ids) < 2: return []
        random_tuber_id = random.choice([tid for tid in all_tuber_ids if tid != tuber_id])
        random_tuber_content = self.tuber_model.get_tuber_content(random_tuber_id)
        if not random_tuber_content: return []

        prompt = f"""
        Combine the concept: \'{tuber_content}\' with the concept: \'{random_tuber_content}\'.
        Generate a new, innovative concept that emerges from this unusual combination.
        Explain the synergy or unexpected insight.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.9)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"strange_combination_with_{random_tuber_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'combined_with', random_tuber_id)
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'combination_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'strange_combination_discovery')
            return new_tuber_ids
        return []

    async def _conceptual_bridging_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Identifies a gap between two concepts and generates a bridging concept.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        # Find another tuber that is conceptually distant but potentially related
        all_tuber_ids = list(self.tuber_model.graph.nodes)
        if len(all_tuber_ids) < 2: return []

        # Simple heuristic: pick a random one that\'s not too close semantically
        target_tuber_id = random.choice([tid for tid in all_tuber_ids if tid != tuber_id])
        target_tuber_content = self.tuber_model.get_tuber_content(target_tuber_id)
        if not target_tuber_content: return []

        prompt = f"""
        Identify the conceptual gap between \'{tuber_content}\' and \'{target_tuber_content}\'.
        Generate a new concept that acts as a bridge or a missing link between them.
        Explain how this new concept connects the two.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"bridge_to_{target_tuber_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'bridges_between', [tuber_id, target_tuber_id])
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'bridging_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'conceptual_bridge_discovery')
            return new_tuber_ids
        return []

    async def _contradiction_synthesis_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Identifies a contradiction or paradox within a concept and synthesizes a new concept that resolves it.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        prompt = f"""
        Identify a potential contradiction or paradox within the concept: \'{tuber_content}\'.
        Generate a new concept that resolves this contradiction or synthesizes a higher-level understanding.
        Explain the contradiction and its resolution.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"contradiction_resolution_for_{tuber_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'resolved_contradiction_in', tuber_id)
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'resolution_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'contradiction_synthesis_discovery')
            return new_tuber_ids
        return []

    async def _temporal_shift_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Examines a concept from a different temporal perspective (past, future, accelerated, slowed).
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        temporal_perspectives = ["from the distant past", "from a far future", "in an accelerated timeline", "in a slowed-down process"]
        chosen_perspective = random.choice(temporal_perspectives)

        prompt = f"""
        Re-examine the concept: \'{tuber_content}\' from the perspective of \'{chosen_perspective}\'.
        Generate a new insight or concept that emerges from this temporal shift.
        Explain how the temporal shift changes the understanding.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"temporal_shift_{chosen_perspective[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'temporal_perspective', chosen_perspective)
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'temporal_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'temporal_shift_discovery')
            return new_tuber_ids
        return []

    async def _scale_transformation_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Examines a concept by drastically changing its scale (micro to macro, individual to global).
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        scale_transforms = ["at a microscopic level", "from a cosmic scale", "as an individual entity", "as a global phenomenon"]
        chosen_transform = random.choice(scale_transforms)

        prompt = f"""
        Re-examine the concept: \'{tuber_content}\' by transforming its scale to \'{chosen_transform}\'.
        Generate a new insight or concept that emerges from this scale transformation.
        Explain how the scale transformation changes the understanding.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"scale_transform_{chosen_transform[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'scale_transformation', chosen_transform)
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'scale_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'scale_transformation_discovery')
            return new_tuber_ids
        return []

    async def _metaphor_mining_expansion(self, tuber_id: str, depth: int = 1) -> List[str]:
        """
        Extracts underlying metaphors from a concept and explores them.
        """
        tuber_content = self.tuber_model.get_tuber_content(tuber_id)
        if not tuber_content: return []

        prompt = f"""
        Identify a core metaphor or analogy inherent in the concept: \'{tuber_content}\'.
        Explore this metaphor and generate a new concept or insight by extending the metaphor.
        Explain the original metaphor and how its extension leads to the new insight.
        """
        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.8)
        parts = response.split('\n', 1)
        new_concept_content = parts[0] if parts else response

        new_tuber_ids = await self.tuber_model.expand(parent_id=tuber_id, directions=[f"metaphor_mining_from_{tuber_content[:20]}..."])
        if new_tuber_ids:
            new_tuber_id = new_tuber_ids[0]
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'original_concept', tuber_id)
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'metaphor_explanation', parts[1] if len(parts) > 1 else "")
            self.tuber_model.update_tuber_metadata(new_tuber_id, 'node_type', 'metaphor_mining_discovery')
            return new_tuber_ids
        return []


# Example usage (for testing):
if __name__ == "__main__":
    # Mock TuberModel and LLMInterface for testing ChallengeAgent in isolation
    class MockTuberModel:
        def __init__(self):
            self.nodes = {}
            self.graph = nx.DiGraph() # Mock graph for node existence check
            self.root_id = None

        def get_tuber_content(self, tuber_id: str) -> Optional[str]:
            return self.nodes.get(tuber_id, {}).get('content')

        def update_tuber_metadata(self, tuber_id: str, key: str, value: Any):
            if tuber_id in self.nodes:
                self.nodes[tuber_id]['metadata'][key] = value

        async def expand(self, parent_id: str, directions: List[str], expansion_type: str = 'general') -> List[str]:
            new_ids = []
            for direction in directions:
                new_id = str(uuid.uuid4())
                self.nodes[new_id] = {'content': f"Expanded content for {direction}", 'metadata': {}}
                self.graph.add_node(new_id)
                self.graph.add_edge(parent_id, new_id)
                new_ids.append(new_id)
            return new_ids

        def seed_idea(self, content: str, node_type: str = 'root') -> str:
            node_id = str(uuid.uuid4())
            self.nodes[node_id] = {'content': content, 'metadata': {}, 'node_type': node_type}
            self.graph.add_node(node_id)
            self.root_id = node_id
            return node_id

    class MockLLMInterface:
        async def generate_text(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Tuple[str, float]:
            if "Evaluate the novelty" in prompt:
                return json.dumps({"novelty": random.randint(5, 10), "risk": random.randint(1, 5), "potential_impact": random.randint(5, 10), "estimated_regret_potential": random.randint(1, 5)}), 0.005
            return f"Mock LLM response for: {prompt[:50]}...\nExplanation: This is a detailed explanation of the mock concept.", 0.01

    # Setup
    mock_tuber_model = MockTuberModel()
    mock_llm_interface = MockLLMInterface()
    challenge_agent = EnhancedChallengeAgent(mock_tuber_model, mock_llm_interface)

    # Seed a starting tuber
    start_tuber_id = mock_tuber_model.seed_idea("Initial concept for AI creativity")
    print(f"Starting exploration from: {start_tuber_id}")

    # Run an exploration cycle
    import asyncio
    asyncio.run(challenge_agent.run_adaptive_exploration_cycle(start_tuber_id))

    print("\n--- Exploration History ---")
    for record in challenge_agent.exploration_history:
        print(f"Strategy: {record['strategy']}, Success: {record['success']}, Metrics: {record['metrics']}")

    print("\n--- Successful Patterns ---")
    for strategy, metrics_list in challenge_agent.successful_patterns.items():
        print(f"Strategy {strategy}: {len(metrics_list)} successes")

    print("\n--- Failed Patterns ---")
    for strategy, metrics_list in challenge_agent.failed_patterns.items():
        print(f"Strategy {strategy}: {len(metrics_list)} failures")

    # Simulate multiple cycles
    print("\n--- Running multiple cycles ---")
    for _ in range(5):
        asyncio.run(challenge_agent.run_adaptive_exploration_cycle(start_tuber_id))

    print("\n--- Final Exploration History Summary ---")
    success_count = sum(1 for r in challenge_agent.exploration_history if r['success'])
    total_count = len(challenge_agent.exploration_history)
    print(f"Total explorations: {total_count}, Successful: {success_count}, Failed: {total_count - success_count}")

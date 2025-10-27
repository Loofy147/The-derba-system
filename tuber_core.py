import uuid
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time

from config import Config
from embedding_service import EmbeddingService
from pruning_strategy import PruningStrategy
from prompt_templates import AKT_UPDATE_PROMPT

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class TuberNode:
    """
    Represents a 'drna' (tuber) in the metaphorical model.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    node_type: str = "subtuber"  # 'root', 'subtuber', 'latent', 'meta_knowledge', 'decision_log', 'challenge_discovery', 'meta_process'
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        # Initialize metadata with default values if not present
        self.metadata.setdefault("predicted_future_reward", Config.INITIAL_REWARD_PREDICTION)
        self.metadata.setdefault("prediction_confidence", 0.5)
        self.metadata.setdefault("short_term_reward", 0.0)
        self.metadata.setdefault("long_term_reward", 0.0)
        self.metadata.setdefault("last_prediction_error", 0.0)
        self.metadata.setdefault("value_score", 0.0) # Overall value for pruning
        self.metadata.setdefault("expected_regret_cost", 0.0)
        self.metadata.setdefault("transport_speed", Config.DEFAULT_TRANSPORT_SPEED)
        self.metadata.setdefault("transport_cost", Config.DEFAULT_TRANSPORT_COST)
        self.metadata.setdefault("transport_capacity", Config.DEFAULT_TRANSPORT_CAPACITY)
        self.metadata.setdefault("last_llm_insight_time", 0.0)
        self.metadata.setdefault("creation_time", time.time())
        self.metadata.setdefault("last_accessed_time", time.time())
        self.metadata.setdefault("access_count", 0)


class MetaphoricalTuberingModel:
    """
    A dynamic, branching knowledge graph inspired by tubers:
    - Nodes grow in depth and breadth
    - Can seed ideas, expand, prune, and query latent knowledge
    - Implements the Active Knowledge Triangle (AKT) for dynamic focus.
    """
    def __init__(self, embedding_service: EmbeddingService):
        self.graph = nx.DiGraph()  # Directed graph: edges point from parent to child
        self.root_id: Optional[str] = None
        self.embedding_service = embedding_service
        self.active_knowledge_triangle: List[str] = [] # Stores IDs of the 3 active tubers
        self.llm_interface = None # Will be set by TuberOrchestratorAI

    def set_llm_interface(self, llm_interface_instance):
        self.llm_interface = llm_interface_instance

    def seed_idea(self, content: str, node_type: str = "root", metadata: Optional[Dict] = None) -> str:
        """
        Create the root tuber (الفكرة الأصلية).
        """
        embedding = self.embedding_service.embed_content(content)
        node = TuberNode(content=content, node_type=node_type, embedding=embedding, metadata=metadata if metadata else {})
        self.graph.add_node(node.id, data=node)
        self.root_id = node.id
        if not self.active_knowledge_triangle:
            self.active_knowledge_triangle = [node.id] * 3 # Initialize AKT with root
        logging.info(f"Root tuber seeded: {node.id} - '{node.content[:50]}...' ")
        return node.id

    async def expand(self, parent_id: str, directions: List[str], expansion_type: str = "general") -> List[str]:
        """
        Expand a parent tuber into new child tubers in different conceptual directions.
        Uses LLM to generate content for new tubers based on parent and direction.
        """
        if parent_id not in self.graph:
            logging.warning(f"Parent tuber {parent_id} not found for expansion.")
            return []

        parent_node = self.graph.nodes[parent_id]["data"]
        new_tuber_ids = []

        logging.info(f"Expanding tuber {parent_id} ('{parent_node.content[:30]}...') in directions: {directions}")

        # Prepare prompts for batching
        prompts = []
        for direction in directions:
            prompt = f"""
            Given the parent concept: '{parent_node.content}'
            Expand on this concept in the following direction/aspect: '{direction}'.
            Focus on a '{expansion_type}' type of expansion.
            Provide a concise, single-paragraph content for the new tuber.
            """
            prompts.append(prompt)

        if self.llm_interface:
            # Use batch generation for efficiency
            results = await self.llm_interface.batch_generate_text(prompts, max_tokens=200, temperature=0.7)
            for i, (new_content, cost) in enumerate(results):
                direction = directions[i] # Get the corresponding direction
                if new_content.startswith("Error:"):
                    logging.error(f"LLM error during expansion for direction {direction}: {new_content}")
                    continue

                embedding = self.embedding_service.embed_content(new_content)
                new_node = TuberNode(content=new_content, node_type="subtuber", embedding=embedding)
                self.graph.add_node(new_node.id, data=new_node)
                self.graph.add_edge(parent_id, new_node.id, type=expansion_type, direction=direction, cost=cost)
                new_tuber_ids.append(new_node.id)
                logging.info(f"  - Created child tuber {new_node.id} for direction '{direction}'")
        else:
            # Fallback if LLM interface is not set
            for direction in directions:
                new_content = f"Expansion of '{parent_node.content[:20]}...' in direction '{direction}'"
                embedding = self.embedding_service.embed_content(new_content)
                new_node = TuberNode(content=new_content, node_type="subtuber", embedding=embedding)
                self.graph.add_node(new_node.id, data=new_node)
                self.graph.add_edge(parent_id, new_node.id, type=expansion_type, direction=direction, cost=0.0)
                new_tuber_ids.append(new_node.id)
                logging.info(f"  - Created child tuber {new_node.id} for direction '{direction}'")

        return new_tuber_ids

    def prune(self) -> List[str]:
        """
        Prune tubers with low value scores, excluding root and active knowledge triangle.
        """
        pruned_ids = []
        for node_id in list(self.graph.nodes):
            if node_id == self.root_id or node_id in self.active_knowledge_triangle:
                continue

            node_data = self.graph.nodes[node_id]["data"]
            should_prune, combined_value = PruningStrategy.should_prune(node_data)
            if should_prune:
                self.graph.remove_node(node_id)
                pruned_ids.append(node_id)
                logging.info(f"Pruned tuber: {node_id} - '{node_data.content[:30]}...' (Value: {combined_value:.2f})")
        return pruned_ids

    def traverse(self, start_node_id: Optional[str] = None, depth: int = 3) -> Dict:
        """
        Traverse the tuber network from a start node (or root) up to a certain depth.
        """
        if not self.graph.nodes:
            return {"nodes": [], "edges": []}

        if start_node_id is None:
            start_node_id = self.root_id
            if start_node_id is None:
                return {"nodes": [], "edges": []}

        visited_nodes = set()
        nodes_to_visit = [(start_node_id, 0)]
        traversed_nodes_data = []
        traversed_edges_data = []

        while nodes_to_visit:
            current_node_id, current_depth = nodes_to_visit.pop(0)

            if current_node_id in visited_nodes or current_depth > depth:
                continue

            visited_nodes.add(current_node_id)
            node_data = self.graph.nodes[current_node_id]["data"]
            traversed_nodes_data.append({
                "id": node_data.id,
                "content": node_data.content,
                "type": node_data.node_type,
                "metadata": node_data.metadata
            })

            for neighbor_id in self.graph.neighbors(current_node_id):
                edge_data = self.graph.get_edge_data(current_node_id, neighbor_id)
                traversed_edges_data.append({
                    "source": current_node_id,
                    "target": neighbor_id,
                    "type": edge_data.get("type", "relates_to"),
                    "direction": edge_data.get("direction", "N/A"),
                    "cost": edge_data.get("cost", 0.0)
                })
                nodes_to_visit.append((neighbor_id, current_depth + 1))

        return {"nodes": traversed_nodes_data, "edges": traversed_edges_data}

    def to_dict(self) -> Dict:
        """
        Serialize the graph to a dictionary for easy representation/storage.
        """
        nodes = [{
            "id": nid,
            "content": self.graph.nodes[nid]["data"].content,
            "type": self.graph.nodes[nid]["data"].node_type,
            "metadata": self.graph.nodes[nid]["data"].metadata,
            "embedding": self.graph.nodes[nid]["data"].embedding # Include embedding for external use if needed
        } for nid in self.graph.nodes]
        edges = [{
            "source": u,
            "target": v,
            "type": self.graph.get_edge_data(u, v).get("type", "relates_to"),
            "direction": self.graph.get_edge_data(u, v).get("direction", "N/A"),
            "cost": self.graph.get_edge_data(u, v).get("cost", 0.0)
        } for u, v in self.graph.edges]
        return {"nodes": nodes, "edges": edges, "root_id": self.root_id, "active_knowledge_triangle": self.active_knowledge_triangle}

    async def update_active_knowledge_triangle(self, new_tuber_id: str):
        """
        Updates the Active Knowledge Triangle (AKT) by replacing one of its members.
        The replacement strategy is now LLM-driven, aiming for a balanced and relevant triangle.
        """
        if new_tuber_id not in self.graph:
            logging.warning(f"New tuber {new_tuber_id} not found in graph. Cannot update AKT.")
            return

        # Update access time and count for the new tuber
        self.update_tuber_metadata(new_tuber_id, "last_accessed_time", time.time())
        self.update_tuber_metadata(new_tuber_id, "access_count", self.get_tuber_data(new_tuber_id).metadata.get("access_count", 0) + 1)

        if len(self.active_knowledge_triangle) < 3:
            if new_tuber_id not in self.active_knowledge_triangle:
                self.active_knowledge_triangle.append(new_tuber_id)
                logging.info(f"AKT: Added {new_tuber_id}. Current AKT: {self.active_knowledge_triangle}")
        else:
            # LLM-driven replacement strategy
            current_akt_contents = [self.get_tuber_content(tid) for tid in self.active_knowledge_triangle]
            new_tuber_content = self.get_tuber_content(new_tuber_id)

            prompt = AKT_UPDATE_PROMPT.format(
                current_akt_contents=current_akt_contents,
                new_tuber_content=new_tuber_content
            )
            response, _ = await self.llm_interface.generate_text(prompt, max_tokens=50, temperature=0.5)
            tuber_to_replace = response.strip()

            if tuber_to_replace and tuber_to_replace != 'None' and tuber_to_replace in self.active_knowledge_triangle:
                self.active_knowledge_triangle.remove(tuber_to_replace)
                self.active_knowledge_triangle.append(new_tuber_id)
                logging.info(f"AKT: Replaced {tuber_to_replace} with {new_tuber_id}. Current AKT: {self.active_knowledge_triangle}")
            else:
                logging.info(f"AKT: No replacement made. New tuber {new_tuber_id} not added to AKT.")

    def get_tuber_data(self, tuber_id: str) -> Optional[TuberNode]:
        """Returns the TuberNode object for a given ID."""
        return self.graph.nodes.get(tuber_id, {}).get("data")

    def get_tuber_content(self, tuber_id: str) -> Optional[str]:
        """Returns the content of a tuber by its ID."""
        node_data = self.get_tuber_data(tuber_id)
        return node_data.content if node_data else None

    def update_tuber_metadata(self, tuber_id: str, key: str, value: Any):
        """Updates a specific metadata field for a tuber."""
        node_data = self.get_tuber_data(tuber_id)
        if node_data:
            node_data.metadata[key] = value
        else:
            logging.warning(f"Attempted to update metadata for non-existent tuber: {tuber_id}")

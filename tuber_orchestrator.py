import json
import copy
import time
import logging
import numpy as np
import random
import itertools
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config import Config
from llm_interface import LLMInterface
from tuber_core import MetaphoricalTuberingModel, TuberNode
from challenge_agent import EnhancedChallengeAgent
from meta_cognitive import MetaCognitiveModule
from question_generator import QuestionGenerator
from question_classifier import QuestionQualityClassifier
from question_memory import FailedQuestionMemory

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExperimentResult:
    parameter_values: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    errors: List[str]

class TuberOrchestratorAI:
    def __init__(self):
        self.config = Config()
        self.llm_interface = LLMInterface()
        self.tuber_model = MetaphoricalTuberingModel()
        self.tuber_model.set_llm_interface(self.llm_interface)
        self.challenge_agent = EnhancedChallengeAgent(self.tuber_model, self.llm_interface)

        # New Question Generation Modules
        self.meta_cog = MetaCognitiveModule()
        self.qgen = QuestionGenerator()
        self.qclf = QuestionQualityClassifier()
        self.qmem = FailedQuestionMemory()
        self.session_questions = 0

        self.experiment_history = []
        self.active_experiments = {}
        self.code_suggestions_cache = {}
        self.total_cost = 0.0

        logging.info("TuberOrchestratorAI initialized with Question Generation capabilities.")

    def seed_root_vision(self, vision_statement: str):
        """Seeds the root tuber with the core vision of the system."""
        self.tuber_model.seed_idea(vision_statement, node_type=\'root\')
        logging.info(f"Root vision seeded: \n{vision_statement}")

    def _select_llm_strategy(self, task_complexity: float) -> str:
        """
        Selects the appropriate LLM strategy (local or cloud) based on task complexity.
        Returns the provider name (\'local\', \'openai\', \'anthropic\').
        """
        if task_complexity < Config.LOCAL_LLM_COMPLEXITY_THRESHOLD and Config.LLM_PROVIDER == "local":
            return "local"
        elif Config.LLM_PROVIDER == "openai":
            return "openai"
        elif Config.LLM_PROVIDER == "anthropic":
            return "anthropic"
        else:
            # Fallback to default if local is not configured or complexity is too high
            return Config.LLM_PROVIDER

    async def _call_llm_with_strategy(self, prompt: str, task_complexity: float, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Tuple[str, float]:
        """
        Calls the LLM using the selected strategy.
        """
        selected_provider = self._select_llm_strategy(task_complexity)

        # Temporarily override LLM_PROVIDER in Config for the call
        original_provider = self.llm_interface.provider
        self.llm_interface.provider = selected_provider

        try:
            response, cost = await self.llm_interface.generate_text(prompt, max_tokens, temperature)
        finally:
            # Restore original provider
            self.llm_interface.provider = original_provider

        return response, cost

    async def run_automated_cycle(self, input_data: Dict[str, Any]):
        """
        Runs a full automated cycle of the orchestrator, including the new question generation logic.
        """
        logging.info("--- Starting Automated Cycle ---")

        # 1. Run the internal improvement cycle (Question Generation)
        await self._run_internal_improvement_cycle(input_data)

        # 2. Run the challenge agent exploration
        await self.challenge_agent.run_adaptive_exploration_cycle()

        # 3. Perform regular maintenance (e.g., pruning)
        self.tuber_model.prune()

        # 4. Self-reflection and self-modeling
        await self._perform_self_reflection()

        logging.info("--- Automated Cycle Finished ---")

    async def _run_internal_improvement_cycle(self, input_data: Dict[str, Any]):
        """
        The core logic for self-improvement through question generation.
        """
        logging.info("Running meta-cognitive analysis and question generation...")
        self.session_questions = 0 # Reset for each cycle

        # 1. Detect knowledge gaps
        gaps = self.meta_cog.detect_gaps(input_data)

        if gaps and self.meta_cog.curiosity_score(gaps) > self.config.CURIOSITY_THRESHOLD:
            logging.info(f"Curiosity threshold met. Detected gaps: {gaps}")

            # 2. Generate a question
            raw_question = self.qgen.generate(input_data, gaps)
            logging.info(f"Generated raw question: {raw_question}")

            # 3. Classify question quality
            if self.qclf.score(raw_question) >= self.qclf.threshold:
                logging.info("Question quality is acceptable.")

                # Use smart LLM strategy for asking questions (medium complexity)
                answer, cost = await self._call_llm_with_strategy(raw_question, task_complexity=0.5)
                self.total_cost += cost
                self.session_questions += 1

                # 5. Evaluate the answer and learn from it
                if self.qclf.is_useful(answer):
                    logging.info(f"Received a useful answer. Performing micro-tuning.")
                    # This is where the system learns and improves
                    self.qgen.micro_tune_on_qa(raw_question, answer)
                    # You could also update tuber rewards or create new tubers based on the answer
                else:
                    logging.warning(f"Answer was not useful. Storing question in failed memory.")
                    self.qmem.store(raw_question, context=input_data)

                # Check if we\\'ve reached the max questions for this session
                if self.session_questions >= self.config.MAX_QUESTIONS_PER_SESSION:
                    logging.info("Maximum questions per session reached.")
                    return
            else:
                logging.warning("Generated question quality is too low. Rephrasing or discarding.")
                # Optionally, try to rephrase the question
                rephrased_question = self.qclf.rephrase(raw_question)
                # ... and run the process again with the new question
        else:
            logging.info("No significant knowledge gaps detected or curiosity threshold not met.")

    async def _perform_self_reflection(self):
        """
        Performs self-reflection and self-modeling, generating meta-cognitive tubers.
        """
        logging.info("Performing self-reflection and self-modeling...")
        system_status = self.get_system_status()

        prompt = f"""
        As a self-aware AI system, analyze your current status and performance:
        {json.dumps(system_status, indent=2)}

        Identify key strengths, weaknesses, and potential areas for self-improvement.
        Generate insights that can be stored as 'meta_cognitive_process' tubers.
        Focus on how your internal algorithms and knowledge management can be improved.
        Provide actionable insights for your own development.
        """
        # Use smart LLM strategy for self-reflection (very high complexity)
        reflection_insight, cost = await self._call_llm_with_strategy(prompt, task_complexity=0.95, max_tokens=1000, temperature=0.2)
        self.total_cost += cost

        # Create a meta-cognitive tuber with the reflection insight
        self.tuber_model.seed_idea(
            content=reflection_insight,
            node_type=\'meta_cognitive_process\',
            metadata={
                \'source\': \'self_reflection\',
                \'timestamp\': time.time(),
                \'system_status_snapshot\': system_status
            }
        )
        logging.info("Self-reflection complete. Meta-cognitive tuber created.")

    def _update_predicted_rewards(self, contributing_tubers_ids: List[str], actual_reward: float, context: Dict):
        """
        Enhanced reward prediction update with multiple learning strategies.
        """
        for tuber_id in contributing_tubers_ids:
            if tuber_id in self.tuber_model.graph.nodes:
                self._update_single_tuber_reward(tuber_id, actual_reward, context)

    def _update_single_tuber_reward(self, tuber_id: str, actual_reward: float, context: Dict):
        """
        Update reward prediction for a single tuber with enhanced learning.
        """
        tuber_data = self.tuber_model.get_tuber_data(tuber_id)
        if not tuber_data:
            return

        current_predicted_reward = tuber_data.metadata.get(\'predicted_future_reward\', 0.0)

        # Calculate Expected Regret Cost (Simplified for now)
        # This would be a more complex calculation based on potential negative outcomes
        expected_regret_cost = tuber_data.metadata.get(\'expected_regret_cost\', 0.0) # Placeholder

        # Apply regret cost as a discount to the actual reward for learning
        adjusted_actual_reward = actual_reward - expected_regret_cost * Config.REGRET_PENALTY_FACTOR

        # 1. Adaptive Learning Rate based on prediction confidence
        confidence = tuber_data.metadata.get(\'prediction_confidence\', 0.5)
        adaptive_lr = Config.LONG_TERM_LR * (1.0 - confidence)  # Lower confidence -> higher learning rate

        # 2. Multi-timescale Learning
        short_term_reward = tuber_data.metadata.get(\'short_term_reward\', 0.0)
        short_term_reward = (1 - Config.SHORT_TERM_LR) * short_term_reward + Config.SHORT_TERM_LR * adjusted_actual_reward

        long_term_reward = current_predicted_reward
        long_term_reward = (1 - adaptive_lr) * long_term_reward + adaptive_lr * adjusted_actual_reward

        # 3. Context-Aware Prediction (Simplified for now)
        # In a full implementation, this would use context embeddings
        context_similarity_score = 1.0 # Placeholder
        final_reward = long_term_reward * context_similarity_score

        # 4. Update Prediction Confidence
        prediction_error = abs(adjusted_actual_reward - current_predicted_reward)
        if prediction_error < tuber_data.metadata.get(\'last_prediction_error\', 0.0):
            confidence = min(1.0, confidence + Config.CONFIDENCE_GROWTH_RATE)
        else:
            confidence = max(0.0, confidence - Config.CONFIDENCE_DECAY_RATE)

        self.tuber_model.update_tuber_metadata(tuber_id, \'predicted_future_reward\', final_reward)
        self.tuber_model.update_tuber_metadata(tuber_id, \'short_term_reward\', short_term_reward)
        self.tuber_model.update_tuber_metadata(tuber_id, \'prediction_confidence\', confidence)
        self.tuber_model.update_tuber_metadata(tuber_id, \'last_prediction_error\', prediction_error)
        self.tuber_model.update_tuber_metadata(tuber_id, \'expected_regret_cost\', expected_regret_cost) # Update regret cost

        # 5. LLM-driven Insight Generation
        if prediction_error > Config.LLM_INSIGHT_ERROR_THRESHOLD and (time.time() - tuber_data.metadata.get(\'last_llm_insight_time\', 0) > 3600):
            self._generate_llm_insight(tuber_id)

    async def _generate_llm_insight(self, tuber_id: str):
        """
        Use LLM to analyze performance and generate insights.
        """
        tuber_data = self.tuber_model.get_tuber_data(tuber_id)
        if not tuber_data: return

        prompt = f"""
        Analyze the performance of the following knowledge tuber:
        Content: \n{tuber_data.content}
        Metadata: \n{json.dumps(tuber_data.metadata, indent=2)}

        The prediction error for its reward is high.
        What are the potential reasons for this discrepancy?
        What actions could be taken to improve its performance or our prediction accuracy?
        Provide actionable insights.
        """
        # Use smart LLM strategy for insight generation (high complexity)
        insight, cost = await self._call_llm_with_strategy(prompt, task_complexity=0.9, max_tokens=1500, temperature=0.3)
        self.total_cost += cost
        self.tuber_model.update_tuber_metadata(tuber_id, \'llm_insight\', insight)
        self.tuber_model.update_tuber_metadata(tuber_id, \'last_llm_insight_time\', time.time())
        logging.info(f"Generated LLM insight for tuber {tuber_id}")

    async def developer_converse(self, message: str, **kwargs) -> str:
        """
        Enhanced developer conversation with automated development capabilities.
        """
        logging.info(f"Developer interaction: \n{message}", kwargs: {kwargs})
        action_type = kwargs.get("action_type", "chat")
        response_message = ""

        if action_type == "suggest_code_change":
            problem_description = kwargs.get("problem_description", "")
            context = kwargs.get("context", "")
            priority = kwargs.get("priority", "medium")

            if problem_description:
                try:
                    suggested_code_info = await self._generate_code_suggestion(
                        problem_description, context, priority
                    )
                    response_message += f"🔧 **Code Suggestion Generated**\n"
                    response_message += f"Priority: {priority.upper()}\n"
                    response_message += f"Problem: {problem_description}\n\n"
                    response_message += f"{suggested_code_info}\n\n"
                    response_message += "⚠️ **Review Required**: Please review and test before applying.\n"
                    response_message += "Use \'validate_code_suggestion\' to run automated checks.\n"

                    # Cache the suggestion for validation
                    suggestion_id = f"suggestion_{len(self.code_suggestions_cache)}"
                    self.code_suggestions_cache[suggestion_id] = {
                        \'code\': suggested_code_info,
                        \'problem\': problem_description,
                        \'context\': context,
                        \'priority\': priority,
                        \'timestamp\': time.time()
                    }
                    response_message += f"Suggestion ID: {suggestion_id}\n"

                except Exception as e:
                    response_message += f"❌ Error generating code suggestion: {str(e)}\n"
            else:
                response_message += "❌ Missing \'problem_description\' for code suggestion.\n"

        elif action_type == "propose_automated_experiment":
            goal = kwargs.get("goal", "overall system performance")
            complexity = kwargs.get("complexity", "medium")
            duration_limit = kwargs.get("duration_limit_minutes", Config.EXPERIMENT_DURATION_LIMIT_MINUTES)

            try:
                experiment_plan = await self._propose_experiment(goal, complexity, duration_limit)
                experiment_id = f"exp_{len(self.active_experiments)}"

                response_message += f"🧪 **Automated Experiment Proposed**\n"
                response_message += f"Goal: {goal}\n"
                response_message += f"Complexity: {complexity}\n"
                response_message += f"Estimated Duration: {duration_limit} minutes\n\n"
                response_message += f"**Experiment Plan (ID: {experiment_id})**:\n"
                response_message += self._format_experiment_plan(experiment_plan)
                response_message += f"\n\n**Next Steps:**\n"
                response_message += f"- Review the plan above\n"
                response_message += f"- Run: `run_automated_experiment(\\'{experiment_id}\\'`\n"
                response_message += f"- Or modify: `modify_experiment(\\'{experiment_id}\\'`, parameters)`\n"

                # Store the experiment plan
                self.active_experiments[experiment_id] = {
                    \'plan\': experiment_plan,
                    \'status\': ExperimentStatus.PENDING,
                    \'created_at\': time.time(),
                    \'goal\': goal
                }

            except Exception as e:
                response_message += f"❌ Error proposing experiment: {str(e)}\n"

        elif action_type == "validate_code_suggestion":
            suggestion_id = kwargs.get("suggestion_id", "")
            if suggestion_id in self.code_suggestions_cache:
                validation_result = self._validate_code_suggestion(suggestion_id)
                response_message += f"🔍 **Code Validation Results for {suggestion_id}**\n"
                response_message += validation_result
            else:
                response_message += f"❌ Suggestion ID \'{suggestion_id}\' not found.\n"

        elif action_type == "run_automated_experiment":
            experiment_id = kwargs.get("experiment_id", "")
            if experiment_id in self.active_experiments:
                response_message += f"🚀 **Starting Experiment {experiment_id}**\n"
                response_message += "This may take several minutes...\n"

                try:
                    results = await self._run_automated_experiment(experiment_id)
                    response_message += self._format_experiment_results(results)
                except Exception as e:
                    response_message += f"❌ Experiment failed: {str(e)}\n"
                    self.active_experiments[experiment_id][\'status\'] = ExperimentStatus.FAILED
            else:
                response_message += f"❌ Experiment ID \'{experiment_id}\' not found.\n"

        elif action_type == "list_experiments":
            response_message += self._list_experiments()

        elif action_type == "analyze_system_health":
            health_report = self._analyze_system_health()
            response_message += f"🏥 **System Health Report**\n{health_report}\n"

        else: # Default chat behavior
            prompt = f"As the TuberOrchestratorAI, respond to the developer: \n{message}"
            # Use smart LLM strategy for general chat (low complexity)
            response, cost = await self._call_llm_with_strategy(prompt, task_complexity=0.1)
            self.total_cost += cost
            response_message = response

        return response_message

    async def _generate_code_suggestion(self, problem_description: str, context: str = "", priority: str = "medium") -> str:
        """
        Enhanced code suggestion generation with context awareness.
        """
        # Get current system state for context
        system_context = self._get_system_context()

        prompt = f"""
        As a senior Python developer and AI systems architect, provide a detailed code suggestion for:

        **Problem**: {problem_description}
        **Context**: {context}
        **Priority**: {priority}

        **Current System Context**:
        - Active components: {system_context.get(\'active_components\', [])}
        - Recent performance metrics: {system_context.get(\'recent_metrics\', {})}
        - Configuration state: {system_context.get(\'config_summary\', {})}

        **Requirements**:
        1. Provide complete, production-ready Python code
        2. Include error handling and logging
        3. Add docstrings and type hints
        4. Consider backwards compatibility
        5. Suggest tests if applicable
        6. Explain the reasoning behind the solution

        **Format**:
        ```python
        # Your code here
        ```

        **Explanation**: Brief explanation of the solution and its benefits.
        **Testing**: Suggested test cases or validation steps.
        **Risks**: Any potential risks or considerations.
        """

        # Use smart LLM strategy for code suggestion (high complexity)
        code_suggestion, cost = await self._call_llm_with_strategy(prompt, task_complexity=0.8, max_tokens=1500, temperature=0.3)
        self.total_cost += cost
        return code_suggestion

    async def _propose_experiment(self, goal: str, complexity: str = "medium", duration_limit: int = 30) -> Dict:
        """
        Enhanced experiment proposal with adaptive complexity.
        """
        # Get system metrics history for context
        metrics_history = self._get_metrics_history()

        complexity_configs = {
            "simple": {"max_parameters": 3, "max_iterations": 10, "max_values_per_param": 3},
            "medium": {"max_parameters": Config.MAX_EXPERIMENT_PARAMETERS, "max_iterations": Config.MAX_EXPERIMENT_ITERATIONS, "max_values_per_param": Config.MAX_VALUES_PER_PARAMETER},
            "complex": {"max_parameters": 8, "max_iterations": 50, "max_values_per_param": 7}
        }

        config = complexity_configs.get(complexity, complexity_configs["medium"])

        prompt = f"""
        Design an automated experiment plan to improve \'{goal}\' in the TuberOrchestratorAI system.

        **Constraints**:
        - Maximum {config[\'max_parameters\']} parameters to test
        - Maximum {config[\'max_iterations\']} total iterations
        - Maximum {config[\'max_values_per_param\']} values per parameter
        - Duration limit: {duration_limit} minutes

        **Current System State**:
        - Recent performance: {metrics_history.get(\'recent_avg\', {})}
        - Performance trends: {metrics_history.get(\'trends\', {})}
        - Bottlenecks identified: {metrics_history.get(\'bottlenecks\', [])}

        **Available Parameters to Test**:
        - llm_temperature (0.1-1.0)
        - llm_max_tokens (100-2000)
        - pruning_threshold (0.1-0.9)
        - cache_size (100-1000)
        - batch_size (1-20)
        - timeout_seconds (5-60)

        Return a JSON object with this structure:
        {{
            "experiment_name": "descriptive name",
            "hypothesis": "what you expect to find",
            "parameters": {{
                "param_name": {{
                    "values": [list of values to test],
                    "description": "what this parameter controls"
                }}
            }},
            "metrics": ["list", "of", "metrics", "to", "measure"],
            "iterations": number_of_iterations,
            "success_criteria": "how to determine if experiment succeeded",
            "estimated_duration_minutes": estimated_time
        }}
        """

        # Use smart LLM strategy for experiment proposal (medium-high complexity)
        experiment_json, cost = await self._call_llm_with_strategy(prompt, task_complexity=0.7, max_tokens=800, temperature=0.4)
        self.total_cost += cost

        try:
            experiment_plan = json.loads(experiment_json)
            # Validate and sanitize the plan
            experiment_plan = self._validate_experiment_plan(experiment_plan)
            return experiment_plan
        except json.JSONDecodeError:
            logging.error(f"LLM failed to generate valid JSON for experiment plan: {experiment_json}")
            # Fallback to a simple default experiment
            return self._create_default_experiment(goal)

    async def _run_automated_experiment(self, experiment_id: str) -> Dict:
        """
        Execute automated experiment with real-time monitoring.
        """
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]
        plan = experiment[\'plan\']
        experiment[\'status\'] = ExperimentStatus.RUNNING

        results = []
        original_config = copy.deepcopy(self.config) # Store original config

        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(plan[\'parameters\'])
            total_runs = min(len(param_combinations), plan.get(\'iterations\', Config.MAX_EXPERIMENT_ITERATIONS))

            logging.info(f"Starting experiment {experiment_id} with {total_runs} runs")

            for i, param_values in enumerate(param_combinations[:total_runs]):
                start_time = time.time()

                try:
                    # Apply parameters to system
                    self._apply_experiment_parameters(param_values)

                    # Run test queries and measure metrics
                    metrics = self._measure_experiment_metrics(plan[\'metrics\'])

                    execution_time = time.time() - start_time

                    result = ExperimentResult(
                        parameter_values=param_values,
                        metrics=metrics,
                        execution_time=execution_time,
                        errors=[]
                    )
                    results.append(result)

                except Exception as e:
                    result = ExperimentResult(
                        parameter_values=param_values,
                        metrics={},
                        execution_time=time.time() - start_time,
                        errors=[str(e)]
                    )
                    results.append(result)

                # Restore original configuration after each run
                self.config = copy.deepcopy(original_config)

                # Progress update every 10% of runs
                if (i + 1) % max(1, total_runs // 10) == 0:
                    progress = (i + 1) / total_runs * 100
                    logging.info(f"Experiment {experiment_id} progress: {progress:.1f}%")

            # Analyze results
            analysis = await self._analyze_experiment_results(plan, results)

            experiment[\'status\'] = ExperimentStatus.COMPLETED
            experiment[\'completed_at\'] = time.time()
            experiment[\'results\'] = results
            experiment[\'analysis\'] = analysis

            # Store in history
            self.experiment_history.append(experiment)

            return {
                \'experiment_id\': experiment_id,
                \'plan\': plan,
                \'results\': results,
                \'analysis\': analysis,
                \'status\': \'completed\'
            }

        except Exception as e:
            # Ensure config is restored even if experiment fails
            self.config = original_config
            experiment[\'status\'] = ExperimentStatus.FAILED
            raise e
        finally:
            # Always restore original configuration
            self.config = original_config

    async def _analyze_experiment_results(self, plan: Dict, results: List[ExperimentResult]) -> str:
        """
        Use LLM to analyze experiment results and provide insights.
        """
        # Prepare results summary
        successful_results = [r for r in results if not r.errors]
        failed_results = [r for r in results if r.errors]

        if not successful_results:
            return "⚠️ All experiment runs failed. No meaningful analysis possible."

        # Find best and worst performing configurations
        primary_metric = plan[\'metrics\'][0] if plan[\'metrics\'] else \'overall_score\'

        if primary_metric in successful_results[0].metrics:
            best_result = max(successful_results, key=lambda r: r.metrics.get(primary_metric, 0))
            worst_result = min(successful_results, key=lambda r: r.metrics.get(primary_metric, 0))
        else:
            best_result = successful_results[0] # Fallback if metric not found
            worst_result = successful_results[-1]

        results_summary = {
            \'total_runs\': len(results),
            \'successful_runs\': len(successful_results),
            \'failed_runs\': len(failed_results),
            \'best_config\': best_result.parameter_values,
            \'best_metrics\': best_result.metrics,
            \'worst_config\': worst_result.parameter_values,
            \'worst_metrics\': worst_result.metrics,
            \'average_metrics\': self._calculate_average_metrics(successful_results)
        }

        analysis_prompt = f"""
        Analyze the following automated experiment results:

        **Experiment Goal**: {plan.get(\'hypothesis\', \'Improve system performance\')}
        **Parameters Tested**: {list(plan[\'parameters\'].keys())}
        **Metrics Measured**: {plan[\'metrics\]}

        **Results Summary**:
        {json.dumps(results_summary, indent=2)}

        **Key Questions to Address**:
        1. Which parameter values had the most positive impact?
        2. Are there any clear patterns or correlations?
        3. What are the recommended settings based on these results?
        4. What should be tested next?
        5. Are there any surprising or unexpected findings?

        Provide actionable insights and specific recommendations for the developer.
        Keep the analysis concise but comprehensive.
        """

        # Use smart LLM strategy for experiment analysis (high complexity)
        analysis_report, cost = await self._call_llm_with_strategy(analysis_prompt, task_complexity=0.9, max_tokens=1200, temperature=0.2)
        self.total_cost += cost
        return analysis_report

    def _validate_code_suggestion(self, suggestion_id: str) -> str:
        """
        Validate a code suggestion with automated checks.
        """
        if suggestion_id not in self.code_suggestions_cache:
            return "❌ Suggestion not found."

        suggestion = self.code_suggestions_cache[suggestion_id]
        code = suggestion[\'code\']

        validation_results = []

        # Syntax validation
        try:
            compile(code, \'<string>\', \'exec\')
            validation_results.append("✅ Syntax: Valid Python syntax")
        except SyntaxError as e:
            validation_results.append(f"❌ Syntax: {str(e)}")

        # Static analysis checks
        validation_results.extend(self._static_code_analysis(code))

        # Security checks
        validation_results.extend(self._security_code_analysis(code))

        # Performance considerations
        validation_results.extend(self._performance_code_analysis(code))

        return "\n".join(validation_results)

    def _static_code_analysis(self, code: str) -> List[str]:
        """
        Basic static analysis of code.
        """
        results = []

        # Check for common issues
        if \'import *\' in code:
            results.append("⚠️ Style: Avoid wildcard imports")

        if \'except:\' in code and \'except Exception:\' not in code:
            results.append("⚠️ Error Handling: Use specific exception types")

        if \'print(\' in code:
            results.append("ℹ️ Logging: Consider using logging instead of print")

        if code.count(\'\n\') > 50:
            results.append("ℹ️ Length: Function is quite long, consider breaking it down")

        return results

    def _security_code_analysis(self, code: str) -> List[str]:
        """
        Basic security analysis of code.
        """
        results = []

        dangerous_patterns = [
            (\'eval(\'', \'Dangerous: eval() can execute arbitrary code\'),
            (\'exec(\'', \'Dangerous: exec() can execute arbitrary code\'),
            (\'__import__\'', \'Caution: Dynamic imports can be risky\'),
            (\'subprocess\'', \'Caution: Subprocess calls need careful validation\'),
            (\'os.system\'', \'Dangerous: System calls can be exploited\')
        ]

        for pattern, warning in dangerous_patterns:
            if pattern in code:
                results.append(f"🔒 Security: {warning}")

        return results

    def _performance_code_analysis(self, code: str) -> List[str]:
        """
        Basic performance analysis of code.
        """
        results = []

        if \'for\' in code and \'in range(len(\' in code:
            results.append("⚡ Performance: Consider using enumerate() instead of range(len())")

        if code.count(\'+\') > 3 and \'str\' in code:
            results.append("⚡ Performance: Consider using join() for string concatenation")

        return results

    def _get_system_context(self) -> Dict:
        # Helper for getting context for prompts
        return {
            "active_components": ["TuberCore", "ChallengeAgent", "LLMInterface", "QuestionGenerator"],
            "recent_metrics": {"avg_latency": "150ms", "error_rate": "0.5%"},
            "config_summary": {"llm_provider": self.config.LLM_PROVIDER, "pruning_threshold": self.config.PRUNING_THRESHOLD}
        }

    def _get_metrics_history(self) -> Dict:
        # Helper for getting metrics history for prompts
        return {
            "recent_avg": {"accuracy": 0.85, "cost_per_query": 0.002},
            "trends": {"accuracy": "+5%", "cost_per_query": "-2%"},
            "bottlenecks": ["semantic_search_latency"]
        }

    def _apply_experiment_parameters(self, params: Dict):
        # Applies parameters to self.config for an experiment run
        for key, value in params.items():
            # Convert string values from LLM to appropriate types if necessary
            if key.upper() == \'LLM_TEMPERATURE\':
                setattr(self.config, key.upper(), float(value))
            elif key.upper() == \'LLM_MAX_TOKENS\':
                setattr(self.config, key.upper(), int(value))
            elif key.upper() == \'PRUNING_THRESHOLD\':
                setattr(self.config, key.upper(), float(value))
            elif hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        logging.info(f"Applied experiment parameters: {params}")

    def _measure_experiment_metrics(self, metrics_to_measure: List[str]) -> Dict:
        # Simulates running tests and measuring metrics
        # In a real system, this would trigger a benchmark suite
        time.sleep(2) # Simulate work
        measured_metrics = {}
        for metric in metrics_to_measure:
            if metric == "query_latency":
                measured_metrics[metric] = random.uniform(100, 500)
            elif metric == "accuracy_score":
                measured_metrics[metric] = random.uniform(0.7, 0.98)
            elif metric == "memory_usage":
                measured_metrics[metric] = random.uniform(500, 2000)
            elif metric == "cache_hit_rate":
                measured_metrics[metric] = random.uniform(0.5, 0.99)
            elif metric == "error_rate":
                measured_metrics[metric] = random.uniform(0.01, 0.1)
            elif metric == "overall_score":
                measured_metrics[metric] = random.uniform(0.6, 0.95)
            else:
                measured_metrics[metric] = random.random()
        return measured_metrics

    def _generate_parameter_combinations(self, params_to_test: Dict) -> List[Dict]:
        # Generates combinations of parameters for an experiment
        # This is a simplified version; libraries like scikit-learn have more advanced methods
        keys, values = zip(*[(k, v[\'values\']) for k, v in params_to_test.items()])
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        random.shuffle(combinations)
        return combinations

    def _validate_experiment_plan(self, plan: Dict) -> Dict:
        # Simple validation of the plan generated by LLM
        plan[\'iterations\'] = min(plan.get(\'iterations\', Config.MAX_EXPERIMENT_ITERATIONS), Config.MAX_EXPERIMENT_ITERATIONS)
        # Ensure metrics are valid
        valid_metrics = ["query_latency", "accuracy_score", "memory_usage", "cache_hit_rate", "error_rate", "overall_score"]
        plan[\'metrics\'] = [m for m in plan[\'metrics\'] if m in valid_metrics]
        if not plan[\'metrics\']:
            plan[\'metrics\'] = [\'overall_score\']

        # Ensure parameters are valid and values are lists
        for param_name, param_data in plan[\'parameters\'].items():
            if \'values\' not in param_data or not isinstance(param_data[\'values\'], list):
                param_data[\'values\'] = [] # Clear invalid values
            # Add more validation for parameter ranges if needed

        return plan

    def _create_default_experiment(self, goal: str) -> Dict:
        # Fallback if LLM fails to generate a valid plan
        return {
            "experiment_name": f"Default Experiment for {goal}",
            "hypothesis": "Test basic parameters to see impact.",
            "parameters": {
                "llm_temperature": {"values": [0.5, 0.8], "description": "LLM creativity"},
                "pruning_threshold": {"values": [0.1, 0.2], "description": "Tuber pruning aggressiveness"}
            },
            "metrics": ["query_latency", "accuracy_score"],
            "iterations": 4,
            "success_criteria": "Find settings that improve accuracy without significantly increasing latency.",
            "estimated_duration_minutes": 5
        }

    def _calculate_average_metrics(self, results: List[ExperimentResult]) -> Dict[str, float]:
        avg_metrics = defaultdict(float)
        if not results: return {}

        for result in results:
            for metric, value in result.metrics.items():
                avg_metrics[metric] += value

        for metric in avg_metrics:
            avg_metrics[metric] /= len(results)
        return dict(avg_metrics)

    def _format_experiment_plan(self, plan: Dict) -> str:
        formatted_plan = f"Experiment Name: {plan.get(\'experiment_name\', \'N/A\')}\nHypothesis: {plan.get(\'hypothesis\', \'N/A\')}\n"
        formatted_plan += "Parameters to Test:\n"
        for param, data in plan.get(\'parameters\', {}).items():
            formatted_plan += f"  - {param}: {data.get(\'values\', [])} ({data.get(\'description\', \'\')})\n"
        formatted_plan += f"Metrics to Measure: {plan.get(\'metrics\', [])}\n"
        formatted_plan += f"Iterations: {plan.get(\'iterations\', \'N/A\')}\n"
        formatted_plan += f"Success Criteria: {plan.get(\'success_criteria\', \'N/A\')}\n"
        formatted_plan += f"Estimated Duration: {plan.get(\'estimated_duration_minutes\', \'N/A\')} minutes\n"
        return formatted_plan

    def _format_experiment_results(self, results_data: Dict) -> str:
        formatted_results = f"\n--- Experiment Results for {results_data.get(\'experiment_id\', \'N/A\')} ---\n"
        formatted_results += f"Status: {results_data.get(\'status\', \'N/A\')}\n"
        formatted_results += f"Total Runs: {len(results_data.get(\'results\', []))}\n"

        analysis = results_data.get(\'analysis\', \'No analysis provided.\')
        formatted_results += f"\n**Analysis and Recommendations:**\n{analysis}\n"

        return formatted_results

    def _list_experiments(self) -> str:
        if not self.active_experiments and not self.experiment_history:
            return "No experiments proposed or run yet."

        report = "**Active Experiments:**\n"
        if self.active_experiments:
            for exp_id, exp_data in self.active_experiments.items():
                report += f"- ID: {exp_id}, Goal: {exp_data.get(\'goal\', \'N/A\')}, Status: {exp_data.get(\'status\', \'N/A\').value}\n"
        else:
            report += "  None\n"

        report += "\n**Completed Experiments:**\n"
        if self.experiment_history:
            for exp_data in self.experiment_history:
                report += f"- ID: {exp_data.get(\'experiment_id\', \'N/A\')}, Goal: {exp_data.get(\'goal\', \'N/A\')}, Status: {exp_data.get(\'status\', \'N/A\').value}, Completed: {time.ctime(exp_data.get(\'completed_at\', 0))}\n"
        else:
            report += "  None\n"
        return report

    def _analyze_system_health(self) -> str:
        # Placeholder for a more comprehensive health analysis
        status = self.get_system_status()
        health_report = f"System Health Check:\n"
        health_report += f"- Total Tubers: {status[\'total_tubers\]}\n"
        health_report += f"- LLM Calls: {status[\'total_llm_calls\]} (Estimated Cost: ${status[\'total_estimated_cost\]:.4f})\n"
        health_report += f"- Active Experiments: {status[\'active_experiments\]}\n"
        health_report += f"- Completed Experiments: {status[\'completed_experiments\]}\n"
        health_report += f"- Failed Questions in Memory: {self.qmem.get_memory_size()}\n"

        # Add some simulated health checks
        if status[\'total_tubers\] > Config.MAX_TUBERS * 0.8:
            health_report += "⚠️ Warning: Tuber network approaching max capacity. Consider reviewing pruning strategy.\n"
        if status[\'total_estimated_cost\] > 10.0:
            health_report += "💰 Alert: LLM costs are accumulating. Monitor usage.\n"
        if self.qmem.get_memory_size() > 20:
            health_report += "🤔 Info: The system is asking many questions that are not yielding useful answers. Consider reviewing the question generation or classification logic.\n"

        return health_report

    def get_system_status(self) -> Dict:
        """Returns a summary of the system\'s current state."""
        return {
            "total_tubers": len(self.tuber_model.graph.nodes) if self.tuber_model.graph else 0,
            "root_tuber_id": self.tuber_model.root_id,
            "active_knowledge_triangle": self.tuber_model.active_knowledge_triangle,
            "total_llm_calls": self.llm_interface.call_count,
            "total_estimated_cost": self.total_cost,
            "active_experiments": len(self.active_experiments),
            "completed_experiments": len(self.experiment_history),
            "failed_questions_in_memory": self.qmem.get_memory_size()
        }

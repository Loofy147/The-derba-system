from typing import Dict, Any, List

from llm_interface import LLMInterface


class QuestionQualityClassifier:
    def __init__(self, llm_interface: LLMInterface, threshold: float = 0.7):
        self.llm_interface = llm_interface
        self.threshold = threshold

    async def score(self, question: str) -> float:
        """
        يصنف وضوح ومناسبة الأسئلة قبل عرضها.
        هذه دالة وهمية تحتاج إلى منطق حقيقي (ربما باستخدام LLM).
        """
        prompt = (
            f"Evaluate the quality of the following question for a self-improving AI system on a scale from 0.0 to 1.0. "
            f"Consider its clarity, relevance, and potential to reveal deep insights. "
            f"A good question is specific, profound, and actionable."
            f"\n\nQuestion: \"{question}\""
            f"\n\nRespond with only a single floating-point number (e.g., 0.85)."
        )

        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=10, temperature=0.2)

        try:
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except (ValueError, AttributeError):
            # Fallback to heuristic if LLM response is invalid
            return 0.5

    async def is_useful(self, answer: str, question: str) -> bool:
        """
        Determines if an answer is useful in the context of the question using an LLM.
        """
        if not answer or len(answer.strip()) < 10:
            return False

        prompt = (
            f"Given the question: \"{question}\"\n"
            f"And the answer: \"{answer}\"\n"
            f"Is this a useful and informative answer? "
            f"A useful answer is one that is relevant, specific, and provides actionable information or new insights. "
            f"Respond with only 'Yes' or 'No'."
        )

        response, _ = await self.llm_interface.generate_text(prompt, max_tokens=5, temperature=0.1)

        return 'yes' in response.lower()

    async def rephrase(self, question: str) -> str:
        """
        Rephrases a weak or unclear question using an LLM.
        """
        prompt = (
            f"The following question is unclear or not impactful. Rephrase it to be more specific, profound, and actionable "
            f"for a self-improving AI system."
            f"\n\nOriginal question: \"{question}\""
            f"\n\nRephrased question:"
        )

        rephrased_question, _ = await self.llm_interface.generate_text(prompt, max_tokens=100, temperature=0.7)
        return rephrased_question.strip()

from typing import List, Dict, Any
import time

from llm_interface import LLInterface


class FailedQuestionMemory:
    def __init__(self, llm_interface: LLMInterface):
        self.failed_questions: List[Dict[str, Any]] = []
        self.llm_interface = llm_interface

    def store(self, question: str, reason: str, context: Dict[str, Any] = None):
        """
        يخزن الأسئلة التي لم تحصل على إجابات مفيدة.
        """
        entry = {
            "question": question,
            "reason": reason, # e.g., 'low_quality_score', 'unuseful_answer'
            "context": context if context else {},
            "timestamp": time.time()
        }
        self.failed_questions.append(entry)
        print(f"[FailedQuestionMemory] Stored failed question: \"{question[:50]}...\"")

    def retrieve_for_retraining(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        يسترجع الأسئلة الفاشلة لاستخدامها كنماذج سلبية لإعادة التدريب الداخلي.
        """
        # يمكن إضافة منطق أكثر تعقيدًا هنا (مثل استرجاع الأسئلة الأقدم، أو الأكثر تكرارًا)
        return self.failed_questions[-limit:]

    def clear_memory(self):
        """
        يمسح الذاكرة.
        """
        self.failed_questions = []
        print("[FailedQuestionMemory] Memory cleared.")

    def get_memory_size(self) -> int:
        return len(self.failed_questions)

    async def analyze_failures(self) -> str:
        """
        Analyzes the stored failed questions to identify patterns and suggest improvements.
        """
        if len(self.failed_questions) < 5:
            return "Not enough failed questions to perform a meaningful analysis."

        # Prepare a summary of failed questions for the LLM
        summary = "\n".join([
            f"- Question: \"{q['question']}\", Reason: {q.get('reason', 'N/A')}"
            for q in self.failed_questions[-20:] # Analyze the last 20 failures
        ])

        prompt = (
            f"Analyze the following list of failed questions from a self-improving AI system. "
            f"Identify common patterns in why these questions are failing. "
            f"Are the questions too vague, too broad, out of scope, or is there another issue? "
            f"Provide a concise summary of the main failure pattern and suggest a concrete improvement for the question generation process."
            f"\n\nFailed Questions:\n{summary}"
            f"\n\nAnalysis and Suggestion:"
        )

        analysis, _ = await self.llm_interface.generate_text(prompt, max_tokens=300, temperature=0.6)
        return analysis.strip()

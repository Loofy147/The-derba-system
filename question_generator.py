from typing import Dict, Any, List
import random

import json
from llm_interface import LLMInterface


class QuestionGenerator:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    async def generate(self, context_data: Dict[str, Any], gaps: List[str]) -> str:
        """
        يولد سؤالاً بناءً على سياق البيانات والفجوات المعرفية.
        يعتمد على نموذج فرعي مدرَّب على توليد أسئلة من إجابات (reverse QA datasets).
        """
        # هذا الجزء سيستخدم LLMInterface لإنشاء السؤال
        # يمكن استخدام Prompt Engineering هنا لتوجيه LLM

        if not gaps:
            return "What is the most important area to explore next for maximum system improvement?"

        prompt = (
            "Given the following system context and identified knowledge gaps, generate one concise, high-impact question. "
            "The question should be profound and aim to uncover fundamental insights for improving the AI system."
            f"\n\nSystem Context: {json.dumps(context_data, indent=2)}"
            f"\nIdentified Knowledge Gaps: {'; '.join(gaps)}"
            "\n\nGenerate the single most important question to ask right now:"
        )

        question, _ = await self.llm_interface.generate_text(prompt, max_tokens=100, temperature=0.8)
        return question.strip()

    def micro_tune_on_qa(self, question: str, answer: str):
        """
        Placeholder for micro-tuning the question generator model on a question-answer pair.
        This would require a trainable LLM and a more complex training pipeline.
        """
        # In a real implementation, this would trigger a fine-tuning job.
        # For now, we just log the intent.
        logging.info(f"Micro-tuning opportunity identified for Q: '{question}'")
        pass

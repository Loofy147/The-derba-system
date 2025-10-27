AKT_UPDATE_PROMPT = """
You are managing an Active Knowledge Triangle (AKT) for an AI system. The AKT should represent the system's current core focus.
Current AKT members: {current_akt_contents}
New tuber proposed for inclusion: {new_tuber_content}

Based on the goal of maintaining a balanced, relevant, and dynamic core focus, which of the current AKT members (if any) should be replaced by the new tuber?
Consider semantic relevance, novelty, potential impact, and current system needs.

Respond with ONLY the ID of the tuber to be replaced, or 'None' if no replacement is necessary.
"""

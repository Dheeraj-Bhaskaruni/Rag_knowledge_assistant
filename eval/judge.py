from openai import OpenAI
import os
import json

JUDGE_PROMPT = """
You are an impartial judge evaluating a RAG system.
Given the QUESTION, CONTEXT, and ANSWER, evaluate:
1. Grounding: Is the answer fully supported by the context? (1-5)
2. Correctness: Does the answer answer the question? (1-5)

Output JSON: {"grounding": int, "correctness": int, "reasoning": "string"}
"""

class Judge:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def evaluate(self, question: str, context: str, answer: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"QUESTION: {question}\nCONTEXT: {context}\nANSWER: {answer}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Judge error: {e}")
            return {"grounding": 0, "correctness": 0, "reasoning": str(e)}

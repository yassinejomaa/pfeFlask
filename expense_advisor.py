import requests
import json
import time

class ExpenseAdvisor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model_name = "phi3"
    
    def build_messages(self, expenses):
        messages = [
            {"role": "system", "content": "Here is a breakdown of my current expenses. Based on this, can you give me recommendations to help me save money."},
            {"role": "user", "content": (
                f"My recent expenses are:\n" + "\n".join(f"- {expense}" for expense in expenses) +
                "\nPlease give me customized financial advice based on this information and never mention about any budgeting apps and the response in the form of html started from the body balise and give me only recomendation dont generate anything else in your rsponse and i want the response to be well structred because i will visualise it directly on my website write the expense then the recomandation of it and make the same style for all expenses"
)}]
        return messages
    
    def generate_advice_stream(self, expenses):
        messages = self.build_messages(expenses)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

        def stream_response():
            inside_think_block = False
            first_chunk_skipped = False  # ✅ Flag to fix the first bad piece

            with requests.post(self.ollama_url, json=payload, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                if isinstance(line, bytes):
                                    line = line.decode('utf-8')
                                chunk = json.loads(line)

                                content_piece = chunk.get('message', {}).get('content', '')
                                if content_piece:
                                    if "<think>" in content_piece:
                                        inside_think_block = True
                                        continue
                                    if "</think>" in content_piece:
                                        inside_think_block = False
                                        continue
                                    if inside_think_block:
                                        continue

                                    # ✅ Remove the first ""html manually
                                    if not first_chunk_skipped:
                                        content_piece = content_piece.lstrip()  # Remove extra spaces just in case
                                        if content_piece.startswith('```html'):
                                            content_piece = content_piece.replace('```html', '', 1).lstrip()
                                        first_chunk_skipped = True

                                    yield content_piece
                            except Exception as e:
                                yield f"\n[Streaming Error]: {str(e)}\n"
                else:
                    yield f"Error: {response.status_code} - {response.text}"

        return stream_response()
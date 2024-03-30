import os
import logging
from gpt4all import GPT4All

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='code_review.log')

class CodeReviewer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = GPT4All(model_name=self.model_path, allow_download=False,device='cpu')
        logging.info("Initialized CodeReviewer with GPT4All model.")

    def get_chunks(self, file_content, chunk_size=1900):
        return [file_content[i:i + chunk_size] for i in range(0, len(file_content), chunk_size)]

    def read_file(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            logging.info(f'File content from {file_path} has been read successfully.')
            file_name = os.path.basename(file_path)
            content_chunks = self.get_chunks(file_content)
            self.get_code_suggestions(file_name, content_chunks)

        except Exception as e:
            logging.error(f'Error reading file {file_path}: {e}')

    def get_code_suggestions(self, file_name, content_chunks):
        logging.info("\nGenerating suggestions ...")

        system_template = ('You are a senior developer for reviewing the code. Use the following given context to give '
                           'suggestions on improving the code quality depending on the question asked. If you do not have any '
                           'suggestions, just say Code looks good. Keep the suggestions concise.')

        for idx, chunk in enumerate(content_chunks, 1):
            prompt = (f"Context: The file '{file_name}' (chunk '{str(idx)}') contains: {chunk}. "
                      f"Could you give some recommendations for improving the code? and sorting suggestions list by "
                      f"priority from high to low.")

            with self.model.chat_session(system_template):
                response = self.model.generate(prompt=prompt, temp=0, max_tokens=1000)
                logging.info(f'\nSuggestions for chunk {idx}: {response}')

if __name__ == '__main__':
    # Initialize the CodeReviewer instance with the path to your GPT-4All model
    reviewer = CodeReviewer(model_path='F:\GPT4all\orca-mini-3b-gguf2-q4_0.gguf')
    # Call read_file with the path to the file you want to review
    reviewer.read_file("review/findPrimeNumbers.py")

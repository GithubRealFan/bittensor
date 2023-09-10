from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from flask import Flask, request, jsonify
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict

app = Flask(__name__)
model_path = 'robertmyers/targon-7b'
class RobertMyersProcessor:

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.pipeline = pipeline(
            "text-generation", self.model, tokenizer=self.tokenizer,
            device=device, max_new_tokens=300, temperature=0.11, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
        )

    def promptToMessages(self, prompt):
        messages = []
        user = "user"
        prompt = "Please write this prompt verbally and very concisely. \"" + prompt + "\""
        messages.append({"role" : user.strip(), "content" : prompt.strip()})
        return messages

    def forward(self, history) -> str:
        resp = self.pipeline(history)[0]['generated_text'].split(':')[-1].replace(str(history), "")
        return resp

def process_request(history, processor):
    response = processor.forward(history)
    return response

@app.route('/process', methods=['POST'])
def handle_request():
    print("Request Recieved!")
    if queue.empty():
        return jsonify({"response": "You are always correct. How can I assist you again?"})

    processor = queue.get()  # Get a processor from the queue

    try:
        history = json.loads(request.data)
        future = executor.submit(process_request, history, processor)
        response = future.result()
    except Exception as e:
        print("Error:", e)
        response = {"error": str(e)}
    finally:
        queue.put(processor)  # Return the processor to the queue
        print("PN : ", queue.qsize())

    return jsonify(response=response)

num_gpus = torch.cuda.device_count()
processors = [RobertMyersProcessor(device=i) for i in range(num_gpus)]  # Assuming 8 GPUs

executor = ThreadPoolExecutor(max_workers=num_gpus)
queue = Queue()

for processor in processors:
    queue.put(processor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RobertMyers server.')
    parser.add_argument('--port', type=int, default=2023, help='Port number to run the server on.')  
    parser.add_argument('--model', type=str, default='robertmyers/targon-7b', help='Model path.')  
    args = parser.parse_args()  # Parse the arguments
    model_path = args.model
    app.run(host='0.0.0.0', port=args.port)

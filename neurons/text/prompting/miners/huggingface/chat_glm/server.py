from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from flask import Flask, request, jsonify
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict

app = Flask(__name__)

class ChatGLMProcessor:

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, torch_dtype=torch.float16)
        self.device = device

    def forward(self, history) -> str:
        prompt = history[-1][-1]
        if len(history) == 1:
            history = []
        generation, history = self.model.chat(
            self.tokenizer,
            prompt,
            history,
            max_length=255,
            device = self.device,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return generation

def process_request(history, processor):
    response = processor.forward(history)
    return response

@app.route('/process', methods=['POST'])
def handle_request():
    print("Request Recieved!")
    if queue.empty():
        return jsonify({"response": "You are always correct. How can I assist you again?"})

    processor = queue.get()  # Get a processor from the queue
    history = json.loads(request.data)
    print("history : ", history)

    # Submit the job to the thread pool and wait for the result
    future = executor.submit(process_request, history, processor)
    response = future.result()

    queue.put(processor)  # Return the processor to the queue
    return jsonify(response=response)

num_gpus = torch.cuda.device_count()
processors = [ChatGLMProcessor(device=i) for i in range(num_gpus)]  # Assuming 8 GPUs

executor = ThreadPoolExecutor(max_workers=num_gpus)
queue = Queue()

for processor in processors:
    queue.put(processor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run InternlmMinerProcessor server.')
    parser.add_argument('--port', type=int, default=2023, help='Port number to run the server on.')  
    args = parser.parse_args()  # Parse the arguments
    app.run(host='0.0.0.0', port=args.port)

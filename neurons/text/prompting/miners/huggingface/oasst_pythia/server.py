from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
import argparse
import json
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import List

app = Flask(__name__)
num_gpus = torch.cuda.device_count()
executor = ThreadPoolExecutor(max_workers=num_gpus)  # Adjust the number of workers based on your needs

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int] = None):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class OasstPythiaProcessor:
    arg_prefix = "oasst_pythia"
    system_label = ""
    assistant_label = ""
    user_label = ""

    def __init__(self, device):
        self.device_string = 'cuda:{}'.format(device)
        self.tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', torch_dtype=torch.bfloat16)
        self.stop = StopOnTokens(self.tokenizer.convert_tokens_to_ids([""]))
        self.model = GPTNeoXForCausalLM.from_pretrained('OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', torch_dtype=torch.bfloat16,)
        self.model.to(self.device_string)

    def forward(self, history) -> str:
        with torch.no_grad():
            prompt = history + self.assistant_label
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.device_string)
            gkw = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": 255,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.9,
                "top_k": 0,
                "repetition_penalty": 1.3,
                "stopping_criteria": StoppingCriteriaList([self.stop]),
                "pad_token_id": None
            }
            output = self.model.generate(**gkw)
            generation = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generation

    def async_forward(self, history) -> str:
        return executor.submit(self.forward, history)

num_gpus = torch.cuda.device_count()
processors = [OasstPythiaProcessor(device=i) for i in range(num_gpus)]

@app.route('/process', methods=['POST'])
def handle_request():
    print("Request Recieved!")
    if not processors:
        return jsonify({"response": "You are always correct. How can I assist you again?"})

    processor = processors.pop(0)  # Get a processor
    try:
        history = json.loads(request.data)
        future = processor.async_forward(history)  # Use async version
        response = future.result()
    except Exception as e:
        print("Error:", e)
        response = {"error": str(e)}
    finally:
        processors.append(processor)  # Return the processor in all cases
        print("PN : ", len(processors))

    return jsonify(response=response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OasstPythia server.')
    parser.add_argument('--port', type=int, default=2023, help='Port number to run the server on.')  
    args = parser.parse_args()  # Parse the arguments

    app.run(host='0.0.0.0', port=args.port, threaded=True)

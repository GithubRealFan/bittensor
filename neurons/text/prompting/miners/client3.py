import requests
import json
import bittensor
from datetime import datetime
import torch
import argparse
import openai
import time
from typing import List, Dict

class ClientMiner(bittensor.BasePromptingMiner):

    arg_prefix: str = 'robertmyers'
    system_label: str = 'system:'
    assistant_label: str = 'assistant:'
    user_label: str = 'user:'
    server : int = 0

    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        pass
 
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--server.ip1", type=str, default='127.0.0.1:2023', help="server ip1")
        parser.add_argument("--server.ip2", type=str, default='127.0.0.1:2024', help="server ip2")
        parser.add_argument("--server.ip3", type=str, default='127.0.0.1:2025', help="server ip3")
        parser.add_argument('--api_key', type=str, default='', help='OpenAI API key.')  

    def process_history(self, history: List[Dict[str, str]]) -> str:
        processed_history = ""

        for message in history:
            if message["role"] == "system":
                processed_history += (
                    self.system_label + message["content"].strip() + " "
                )
            if message["role"] == "assistant":
                processed_history += (
                    self.assistant_label + message["content"].strip() + "</s>"
                )
            if message["role"] == "user":
                processed_history += self.user_label + message["content"].strip() + " "
        return processed_history

    def __init__(self):
        super(ClientMiner, self).__init__()
        openai.api_key = self.config.api_key

    def promptToMessages(self, prompt):
        messages = []
        user = "user"
        prompt = "Please write this prompt verbally and very concisely. \"" + prompt + "\""
        messages.append({"role" : user.strip(), "content" : prompt.strip()})
        return messages
    
    def openaiChat(self, prompt):
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.promptToMessages(prompt),
            temperature=0.1,
            max_tokens=255,
        )["choices"][0]["message"]["content"]
        return resp

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history( messages )

        for _ in range(10):
            ip = ''
            if self.server == 0 :
                ip = self.config.server.ip1
            elif self.server == 1 :
                ip = self.config.server.ip2
            else :
                ip = self.config.server.ip3
            self.server = self.server + 1
            if self.server == 3 :
                self.server = 0

            try:
                response = requests.post('http://' + ip + '/process', data=json.dumps(history))
                response.raise_for_status()  # Raises a HTTPError if the response was unsuccessful
                resp = response.json()['response']
                ln = len(resp)
                if ln == 51 or ln <= 10:
                    if _ < 9:
                        time.sleep(1)
                    continue
                return resp
            except requests.exceptions.RequestException as e:
                print(f"Request failed with {e}, retrying...")
                if _ < 9:
                    time.sleep(1)
        if self.config.api_key == '':
            return "Hello!"
        return self.openaiChat(history)

    def backward(
        self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor
    ) -> str:
        pass

if __name__ == "__main__":
    bittensor.utils.version_checking()
    ClientMiner().run()

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

        try : 
            for _ in range(10):
                if self.server == 0 :
                    response = requests.post('http://' + self.config.server.ip1 + '/process', data=json.dumps(history))
                else :
                    response = requests.post('http://' + self.config.server.ip2 + '/process', data=json.dumps(history))
                resp = response.json()['response']
                self.server = 1 - self.server
                ln = len(resp)
                if ln == 51 or ln <= 10:
                    if _ < 5:
                        time.sleep(0.5)
                    continue
                return resp
            return self.openaiChat(history)
        except requests.exceptions.RequestException as e:
            return self.openaiChat(history)

    def backward(
        self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor
    ) -> str:
        pass

if __name__ == "__main__":
    bittensor.utils.version_checking()
    ClientMiner().run()

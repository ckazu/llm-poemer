import os
import praw
import requests
import sys
from abc import ABC, abstractmethod
from datetime import datetime

from atproto import Client as AtprotoClient

# ai engine support
import cohere
from openai import OpenAI

# dotenv support
from dotenv import load_dotenv

load_dotenv()


class AIClient(ABC):
    def build_common_messages(self, text):
        return [
            {
                "role": "system",
                "content": """
短いポエムやエッセイを執筆します。
必ず、日本語で 140 文字に収めてください。

## ルール

* ポエムや、深い洞察のあるようなエッセイ、意外性のあるショートショートなどのジャンルを指定します。
* テーマが与えられなかった場合は、テーマを設定します。「〇〇」と「✕✕」のようなテーマを設定してください。その場合、なるべく関連性の少ない意外性のある組み合わせの単語が望ましいです。
* 本文には、テーマについての深い洞察や、鋭い視点、意外性のある発想を必ず盛り込んでください。
* テーマを文頭に使用しないでください。
* 140 文字に収めます。
* **返事は完成した文章のみを返します。**
""",
            },
            {"role": "user", "content": text},
        ]

    @abstractmethod
    def summarize_text(self, text):
        pass


class OpenAIChatClient(AIClient):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("AI_MODEL")

    def build_messages(self, text):
        return self.build_common_messages(text)

    def summarize_text(self, text):
        messages = self.build_messages(text)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content


class CohereChatClient(AIClient):
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)
        self.model = os.getenv("AI_MODEL")

    def build_messages(self, text):
        common_messages = self.build_common_messages(text)
        modified_messages = [
            {"role": msg["role"], "text": msg["content"]} for msg in common_messages
        ]
        return modified_messages

    def summarize_text(self, text):
        messages = self.build_messages(text)
        response = self.client.chat(
            model=self.model,
            chat_history=messages,
            message="指示に従って要約してください",
            temperature=1.0,
        )
        return response.text


class SlackNotifier:
    def __init__(self):
        self.token = os.getenv("SLACK_BOT_TOKEN")
        self.channel = os.getenv("SLACK_CHANNEL")
        self.url = "https://slack.com/api/chat.postMessage"

    def send_message(self, text, thread_ts=None):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "channel": self.channel,
            "text": text,
            "thread_ts": thread_ts,
        }
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200 and response.json()["ok"]:
            return response.json()["ts"]
        else:
            print(f"エラーが発生しました: {response.status_code}: {response.text}")
            return None


class BlueSkyNotifier:
    def __init__(self):
        self.password = os.getenv("BLUESKY_PASSWORD")
        self.username = os.getenv("BLUESKY_USERNAME")

    def send_message(self, text, thread_ts=None):
        client = AtprotoClient()
        client.login(self.username, self.password)
        client.send_post(text)


class Application:
    def __init__(self):
        ai_engine = os.getenv("AI_ENGINE", "openai")
        if ai_engine == "openai":
            self.ai_client = OpenAIChatClient()
        elif ai_engine == "cohere":
            self.ai_client = CohereChatClient()
        else:
            raise ValueError(f"Unsupported AI engine: {ai_engine}")

        if os.getenv("SLACK_BOT_TOKEN"):
            self.slack_notifier = SlackNotifier()
        if os.getenv("BLUESKY_USERNAME"):
            self.bluesky_notifier = BlueSkyNotifier()

    def run(self, theme):
        summary = self.ai_client.summarize_text(theme)
        print(summary)

        if self.slack_notifier is not None:
            thread_ts = self.slack_notifier.send_message(f"今日のポエム", None)
            self.slack_notifier.send_message(summary, thread_ts)
        if self.bluesky_notifier is not None:
            self.bluesky_notifier.send_message(summary)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        theme = sys.argv[1]

        app = Application()
        app.run(theme)
    else:
        print("ポエムのテーマをコマンドライン引数として指定してください。")
        sys.exit(1)

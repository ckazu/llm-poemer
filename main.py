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


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

    def get_hot_posts_with_comments(self, subreddit_name, limit=3, time_filter="week"):
        subreddit = self.reddit.subreddit(subreddit_name)
        best_posts = list(subreddit.top(limit=limit, time_filter=time_filter))
        all_posts_text = ""
        for post in best_posts:
            post_date = datetime.utcfromtimestamp(post.created_utc)
            post_text = f"タイトル: {post.title}\nURL: {post.url}\n投稿日時: {post_date:%Y/%m/%d %H:%M:%S}\n"
            post_text += f"スコア: {post.score}\nコメント数: {post.num_comments}\n本文:\n{post.selftext}\n"
            comments = post.comments
            comment_list = [
                comment.body
                for comment in comments
                if isinstance(comment, praw.models.Comment)
            ]
            post_text += "コメントリスト:\n" + "\n".join(comment_list)
            all_posts_text += post_text + "\n\n"
        return all_posts_text


class AIClient(ABC):
    def build_common_messages(self, prompt):
        return prompt

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
            message="指示に従ってください",
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
        self.reddit_client = RedditClient()

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

    def run(self, subreddit_name, limit):
        # 1. Reddit の本文からテーマを抽出する
        all_posts_text = self.reddit_client.get_hot_posts_with_comments(
            subreddit_name, limit
        )
        print(all_posts_text)
        prompt = self.build_theme_prompt(all_posts_text)

        theme = self.ai_client.summarize_text(prompt)
        print(theme)

        # 2. テーマからポエムを生成する
        prompt = self.build_poem_prompt(theme)
        poem = self.ai_client.summarize_text(prompt)
        print(poem)

        if self.slack_notifier is not None:
            thread_ts = self.slack_notifier.send_message(f"今日のポエム", None)
            self.slack_notifier.send_message(poem, thread_ts)
        if self.bluesky_notifier is not None:
            if len(poem) > 140:
                print("140文字を超えるためスキップ")
            else:
                text = f"{poem}\n#ポエム #peom"
                self.bluesky_notifier.send_message(text)

    def build_theme_prompt(self, text):
        return [
            {
                "role": "system",
                "content": """
ポエムを作成するためのテーマを抽出します。
次の文章からポエムのテーマとなりそうな単語やフレーズを設定してください。
テーマは、単語一語や、フレーズ、あるいは、「単語A」と「単語B」のような組み合わせでも構いません。
技術的なワードや専門用語、固有名詞などは優先してテーマに含めてください。
テーマは一つだけ選定してください。
""",
            },
            {"role": "user", "content": text},
        ]

    def build_poem_prompt(self, theme):
        return [
            {
                "role": "system",
                "content": """
短いポエムを執筆します。
必ず、日本語で 140 文字に収めてください。

## ルール

* テーマが与えられなかった場合は、テーマを設定します。「〇〇」と「✕✕」のようなテーマを設定してください。その場合、なるべく関連性の少ない意外性のある組み合わせの単語が望ましいです。
* 長い文章が与えられた場合は、そこからふさわしいテーマを抽出します。
* ポエム本文には、テーマについての深い洞察や、鋭い視点、意外性のある発想を必ず盛り込んでください。
* テーマを文頭に使用しないでください。
* 140 文字に収めます。
* **返事は完成した文章のみを返します。**
* **テーマやポエムのタイトルは絶対に返答に含めません**
""",
            },
            {"role": "user", "content": theme},
        ]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        theme = sys.argv[1]
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        app = Application()
        app.run(theme, limit)
    else:
        print("ポエムのテーマをコマンドライン引数として指定してください。")
        sys.exit(1)

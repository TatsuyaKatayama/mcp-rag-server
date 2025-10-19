"""
エンベディング生成モジュール

テキストからエンベディングを生成します。
HuggingFace, OpenAI, Google, Anthropicのモデルをサポートします。
"""

import logging
import os
from typing import List, Literal

from dotenv import load_dotenv

# .envの読み込み
load_dotenv()

# Provider type
Provider = Literal["huggingface", "openai", "google", "anthropic"]


class EmbeddingGenerator:
    """
    エンベディング生成クラス

    テキストからエンベディングを生成します。
    HuggingFace, OpenAI, Google, Anthropicのモデルをサポートします。

    Attributes:
        model: 各プロバイダーのモデル/クライアント
        logger: ロガー
        provider: プロバイダー名
    """

    def __init__(self, model_name: str = None):
        """
        EmbeddingGeneratorのコンストラクタ

        Args:
            model_name: 使用するモデル名（.env優先）
        """
        # .envから設定を取得
        self.model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.prefix_query = os.getenv("EMBEDDING_PREFIX_QUERY", "")
        self.prefix_embedding = os.getenv("EMBEDDING_PREFIX_EMBEDDING", "")
        self.provider: Provider = "huggingface"
        self.model = None

        # ロガーの設定
        self.logger = logging.getLogger("embedding_generator")
        self.logger.setLevel(logging.INFO)

        # プロバイダーを特定
        if self.model_name.startswith("openai/"):
            self.provider = "openai"
            self.model_name = self.model_name.replace("openai/", "")
        elif self.model_name.startswith("google/"):
            self.provider = "google"
            self.model_name = self.model_name.replace("google/", "")
        elif self.model_name.startswith("anthropic/"):
            self.provider = "anthropic"
            self.model_name = self.model_name.replace("anthropic/", "")

        # モデルの読み込み
        self.logger.info(f"プロバイダー '{self.provider}' のモデル '{self.model_name}' を読み込んでいます...")
        try:
            if self.provider == "huggingface":
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name)
            elif self.provider == "openai":
                from openai import OpenAI

                self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            elif self.provider == "google":
                import google.generativeai as genai

                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.model = "models/" + self.model_name
            elif self.provider == "anthropic":
                self.logger.warning("Anthropicは現在Embedding APIをサポートしていません。")

            self.logger.info(f"モデル '{self.model_name}' を読み込みました")
        except Exception as e:
            self.logger.error(f"モデル '{self.model_name}' の読み込みに失敗しました: {str(e)}")
            raise

    def _add_prefix(self, text: str, prefix: str) -> str:
        """
        テキストに適切なプレフィックスを追加する

        Args:
            text: 元のテキスト
            prefix: 追加するプレフィックス

        Returns:
            プレフィックス付きのテキスト
        """
        if not prefix or self.provider != "huggingface":
            return text

        # プレフィックスが既に含まれているかチェック（大文字小文字を区別）
        if text.startswith(prefix):
            return text

        return f"{prefix}{text}"

    def generate_embedding(self, text: str) -> List[float]:
        """
        テキストからエンベディングを生成します。

        Args:
            text: エンベディングを生成するテキスト

        Returns:
            エンベディング（浮動小数点数のリスト）
        """
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストからエンベディングを生成します。

        Args:
            texts: エンベディングを生成するテキストのリスト

        Returns:
            エンベディングのリスト
        """
        if not texts:
            self.logger.warning("空のテキストリストからエンベディングを生成しようとしています")
            return []

        try:
            if self.provider == "huggingface":
                processed_texts = [self._add_prefix(text, self.prefix_embedding) for text in texts]
                embeddings = self.model.encode(processed_texts)
                embeddings_list = embeddings.tolist()
            elif self.provider == "openai":
                response = self.model.embeddings.create(input=texts, model=self.model_name)
                embeddings_list = [item.embedding for item in response.data]
            elif self.provider == "google":
                import google.generativeai as genai

                result = genai.embed_content(
                    model=self.model,
                    content=texts,
                    task_type="retrieval_document",
                )
                embeddings_list = result["embedding"]
            elif self.provider == "anthropic":
                self.logger.warning("AnthropicはEmbedding APIをサポートしていないため、空のリストを返します。")
                embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
                embeddings_list = [[0.0] * embedding_dim for _ in texts]

            self.logger.info(f"{len(texts)} 個のテキストのエンベディングを生成しました")
            return embeddings_list
        except Exception as e:
            self.logger.error(f"エンベディングの生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_search_embedding(self, query: str) -> List[float]:
        """
        検索クエリからエンベディングを生成します。

        Args:
            query: 検索クエリ

        Returns:
            エンベディング（浮動小数点数のリスト）
        """
        if not query:
            self.logger.warning("空のクエリからエンベディングを生成しようとしています")
            return []

        try:
            if self.provider == "huggingface":
                processed_query = self._add_prefix(query, self.prefix_query)
                embedding = self.model.encode(processed_query)
                embedding_list = embedding.tolist()
            elif self.provider == "openai":
                response = self.model.embeddings.create(input=[query], model=self.model_name)
                embedding_list = response.data[0].embedding
            elif self.provider == "google":
                import google.generativeai as genai

                result = genai.embed_content(
                    model=self.model,
                    content=query,
                    task_type="retrieval_query",
                )
                embedding_list = result["embedding"]
            elif self.provider == "anthropic":
                self.logger.warning("AnthropicはEmbedding APIをサポートしていないため、空のリストを返します。")
                embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
                embedding_list = [0.0] * embedding_dim

            self.logger.debug(f"クエリ '{query}' のエンベディングを生成しました")
            return embedding_list
        except Exception as e:
            self.logger.error(f"クエリエンベディングの生成中にエラーが発生しました: {str(e)}")
            raise
from newsapi import NewsApiClient
from ict_stock_trader.config.settings import NEWS_API_KEY

class NewsClient:
    def __init__(self):
        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY is not set in the configuration.")
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    def get_top_headlines(self, q: str, language: str = 'en', category: str = 'business'):
        """
        Fetches top business headlines for a given query.
        """
        try:
            return self.newsapi.get_top_headlines(q=q, language=language, category=category)
        except Exception as e:
            print(f"Error fetching news for '{q}': {e}")
            return None

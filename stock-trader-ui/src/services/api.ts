import axios from 'axios';
import {
  StockDataResponse,
  StockAnalysisResponse,
  WatchlistResponse,
  MarketOverview,
  APIError,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError: APIError = {
      detail: error.response?.data?.detail || error.message || 'An error occurred',
      status: error.response?.status,
    };
    return Promise.reject(apiError);
  }
);

export const stockAPI = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get stock data
  async getStockData(
    symbol: string,
    timeframe: string = '1d',
    period: string = '5d',
    includeIndicators: boolean = true
  ): Promise<StockDataResponse> {
    const response = await api.get(`/api/v1/stocks/${symbol}/data`, {
      params: {
        timeframe,
        period,
        include_indicators: includeIndicators,
      },
    });
    return response.data;
  },

  // Get multiple stocks data
  async getMultipleStocks(
    symbols: string[],
    timeframe: string = '1d',
    includeFundamentals: boolean = true
  ): Promise<any> {
    const response = await api.get('/api/v1/stocks/multiple', {
      params: {
        symbols: symbols.join(','),
        timeframe,
        include_fundamentals: includeFundamentals,
      },
    });
    return response.data;
  },

  // Get ICT analysis
  async getICTAnalysis(
    symbol: string,
    timeframe: string = '1d',
    concepts?: number[]
  ): Promise<StockAnalysisResponse> {
    const params: any = {
      timeframe,
    };
    
    if (concepts && concepts.length > 0) {
      params.concepts = concepts.join(',');
    }

    const response = await api.get(`/api/v1/ict/analysis/${symbol}`, {
      params,
    });
    return response.data;
  },

  // Get market overview
  async getMarketOverview(): Promise<MarketOverview> {
    const response = await api.get('/api/v1/market/overview');
    return response.data;
  },

  // Get default watchlist
  async getDefaultWatchlist(): Promise<WatchlistResponse> {
    const response = await api.get('/api/v1/watchlist/default');
    return response.data;
  },
};

export default api;
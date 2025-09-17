// ICT Stock Trader Types
export interface StockData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  symbol: string;
  timeframe: string;
}

export interface MarketStructure {
  current_structure: string;
  trend_direction: string;
  structure_breaks: StructureBreak[];
  confidence: number;
  swing_highs: SwingPoint[];
  swing_lows: SwingPoint[];
}

export interface SwingPoint {
  timestamp: string;
  price: number;
  strength: number;
  index: number;
}

export interface StructureBreak {
  type: string;
  timestamp: string;
  price: number;
  previous_high?: number;
  previous_low?: number;
}

export interface LiquidityLevel {
  price_level: number;
  timestamp: string;
  strength: number;
  type: 'buyside' | 'sellside';
  distance_from_current: number;
}

export interface LiquidityPool {
  timestamp: string;
  price_level: number;
  pool_type: string;
  strength: number;
  touches: number;
}

export interface OrderBlock {
  timestamp: string;
  high_price: number;
  low_price: number;
  block_type: 'bullish' | 'bearish';
  strength: number;
  is_breaker: boolean;
}

export interface FairValueGap {
  timestamp: string;
  gap_high: number;
  gap_low: number;
  gap_type: 'bullish' | 'bearish';
  gap_size: number;
  mitigation_level: number;
  is_mitigated: boolean;
}

export interface PremiumDiscount {
  range_high: number;
  range_low: number;
  current_price: number;
  price_position_percentage: number;
  market_bias: 'premium' | 'discount' | 'neutral';
  premium_zone: boolean;
  discount_zone: boolean;
  optimal_trade_entry: {
    bullish_ote_high: number;
    bullish_ote_low: number;
    bearish_ote_high: number;
    bearish_ote_low: number;
    in_bullish_ote: boolean;
    in_bearish_ote: boolean;
  };
}

export interface ICTAnalysis {
  concept_1_market_structure?: MarketStructure;
  concept_2_liquidity?: {
    buyside_liquidity: LiquidityLevel[];
    sellside_liquidity: LiquidityLevel[];
    nearest_buyside: LiquidityLevel | null;
    nearest_sellside: LiquidityLevel | null;
    liquidity_balance: number;
  };
  concept_3_liquidity_pools?: LiquidityPool[];
  concept_4_order_blocks?: OrderBlock[];
  concept_5_breaker_blocks?: OrderBlock[];
  concept_6_fair_value_gaps?: FairValueGap[];
  concept_10_premium_discount?: PremiumDiscount;
  [key: string]: any;
}

export interface StockAnalysisResponse {
  symbol: string;
  timeframe: string;
  analysis_timestamp: string;
  data_points_analyzed: number;
  ict_analysis: ICTAnalysis;
  summary: {
    total_concepts_analyzed: number;
    key_findings: Array<{
      concept: string;
      finding: string;
    }>;
    overall_bias: 'bullish' | 'bearish' | 'neutral';
    confidence_score: number;
  };
}

export interface StockDataResponse {
  symbol: string;
  timeframe: string;
  data: StockData[];
  fundamentals: {
    market_cap?: number;
    pe_ratio?: number;
    sector?: string;
    industry?: string;
    analyst_target_price?: number;
  };
  economic_context: Record<string, any>;
  last_update: string;
  data_points: number;
}

export interface WatchlistItem {
  symbol: string;
  price_data: {
    current_price: number;
    daily_change: number;
    volume: number;
  };
  fundamentals: {
    market_cap?: number;
    sector?: string;
    pe_ratio?: number;
  };
  ict_snapshot: {
    market_structure: string;
    trend_direction: string;
    liquidity_balance: number;
    market_bias: string;
    in_premium: boolean;
    in_discount: boolean;
  };
}

export interface WatchlistResponse {
  timestamp: string;
  watchlist: Record<string, WatchlistItem>;
  market_context: any;
}

export interface MarketOverview {
  timestamp: string;
  market_indices: Record<string, any>;
  economic_indicators: Record<string, any>;
  market_context: {
    overall_sentiment: string;
    market_trend: string;
    volatility_level: string;
    economic_backdrop: string;
  };
  market_sentiment: {
    overall: string;
    volatility: string;
    trend: string;
    confidence: number;
  };
  news: Array<{
    title: string;
    description: string;
    url: string;
    publishedAt: string;
    source: {
      name: string;
    };
  }>;
}

// Chart Types
export interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartAnnotation {
  time: string;
  price: number;
  text: string;
  type: 'orderblock' | 'fvg' | 'liquidity' | 'structure';
  color: string;
}

// UI State Types
export interface AppState {
  selectedSymbol: string;
  selectedTimeframe: string;
  watchlist: string[];
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
}

export interface ChartState {
  data: ChartData[];
  annotations: ChartAnnotation[];
  loading: boolean;
  error: string | null;
}

// API Error Type
export interface APIError {
  detail: string;
  status?: number;
}

// Time & Price Types
export interface KillzoneAnalysis {
  session: string;
  description: string;
  importance: number;
  data_points: number;
  volume_profile: {
    total_volume: number;
    avg_volume: number;
    volume_spikes: number;
  };
  price_action: {
    session_high: number;
    session_low: number;
    session_range: number;
    net_change: number;
    volatility: number;
  };
  session_bias: 'bullish' | 'bearish' | 'neutral';
}

export interface FibonacciAnalysis {
  fibonacci_levels: Array<{
    level: number;
    percentage: number;
    price: number;
    level_type: string;
    significance: number;
  }>;
  current_price: number;
  current_analysis: {
    nearest_support: any;
    nearest_resistance: any;
    in_golden_zone: boolean;
    in_ote_zone: boolean;
    price_level_strength: string;
  };
  key_zones: Record<string, any[]>;
}
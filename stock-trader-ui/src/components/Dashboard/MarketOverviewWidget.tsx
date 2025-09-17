import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  LinearProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Article,
  Speed,
  AccountBalance,
} from '@mui/icons-material';
import { MarketOverview, APIError } from '../../types';

interface MarketOverviewWidgetProps {
  data?: MarketOverview;
  loading: boolean;
  error: APIError | null;
}

const MarketOverviewWidget: React.FC<MarketOverviewWidgetProps> = ({ data, loading, error }) => {
  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'success';
    if (change < 0) return 'error';
    return 'default';
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp color="success" fontSize="small" />;
    if (change < 0) return <TrendingDown color="error" fontSize="small" />;
    return <TrendingFlat color="warning" fontSize="small" />;
  };

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Market Overview
          </Typography>
          <Alert severity="error">
            Error loading market data: {error.detail}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Market Overview
        </Typography>
        
        {loading ? (
          <LinearProgress />
        ) : (
          <Grid container spacing={2}>
            {/* Major Indices */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                <AccountBalance sx={{ mr: 1, verticalAlign: 'middle' }} />
                Major Indices
              </Typography>
              <Box>
                {data?.market_indices &&
                  Object.entries(data.market_indices).map(([symbol, indexData]: [string, any]) => {
                    if (indexData['1d'] && indexData['1d'].length > 0) {
                      const currentData = indexData['1d'][indexData['1d'].length - 1];
                      const previousData = indexData['1d'][indexData['1d'].length - 2] || currentData;
                      const change = ((currentData.close - previousData.close) / previousData.close) * 100;
                      
                      return (
                        <Box key={symbol} display="flex" justifyContent="space-between" alignItems="center" py={1}>
                          <Box display="flex" alignItems="center">
                            {getTrendIcon(change)}
                            <Typography variant="body2" sx={{ ml: 1, fontWeight: 'bold' }}>
                              {symbol}
                            </Typography>
                          </Box>
                          <Box textAlign="right">
                            <Typography variant="body2">
                              ${currentData.close.toFixed(2)}
                            </Typography>
                            <Chip
                              label={formatPercentage(change)}
                              color={getChangeColor(change)}
                              size="small"
                              variant="outlined"
                            />
                          </Box>
                        </Box>
                      );
                    }
                    return null;
                  })}
              </Box>
            </Grid>

            {/* Economic Indicators */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                <Speed sx={{ mr: 1, verticalAlign: 'middle' }} />
                Economic Indicators
              </Typography>
              <Box>
                {data?.economic_indicators &&
                  Object.entries(data.economic_indicators).slice(0, 5).map(([indicator, indicatorData]: [string, any]) => (
                    <Box key={indicator} display="flex" justifyContent="space-between" alignItems="center" py={1}>
                      <Typography variant="body2" color="textSecondary">
                        {indicator.replace(/_/g, ' ').toUpperCase()}
                      </Typography>
                      <Box textAlign="right">
                        <Typography variant="body2">
                          {indicatorData.current_value?.toFixed(2) || 'N/A'}
                        </Typography>
                        {indicatorData.change && (
                          <Typography
                            variant="caption"
                            color={indicatorData.change > 0 ? 'success.main' : 'error.main'}
                          >
                            {formatPercentage(indicatorData.change)}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  ))}
              </Box>
            </Grid>

            <Grid item xs={12}>
              <Divider sx={{ my: 2 }} />
            </Grid>

            {/* Market Context */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                Market Context
              </Typography>
              <Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography variant="body2">Overall Sentiment:</Typography>
                  <Chip
                    label={data?.market_context?.overall_sentiment || 'Unknown'}
                    color={
                      data?.market_context?.overall_sentiment === 'bullish'
                        ? 'success'
                        : data?.market_context?.overall_sentiment === 'bearish'
                        ? 'error'
                        : 'default'
                    }
                    size="small"
                    variant="outlined"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography variant="body2">Market Trend:</Typography>
                  <Chip
                    label={data?.market_context?.market_trend || 'Unknown'}
                    color="info"
                    size="small"
                    variant="outlined"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography variant="body2">Volatility Level:</Typography>
                  <Chip
                    label={data?.market_context?.volatility_level || 'Unknown'}
                    color={
                      data?.market_context?.volatility_level === 'high'
                        ? 'error'
                        : data?.market_context?.volatility_level === 'low'
                        ? 'success'
                        : 'warning'
                    }
                    size="small"
                    variant="outlined"
                  />
                </Box>
              </Box>
            </Grid>

            {/* Latest News */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" gutterBottom>
                <Article sx={{ mr: 1, verticalAlign: 'middle' }} />
                Latest Market News
              </Typography>
              <List dense>
                {data?.news?.slice(0, 3).map((article, index) => (
                  <ListItem key={index} sx={{ px: 0 }}>
                    <ListItemText
                      primary={
                        <Typography variant="body2" noWrap>
                          {article.title}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="textSecondary">
                          {article.source.name} • {new Date(article.publishedAt).toLocaleDateString()}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>
          </Grid>
        )}
        
        {data?.timestamp && (
          <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
            Last updated: {new Date(data.timestamp).toLocaleString()}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default MarketOverviewWidget;
import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Speed,
  Assessment,
  Timeline,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { stockAPI } from '../../services/api';
import WatchlistWidget from './WatchlistWidget';
import MarketOverviewWidget from './MarketOverviewWidget';
import ICTSummaryWidget from './ICTSummaryWidget';

const Dashboard: React.FC = () => {
  const {
    data: marketOverview,
    isLoading: marketLoading,
    error: marketError,
  } = useQuery({
    queryKey: ['marketOverview'],
    queryFn: stockAPI.getMarketOverview,
    refetchInterval: 60000, // Refetch every minute
  });

  const {
    data: watchlistData,
    isLoading: watchlistLoading,
    error: watchlistError,
  } = useQuery({
    queryKey: ['watchlist'],
    queryFn: stockAPI.getDefaultWatchlist,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const getTrendIcon = (trend: string) => {
    switch (trend.toLowerCase()) {
      case 'uptrend':
      case 'bullish':
        return <TrendingUp color="success" />;
      case 'downtrend':
      case 'bearish':
        return <TrendingDown color="error" />;
      default:
        return <TrendingFlat color="warning" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'bullish':
        return 'success';
      case 'bearish':
        return 'error';
      default:
        return 'warning';
    }
  };

  if (marketError || watchlistError) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Error loading dashboard data: {(marketError as any)?.detail || (watchlistError as any)?.detail}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        ICT Trading Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Market Overview Summary */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Assessment sx={{ mr: 1 }} />
                <Typography variant="h6">Market Sentiment</Typography>
              </Box>
              {marketLoading ? (
                <LinearProgress />
              ) : (
                <>
                  <Typography variant="h4" gutterBottom>
                    <Chip
                      label={marketOverview?.market_sentiment?.overall || 'Unknown'}
                      color={getSentimentColor(marketOverview?.market_sentiment?.overall || '')}
                      variant="outlined"
                    />
                  </Typography>
                  <Box display="flex" alignItems="center">
                    {getTrendIcon(marketOverview?.market_sentiment?.trend || '')}
                    <Typography variant="body2" sx={{ ml: 1 }}>
                      {marketOverview?.market_sentiment?.trend || 'Unknown'}
                    </Typography>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Volatility */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Speed sx={{ mr: 1 }} />
                <Typography variant="h6">Volatility</Typography>
              </Box>
              {marketLoading ? (
                <LinearProgress />
              ) : (
                <>
                  <Typography variant="h4" gutterBottom>
                    <Chip
                      label={marketOverview?.market_sentiment?.volatility || 'Unknown'}
                      color={
                        marketOverview?.market_sentiment?.volatility === 'high'
                          ? 'error'
                          : marketOverview?.market_sentiment?.volatility === 'low'
                          ? 'success'
                          : 'warning'
                      }
                      variant="outlined"
                    />
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {marketOverview?.market_context?.volatility_level || 'Normal'}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Economic Backdrop */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Timeline sx={{ mr: 1 }} />
                <Typography variant="h6">Economic Backdrop</Typography>
              </Box>
              {marketLoading ? (
                <LinearProgress />
              ) : (
                <>
                  <Typography variant="h4" gutterBottom>
                    <Chip
                      label={marketOverview?.market_context?.economic_backdrop || 'Unknown'}
                      color="info"
                      variant="outlined"
                    />
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Overall Market Trend
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Confidence Score */}
        <Grid item xs={12} md={6} lg={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Confidence Score
              </Typography>
              {marketLoading ? (
                <LinearProgress />
              ) : (
                <>
                  <Typography variant="h4" gutterBottom>
                    {Math.round((marketOverview?.market_sentiment?.confidence || 0.5) * 100)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(marketOverview?.market_sentiment?.confidence || 0.5) * 100}
                    sx={{ mt: 1 }}
                  />
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Market Overview Widget */}
        <Grid item xs={12} lg={8}>
          <MarketOverviewWidget
            data={marketOverview}
            loading={marketLoading}
            error={marketError}
          />
        </Grid>

        {/* ICT Summary */}
        <Grid item xs={12} lg={4}>
          <ICTSummaryWidget />
        </Grid>

        {/* Watchlist */}
        <Grid item xs={12}>
          <WatchlistWidget
            data={watchlistData}
            loading={watchlistLoading}
            error={watchlistError}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
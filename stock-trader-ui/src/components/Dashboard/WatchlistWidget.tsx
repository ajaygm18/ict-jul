import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Box,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
} from '@mui/icons-material';
import { WatchlistResponse, APIError } from '../../types';

interface WatchlistWidgetProps {
  data?: WatchlistResponse;
  loading: boolean;
  error: APIError | null;
}

const WatchlistWidget: React.FC<WatchlistWidgetProps> = ({ data, loading, error }) => {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(price);
  };

  const formatChange = (change: number) => {
    const formatted = (change * 100).toFixed(2);
    return `${change >= 0 ? '+' : ''}${formatted}%`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'success';
    if (change < 0) return 'error';
    return 'default';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend.toLowerCase()) {
      case 'uptrend':
        return <TrendingUp color="success" fontSize="small" />;
      case 'downtrend':
        return <TrendingDown color="error" fontSize="small" />;
      default:
        return <TrendingFlat color="warning" fontSize="small" />;
    }
  };

  const getBiasColor = (bias: string) => {
    switch (bias.toLowerCase()) {
      case 'premium':
        return 'error';
      case 'discount':
        return 'success';
      default:
        return 'default';
    }
  };

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Watchlist
          </Typography>
          <Alert severity="error">
            Error loading watchlist: {error.detail}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          ICT Watchlist Analysis
        </Typography>
        {loading ? (
          <LinearProgress />
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell align="right">Price</TableCell>
                  <TableCell align="right">Change</TableCell>
                  <TableCell align="right">Volume</TableCell>
                  <TableCell>Sector</TableCell>
                  <TableCell>Market Structure</TableCell>
                  <TableCell>Trend</TableCell>
                  <TableCell>Market Bias</TableCell>
                  <TableCell>ICT Zone</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data?.watchlist &&
                  Object.entries(data.watchlist).map(([symbol, item]) => (
                    <TableRow key={symbol} hover>
                      <TableCell component="th" scope="row">
                        <Typography variant="subtitle2" fontWeight="bold">
                          {symbol}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {formatPrice(item.price_data.current_price)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={formatChange(item.price_data.daily_change)}
                          color={getChangeColor(item.price_data.daily_change)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" color="textSecondary">
                          {formatVolume(item.price_data.volume)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="textSecondary">
                          {item.fundamentals.sector || 'N/A'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          {getTrendIcon(item.ict_snapshot.trend_direction)}
                          <Typography variant="body2" sx={{ ml: 0.5 }}>
                            {item.ict_snapshot.market_structure}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {item.ict_snapshot.trend_direction}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={item.ict_snapshot.market_bias}
                          color={getBiasColor(item.ict_snapshot.market_bias)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Box>
                          {item.ict_snapshot.in_premium && (
                            <Chip
                              label="Premium"
                              color="error"
                              size="small"
                              variant="filled"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          )}
                          {item.ict_snapshot.in_discount && (
                            <Chip
                              label="Discount"
                              color="success"
                              size="small"
                              variant="filled"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          )}
                          {!item.ict_snapshot.in_premium && !item.ict_snapshot.in_discount && (
                            <Chip
                              label="Equilibrium"
                              color="default"
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
        {data?.timestamp && (
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
            Last updated: {new Date(data.timestamp).toLocaleString()}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default WatchlistWidget;
import React, { useEffect, useRef, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Chip,
  CircularProgress,
  FormGroup,
  FormControlLabel,
  Switch,
  Divider,
} from '@mui/material';
import { Refresh, TrendingUp } from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, Time } from 'lightweight-charts';
import { stockAPI } from '../../services/api';

interface ChartControls {
  symbol: string;
  timeframe: string;
  showOrderBlocks: boolean;
  showFVGs: boolean;
  showLiquidity: boolean;
  showStructure: boolean;
  showKillzones: boolean;
}

const TradingView: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [controls, setControls] = useState<ChartControls>({
    symbol: 'AAPL',
    timeframe: '1h',
    showOrderBlocks: true,
    showFVGs: true,
    showLiquidity: true,
    showStructure: true,
    showKillzones: true,
  });

  // Fetch stock data
  const {
    data: stockData,
    isLoading: stockLoading,
    error: stockError,
    refetch: refetchStock,
  } = useQuery({
    queryKey: ['stockData', controls.symbol, controls.timeframe],
    queryFn: () => stockAPI.getStockData(controls.symbol, controls.timeframe),
    enabled: !!controls.symbol,
  });

  // Fetch ICT analysis
  const {
    data: ictAnalysis,
    isLoading: ictLoading,
    error: ictError,
    refetch: refetchICT,
  } = useQuery({
    queryKey: ['ictAnalysis', controls.symbol],
    queryFn: () => stockAPI.getICTAnalysis(controls.symbol),
    enabled: !!controls.symbol,
  });

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#ffffff',
      },
      grid: {
        vertLines: { color: '#404040' },
        horzLines: { color: '#404040' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff4444',
      borderDownColor: '#ff4444',
      borderUpColor: '#00ff88',
      wickDownColor: '#ff4444',
      wickUpColor: '#00ff88',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, []);

  // Update chart data
  useEffect(() => {
    if (!stockData?.data || !candleSeriesRef.current) return;

    try {
      const chartData: CandlestickData[] = stockData.data.map((item: any) => ({
        time: (new Date(item.timestamp).getTime() / 1000) as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));

      candleSeriesRef.current.setData(chartData);
    } catch (error) {
      console.error('Error updating chart data:', error);
    }
  }, [stockData]);

  // Add ICT pattern overlays
  useEffect(() => {
    if (!ictAnalysis?.ict_analysis || !chartRef.current) return;

    // Clear existing overlays (in a real implementation, you'd track and remove them)
    
    // Add Order Blocks
    if (controls.showOrderBlocks && ictAnalysis.ict_analysis.concept_4_order_blocks) {
      const orderBlocks = ictAnalysis.ict_analysis.concept_4_order_blocks;
      orderBlocks.forEach((ob: any) => {
        if (ob.timestamp && ob.high_price && ob.low_price) {
          // Add order block visualization (simplified)
          const lineSeries = chartRef.current!.addLineSeries({
            color: ob.block_type === 'bullish' ? '#00ff88' : '#ff4444',
            lineWidth: 2,
            title: `${ob.block_type} Order Block`,
          });
          
          const lineData: LineData[] = [{
            time: (new Date(ob.timestamp).getTime() / 1000) as Time,
            value: (ob.high_price + ob.low_price) / 2,
          }];
          
          lineSeries.setData(lineData);
        }
      });
    }

    // Add Fair Value Gaps
    if (controls.showFVGs && ictAnalysis.ict_analysis.concept_6_fair_value_gaps) {
      const fvgs = ictAnalysis.ict_analysis.concept_6_fair_value_gaps;
      fvgs.forEach((fvg: any) => {
        if (fvg.timestamp && fvg.gap_high && fvg.gap_low) {
          const lineSeries = chartRef.current!.addLineSeries({
            color: fvg.gap_type === 'bullish' ? '#00ff8850' : '#ff444450',
            lineWidth: 1,
            title: `${fvg.gap_type} FVG`,
          });
          
          const lineData: LineData[] = [{
            time: (new Date(fvg.timestamp).getTime() / 1000) as Time,
            value: (fvg.gap_high + fvg.gap_low) / 2,
          }];
          
          lineSeries.setData(lineData);
        }
      });
    }
  }, [ictAnalysis, controls]);

  const handleRefresh = () => {
    refetchStock();
    refetchICT();
  };

  const handleControlChange = (field: keyof ChartControls, value: any) => {
    setControls(prev => ({ ...prev, [field]: value }));
  };

  const getPatternSummary = () => {
    if (!ictAnalysis?.ict_analysis) return null;

    const patterns = [];
    if (ictAnalysis.ict_analysis.concept_4_order_blocks && ictAnalysis.ict_analysis.concept_4_order_blocks.length > 0) {
      patterns.push(`${ictAnalysis.ict_analysis.concept_4_order_blocks.length} Order Blocks`);
    }
    if (ictAnalysis.ict_analysis.concept_6_fair_value_gaps && ictAnalysis.ict_analysis.concept_6_fair_value_gaps.length > 0) {
      patterns.push(`${ictAnalysis.ict_analysis.concept_6_fair_value_gaps.length} FVGs`);
    }
    if (ictAnalysis.ict_analysis.concept_2_liquidity && ictAnalysis.ict_analysis.concept_2_liquidity.buyside_liquidity && ictAnalysis.ict_analysis.concept_2_liquidity.buyside_liquidity.length > 0) {
      patterns.push(`${ictAnalysis.ict_analysis.concept_2_liquidity.buyside_liquidity.length} Liquidity Pools`);
    }

    return patterns;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        ICT Trading Charts
      </Typography>

      {/* Chart Controls */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={2}>
              <TextField
                label="Symbol"
                value={controls.symbol}
                onChange={(e) => handleControlChange('symbol', e.target.value.toUpperCase())}
                variant="outlined"
                size="small"
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Timeframe</InputLabel>
                <Select
                  value={controls.timeframe}
                  label="Timeframe"
                  onChange={(e) => handleControlChange('timeframe', e.target.value)}
                >
                  <MenuItem value="5m">5 Minutes</MenuItem>
                  <MenuItem value="15m">15 Minutes</MenuItem>
                  <MenuItem value="1h">1 Hour</MenuItem>
                  <MenuItem value="4h">4 Hours</MenuItem>
                  <MenuItem value="1d">1 Day</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Button
                variant="contained"
                startIcon={stockLoading || ictLoading ? <CircularProgress size={16} /> : <Refresh />}
                onClick={handleRefresh}
                disabled={stockLoading || ictLoading}
                fullWidth
              >
                Refresh
              </Button>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box display="flex" gap={1} flexWrap="wrap">
                {getPatternSummary()?.map((pattern, index) => (
                  <Chip
                    key={index}
                    label={pattern}
                    size="small"
                    color="primary"
                    variant="outlined"
                    icon={<TrendingUp />}
                  />
                ))}
              </Box>
            </Grid>
          </Grid>

          <Divider sx={{ my: 2 }} />

          {/* ICT Overlay Controls */}
          <Typography variant="subtitle2" gutterBottom>
            ICT Pattern Overlays
          </Typography>
          <FormGroup row>
            <FormControlLabel
              control={
                <Switch
                  checked={controls.showOrderBlocks}
                  onChange={(e) => handleControlChange('showOrderBlocks', e.target.checked)}
                />
              }
              label="Order Blocks"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={controls.showFVGs}
                  onChange={(e) => handleControlChange('showFVGs', e.target.checked)}
                />
              }
              label="Fair Value Gaps"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={controls.showLiquidity}
                  onChange={(e) => handleControlChange('showLiquidity', e.target.checked)}
                />
              }
              label="Liquidity Levels"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={controls.showStructure}
                  onChange={(e) => handleControlChange('showStructure', e.target.checked)}
                />
              }
              label="Market Structure"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={controls.showKillzones}
                  onChange={(e) => handleControlChange('showKillzones', e.target.checked)}
                />
              }
              label="Killzones"
            />
          </FormGroup>
        </CardContent>
      </Card>

      {/* Error Display */}
      {(stockError || ictError) && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Error loading chart data: {(stockError as any)?.detail || (ictError as any)?.detail}
        </Alert>
      )}

      {/* Chart Container */}
      <Card>
        <CardContent>
          <Box
            ref={chartContainerRef}
            sx={{
              width: '100%',
              height: 600,
              bgcolor: '#1a1a1a',
              borderRadius: 1,
            }}
          />
          
          {(stockLoading || ictLoading) && (
            <Box
              display="flex"
              justifyContent="center"
              alignItems="center"
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              bgcolor="rgba(0,0,0,0.5)"
              borderRadius={1}
            >
              <CircularProgress />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* ICT Analysis Summary */}
      {ictAnalysis && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              ICT Analysis Summary - {controls.symbol}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" color="primary">
                  Market Structure
                </Typography>
                <Typography variant="body2">
                  {ictAnalysis.summary?.overall_bias || 'Neutral'}
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" color="primary">
                  Confidence Score
                </Typography>
                <Typography variant="body2">
                  {Math.round((ictAnalysis.summary?.confidence_score || 0.5) * 100)}%
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle2" color="primary">
                  Concepts Analyzed
                </Typography>
                <Typography variant="body2">
                  {ictAnalysis.summary?.total_concepts_analyzed || 0} ICT Concepts
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default TradingView;
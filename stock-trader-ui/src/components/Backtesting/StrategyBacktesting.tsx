import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Divider,
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Assessment,
  TrendingUp,
  TrendingDown,
  Speed,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface BacktestConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  timeframe: string;
  strategies: string[];
  riskPerTrade: number;
  maxDrawdown: number;
}

interface BacktestResult {
  strategy: string;
  totalTrades: number;
  winRate: number;
  totalReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  trades: TradeResult[];
}

interface TradeResult {
  id: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  strategy: string;
  reason: string;
  duration: string;
}

const StrategyBacktesting: React.FC = () => {
  const [config, setConfig] = useState<BacktestConfig>({
    symbol: 'AAPL',
    startDate: '2024-01-01',
    endDate: '2024-09-17',
    initialCapital: 10000,
    timeframe: '1d',
    strategies: ['silver_bullet', 'power_hour', 'ote_strategy'],
    riskPerTrade: 2,
    maxDrawdown: 20,
  });

  const [isBacktesting, setIsBacktesting] = useState(false);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);

  const availableStrategies = [
    { id: 'silver_bullet', name: 'Silver Bullet Strategy', description: '9:45-10:00 AM optimal entry window' },
    { id: 'power_hour', name: 'Power Hour Strategy', description: '3:00-4:00 PM institutional activity' },
    { id: 'ote_strategy', name: 'Optimal Trade Entry', description: '62%-79% retracement zones' },
    { id: 'order_block', name: 'Order Block Strategy', description: 'Institutional order block entries' },
    { id: 'fvg_sniper', name: 'FVG Sniper Entry', description: 'Precision Fair Value Gap entries' },
    { id: 'smt_divergence', name: 'SMT Divergence', description: 'Cross-market correlation analysis' },
    { id: 'turtle_soup', name: 'Turtle Soup Strategy', description: '20-day breakout failures' },
    { id: 'market_open_reversal', name: 'Market Open Reversal', description: 'First hour reversal patterns' },
  ];

  const runBacktest = async () => {
    setIsBacktesting(true);
    
    try {
      // Simulate backtest results (in real implementation, this would call the backend)
      await new Promise(resolve => setTimeout(resolve, 3000)); // Simulate processing time
      
      const mockResults: BacktestResult[] = config.strategies.map(strategy => {
        const strategyInfo = availableStrategies.find(s => s.id === strategy);
        const totalTrades = Math.floor(Math.random() * 50) + 20;
        const winRate = Math.random() * 0.4 + 0.4; // 40-80%
        const totalReturn = (Math.random() * 40 - 10); // -10% to +30%
        
        return {
          strategy: strategyInfo?.name || strategy,
          totalTrades,
          winRate: winRate * 100,
          totalReturn,
          maxDrawdown: Math.random() * 15 + 5,
          sharpeRatio: Math.random() * 2 + 0.5,
          profitFactor: Math.random() * 2 + 0.8,
          avgWin: Math.random() * 3 + 1,
          avgLoss: -(Math.random() * 2 + 0.5),
          trades: generateMockTrades(totalTrades, strategy),
        };
      });
      
      setBacktestResults(mockResults);
    } finally {
      setIsBacktesting(false);
    }
  };

  const generateMockTrades = (count: number, strategy: string): TradeResult[] => {
    return Array.from({ length: count }, (_, i) => {
      const isWin = Math.random() > 0.4;
      const entryPrice = Math.random() * 200 + 100;
      const exitPrice = entryPrice * (isWin ? (1 + Math.random() * 0.05) : (1 - Math.random() * 0.03));
      
      return {
        id: `${strategy}_${i}`,
        entryDate: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
        exitDate: new Date(Date.now() - Math.random() * 85 * 24 * 60 * 60 * 1000).toISOString(),
        entryPrice,
        exitPrice,
        pnl: ((exitPrice - entryPrice) / entryPrice) * 100,
        strategy,
        reason: isWin ? 'Target reached' : 'Stop loss hit',
        duration: `${Math.floor(Math.random() * 48) + 1}h`,
      };
    });
  };

  const handleConfigChange = (field: keyof BacktestConfig, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleStrategyToggle = (strategyId: string) => {
    setConfig(prev => ({
      ...prev,
      strategies: prev.strategies.includes(strategyId)
        ? prev.strategies.filter(s => s !== strategyId)
        : [...prev.strategies, strategyId]
    }));
  };

  const getPerformanceColor = (value: number, isReturn: boolean = false) => {
    if (isReturn) {
      return value > 0 ? '#00ff88' : '#ff4444';
    }
    return value > 60 ? '#00ff88' : value > 40 ? '#ffaa00' : '#ff4444';
  };

  const generateEquityCurveData = () => {
    if (backtestResults.length === 0) return [];
    
    // Combine all trades and sort by date
    const allTrades = backtestResults.flatMap(result => 
      result.trades.map(trade => ({
        ...trade,
        strategyName: result.strategy,
      }))
    ).sort((a, b) => new Date(a.entryDate).getTime() - new Date(b.entryDate).getTime());

    let runningTotal = config.initialCapital;
    const equityData = [{ date: config.startDate, equity: runningTotal, drawdown: 0 }];

    allTrades.forEach(trade => {
      const tradeValue = (runningTotal * config.riskPerTrade / 100) * (trade.pnl / 100);
      runningTotal += tradeValue;
      
      equityData.push({
        date: trade.exitDate.split('T')[0],
        equity: runningTotal,
        drawdown: Math.max(0, (Math.max(...equityData.map(d => d.equity)) - runningTotal) / Math.max(...equityData.map(d => d.equity)) * 100),
      });
    });

    return equityData;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        ICT Strategy Backtesting
      </Typography>

      {/* Configuration Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Backtest Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
              <TextField
                label="Symbol"
                value={config.symbol}
                onChange={(e) => handleConfigChange('symbol', e.target.value.toUpperCase())}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Timeframe</InputLabel>
                <Select
                  value={config.timeframe}
                  label="Timeframe"
                  onChange={(e) => handleConfigChange('timeframe', e.target.value)}
                >
                  <MenuItem value="5m">5 Minutes</MenuItem>
                  <MenuItem value="15m">15 Minutes</MenuItem>
                  <MenuItem value="1h">1 Hour</MenuItem>
                  <MenuItem value="1d">1 Day</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="Start Date"
                type="date"
                value={config.startDate}
                onChange={(e) => handleConfigChange('startDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="End Date"
                type="date"
                value={config.endDate}
                onChange={(e) => handleConfigChange('endDate', e.target.value)}
                InputLabelProps={{ shrink: true }}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                label="Initial Capital ($)"
                type="number"
                value={config.initialCapital}
                onChange={(e) => handleConfigChange('initialCapital', Number(e.target.value))}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                label="Risk Per Trade (%)"
                type="number"
                value={config.riskPerTrade}
                onChange={(e) => handleConfigChange('riskPerTrade', Number(e.target.value))}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                label="Max Drawdown (%)"
                type="number"
                value={config.maxDrawdown}
                onChange={(e) => handleConfigChange('maxDrawdown', Number(e.target.value))}
                fullWidth
              />
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          <Typography variant="subtitle1" gutterBottom>
            Select ICT Strategies to Test
          </Typography>
          <FormGroup row>
            {availableStrategies.map((strategy) => (
              <FormControlLabel
                key={strategy.id}
                control={
                  <Checkbox
                    checked={config.strategies.includes(strategy.id)}
                    onChange={() => handleStrategyToggle(strategy.id)}
                  />
                }
                label={
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {strategy.name}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {strategy.description}
                    </Typography>
                  </Box>
                }
              />
            ))}
          </FormGroup>

          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={isBacktesting ? <LinearProgress /> : <PlayArrow />}
              onClick={runBacktest}
              disabled={isBacktesting || config.strategies.length === 0}
            >
              {isBacktesting ? 'Running Backtest...' : 'Run Backtest'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Results */}
      {backtestResults.length > 0 && (
        <>
          {/* Performance Summary */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Backtest Results Summary
              </Typography>
              
              <Grid container spacing={2}>
                {backtestResults.map((result) => (
                  <Grid item xs={12} md={6} lg={4} key={result.strategy}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                          {result.strategy}
                        </Typography>
                        <Grid container spacing={1}>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Total Return
                            </Typography>
                            <Typography 
                              variant="body2" 
                              fontWeight="bold"
                              sx={{ color: getPerformanceColor(result.totalReturn, true) }}
                            >
                              {result.totalReturn.toFixed(2)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Win Rate
                            </Typography>
                            <Typography 
                              variant="body2" 
                              fontWeight="bold"
                              sx={{ color: getPerformanceColor(result.winRate) }}
                            >
                              {result.winRate.toFixed(1)}%
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Total Trades
                            </Typography>
                            <Typography variant="body2">
                              {result.totalTrades}
                            </Typography>
                          </Grid>
                          <Grid item xs={6}>
                            <Typography variant="caption" color="textSecondary">
                              Sharpe Ratio
                            </Typography>
                            <Typography variant="body2">
                              {result.sharpeRatio.toFixed(2)}
                            </Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>

          {/* Equity Curve */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Equity Curve
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={generateEquityCurveData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="equity" 
                    stroke="#00ff88" 
                    strokeWidth={2}
                    dot={false}
                    name="Equity"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="drawdown" 
                    stroke="#ff4444" 
                    strokeWidth={1}
                    dot={false}
                    name="Drawdown %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Detailed Results */}
          {backtestResults.map((result) => (
            <Accordion key={result.strategy} sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">
                  {result.strategy} - Detailed Results
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3} sx={{ mb: 3 }}>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Assessment sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Performance</Typography>
                        </Box>
                        <Typography variant="h6" sx={{ color: getPerformanceColor(result.totalReturn, true) }}>
                          {result.totalReturn.toFixed(2)}%
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Total Return
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <TrendingUp sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Win Rate</Typography>
                        </Box>
                        <Typography variant="h6" sx={{ color: getPerformanceColor(result.winRate) }}>
                          {result.winRate.toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {Math.round(result.totalTrades * result.winRate / 100)} wins
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <TrendingDown sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Max Drawdown</Typography>
                        </Box>
                        <Typography variant="h6" color="error">
                          {result.maxDrawdown.toFixed(2)}%
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Peak to trough
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Speed sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Profit Factor</Typography>
                        </Box>
                        <Typography variant="h6">
                          {result.profitFactor.toFixed(2)}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Gross profit / Gross loss
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                <Typography variant="subtitle1" gutterBottom>
                  Trade History
                </Typography>
                <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Entry Date</TableCell>
                        <TableCell>Exit Date</TableCell>
                        <TableCell align="right">Entry Price</TableCell>
                        <TableCell align="right">Exit Price</TableCell>
                        <TableCell align="right">P&L %</TableCell>
                        <TableCell>Duration</TableCell>
                        <TableCell>Reason</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.trades.slice(0, 20).map((trade) => (
                        <TableRow key={trade.id}>
                          <TableCell>
                            {new Date(trade.entryDate).toLocaleDateString()}
                          </TableCell>
                          <TableCell>
                            {new Date(trade.exitDate).toLocaleDateString()}
                          </TableCell>
                          <TableCell align="right">
                            ${trade.entryPrice.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">
                            ${trade.exitPrice.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={`${trade.pnl.toFixed(2)}%`}
                              size="small"
                              color={trade.pnl > 0 ? 'success' : 'error'}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>{trade.duration}</TableCell>
                          <TableCell>{trade.reason}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                {result.trades.length > 20 && (
                  <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                    Showing first 20 trades of {result.trades.length} total trades
                  </Typography>
                )}
              </AccordionDetails>
            </Accordion>
          ))}
        </>
      )}

      {isBacktesting && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="subtitle2">
            Running backtest analysis...
          </Typography>
          <Typography variant="body2">
            Analyzing {config.strategies.length} strategies on {config.symbol} from {config.startDate} to {config.endDate}
          </Typography>
          <LinearProgress sx={{ mt: 1 }} />
        </Alert>
      )}
    </Box>
  );
};

export default StrategyBacktesting;
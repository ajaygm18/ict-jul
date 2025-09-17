import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Assessment,
  TrendingUp,
  TrendingDown,
  Speed,
  Timeline,
  PieChart,
  BarChart,
  ShowChart,
  AccountBalance,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart as RechartsBarChart,
  Bar,
  PieChart as RechartsPieChart,
  Cell,
  Pie,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface PerformanceMetrics {
  totalReturn: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  avgWin: number;
  avgLoss: number;
  bestTrade: number;
  worstTrade: number;
  avgHoldingPeriod: string;
  calmarRatio: number;
}

interface StrategyPerformance {
  name: string;
  trades: number;
  winRate: number;
  totalReturn: number;
  profitFactor: number;
  avgReturn: number;
  maxDrawdown: number;
}

interface MonthlyPerformance {
  month: string;
  return: number;
  trades: number;
  winRate: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const PerformanceAnalytics: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('6m');
  const [selectedMetric, setSelectedMetric] = useState('return');
  const [tabValue, setTabValue] = useState(0);

  // Mock data - in real implementation, this would come from the backend
  const performanceMetrics: PerformanceMetrics = {
    totalReturn: 23.4,
    winRate: 65.2,
    profitFactor: 2.15,
    sharpeRatio: 1.34,
    maxDrawdown: -8.7,
    totalTrades: 127,
    avgWin: 3.2,
    avgLoss: -1.8,
    bestTrade: 12.5,
    worstTrade: -5.3,
    avgHoldingPeriod: '2.3 days',
    calmarRatio: 2.69,
  };

  const strategyPerformance: StrategyPerformance[] = [
    {
      name: 'Silver Bullet',
      trades: 23,
      winRate: 73.9,
      totalReturn: 8.7,
      profitFactor: 2.8,
      avgReturn: 0.38,
      maxDrawdown: -2.1,
    },
    {
      name: 'Power Hour',
      trades: 18,
      winRate: 61.1,
      totalReturn: 5.4,
      profitFactor: 1.9,
      avgReturn: 0.30,
      maxDrawdown: -3.2,
    },
    {
      name: 'OTE Strategy',
      trades: 31,
      winRate: 58.1,
      totalReturn: 6.8,
      profitFactor: 2.1,
      avgReturn: 0.22,
      maxDrawdown: -4.1,
    },
    {
      name: 'Order Block',
      trades: 25,
      winRate: 68.0,
      totalReturn: 7.2,
      profitFactor: 2.4,
      avgReturn: 0.29,
      maxDrawdown: -1.8,
    },
    {
      name: 'SMT Divergence',
      trades: 15,
      winRate: 66.7,
      totalReturn: 4.1,
      profitFactor: 2.2,
      avgReturn: 0.27,
      maxDrawdown: -2.5,
    },
    {
      name: 'Turtle Soup',
      trades: 15,
      winRate: 60.0,
      totalReturn: 3.2,
      profitFactor: 1.8,
      avgReturn: 0.21,
      maxDrawdown: -2.9,
    },
  ];

  const monthlyPerformance: MonthlyPerformance[] = [
    { month: 'Mar 2024', return: 4.2, trades: 18, winRate: 61.1 },
    { month: 'Apr 2024', return: 2.8, trades: 22, winRate: 59.1 },
    { month: 'May 2024', return: 6.1, trades: 25, winRate: 68.0 },
    { month: 'Jun 2024', return: 1.3, trades: 19, winRate: 52.6 },
    { month: 'Jul 2024', return: 5.7, trades: 21, winRate: 71.4 },
    { month: 'Aug 2024', return: 3.3, trades: 22, winRate: 63.6 },
  ];

  const equityCurveData = useMemo(() => {
    let runningTotal = 10000;
    return monthlyPerformance.map(month => {
      runningTotal *= (1 + month.return / 100);
      return {
        month: month.month,
        equity: runningTotal,
        drawdown: Math.max(0, (12340 - runningTotal) / 12340 * 100), // Mock drawdown
      };
    });
  }, [monthlyPerformance]);

  const tradeDistributionData = [
    { name: 'Wins', value: Math.round(performanceMetrics.totalTrades * performanceMetrics.winRate / 100), color: '#00ff88' },
    { name: 'Losses', value: Math.round(performanceMetrics.totalTrades * (1 - performanceMetrics.winRate / 100)), color: '#ff4444' },
  ];

  const getMetricColor = (value: number, isPositive: boolean = true) => {
    if (isPositive) {
      return value > 0 ? '#00ff88' : '#ff4444';
    }
    return value < 0 ? '#00ff88' : '#ff4444';
  };

  const getPerformanceGrade = (winRate: number, profitFactor: number) => {
    if (winRate > 70 && profitFactor > 2.5) return { grade: 'A+', color: '#00ff88' };
    if (winRate > 60 && profitFactor > 2.0) return { grade: 'A', color: '#00ff88' };
    if (winRate > 50 && profitFactor > 1.5) return { grade: 'B', color: '#ffaa00' };
    return { grade: 'C', color: '#ff4444' };
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Performance Analytics
      </Typography>

      {/* Controls */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Time Period</InputLabel>
            <Select
              value={selectedPeriod}
              label="Time Period"
              onChange={(e) => setSelectedPeriod(e.target.value)}
            >
              <MenuItem value="1m">1 Month</MenuItem>
              <MenuItem value="3m">3 Months</MenuItem>
              <MenuItem value="6m">6 Months</MenuItem>
              <MenuItem value="1y">1 Year</MenuItem>
              <MenuItem value="all">All Time</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth size="small">
            <InputLabel>Primary Metric</InputLabel>
            <Select
              value={selectedMetric}
              label="Primary Metric"
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              <MenuItem value="return">Total Return</MenuItem>
              <MenuItem value="winrate">Win Rate</MenuItem>
              <MenuItem value="profit_factor">Profit Factor</MenuItem>
              <MenuItem value="sharpe">Sharpe Ratio</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Assessment sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="subtitle2">Total Return</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: getMetricColor(performanceMetrics.totalReturn) }}>
                {performanceMetrics.totalReturn.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {performanceMetrics.totalTrades} trades
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <TrendingUp sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="subtitle2">Win Rate</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: getMetricColor(performanceMetrics.winRate - 50) }}>
                {performanceMetrics.winRate.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                {Math.round(performanceMetrics.totalTrades * performanceMetrics.winRate / 100)} winning trades
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Speed sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="subtitle2">Profit Factor</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: getMetricColor(performanceMetrics.profitFactor - 1) }}>
                {performanceMetrics.profitFactor.toFixed(2)}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Gross profit / Gross loss
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <TrendingDown sx={{ mr: 1, color: 'error.main' }} />
                <Typography variant="subtitle2">Max Drawdown</Typography>
              </Box>
              <Typography variant="h4" sx={{ color: getMetricColor(performanceMetrics.maxDrawdown, false) }}>
                {performanceMetrics.maxDrawdown.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Peak to trough
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Analytics Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(_, newValue: number) => setTabValue(newValue)}>
            <Tab label="Equity Curve" icon={<ShowChart />} />
            <Tab label="Strategy Breakdown" icon={<BarChart />} />
            <Tab label="Risk Metrics" icon={<Assessment />} />
            <Tab label="Trade Analysis" icon={<PieChart />} />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          {/* Equity Curve */}
          <Typography variant="h6" gutterBottom>
            Equity Curve & Drawdown
          </Typography>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={equityCurveData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="equity"
                stroke="#00ff88"
                strokeWidth={3}
                dot={false}
                name="Equity ($)"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="drawdown"
                stroke="#ff4444"
                strokeWidth={2}
                dot={false}
                name="Drawdown (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          {/* Strategy Breakdown */}
          <Typography variant="h6" gutterBottom>
            ICT Strategy Performance Comparison
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Strategy</TableCell>
                  <TableCell align="right">Trades</TableCell>
                  <TableCell align="right">Win Rate</TableCell>
                  <TableCell align="right">Total Return</TableCell>
                  <TableCell align="right">Profit Factor</TableCell>
                  <TableCell align="right">Avg Return</TableCell>
                  <TableCell align="right">Max DD</TableCell>
                  <TableCell align="center">Grade</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {strategyPerformance.map((strategy) => {
                  const grade = getPerformanceGrade(strategy.winRate, strategy.profitFactor);
                  return (
                    <TableRow key={strategy.name}>
                      <TableCell>{strategy.name}</TableCell>
                      <TableCell align="right">{strategy.trades}</TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${strategy.winRate.toFixed(1)}%`}
                          size="small"
                          color={strategy.winRate > 60 ? 'success' : strategy.winRate > 50 ? 'warning' : 'error'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right" sx={{ color: getMetricColor(strategy.totalReturn) }}>
                        {strategy.totalReturn.toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">{strategy.profitFactor.toFixed(2)}</TableCell>
                      <TableCell align="right">{strategy.avgReturn.toFixed(2)}%</TableCell>
                      <TableCell align="right" sx={{ color: getMetricColor(strategy.maxDrawdown, false) }}>
                        {strategy.maxDrawdown.toFixed(1)}%
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={grade.grade}
                          size="small"
                          sx={{ bgcolor: grade.color, color: 'white' }}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Strategy Returns Comparison
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RechartsBarChart data={strategyPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="totalReturn" fill="#00ff88" name="Total Return %" />
              </RechartsBarChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          {/* Risk Metrics */}
          <Typography variant="h6" gutterBottom>
            Comprehensive Risk Analysis
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Risk-Adjusted Returns
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemIcon>
                        <Timeline />
                      </ListItemIcon>
                      <ListItemText
                        primary="Sharpe Ratio"
                        secondary={
                          <Box display="flex" alignItems="center">
                            <Typography variant="h6" sx={{ mr: 1 }}>
                              {performanceMetrics.sharpeRatio.toFixed(2)}
                            </Typography>
                            <Chip
                              label={performanceMetrics.sharpeRatio > 1 ? 'Good' : 'Fair'}
                              size="small"
                              color={performanceMetrics.sharpeRatio > 1 ? 'success' : 'warning'}
                              variant="outlined"
                            />
                          </Box>
                        }
                      />
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemIcon>
                        <Speed />
                      </ListItemIcon>
                      <ListItemText
                        primary="Calmar Ratio"
                        secondary={
                          <Typography variant="h6">
                            {performanceMetrics.calmarRatio.toFixed(2)}
                          </Typography>
                        }
                      />
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemIcon>
                        <AccountBalance />
                      </ListItemIcon>
                      <ListItemText
                        primary="Profit Factor"
                        secondary={
                          <Typography variant="h6" sx={{ color: getMetricColor(performanceMetrics.profitFactor - 1) }}>
                            {performanceMetrics.profitFactor.toFixed(2)}
                          </Typography>
                        }
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle1" gutterBottom>
                    Trade Statistics
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Average Win"
                        secondary={
                          <Typography variant="h6" color="success.main">
                            +{performanceMetrics.avgWin.toFixed(1)}%
                          </Typography>
                        }
                      />
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemText
                        primary="Average Loss"
                        secondary={
                          <Typography variant="h6" color="error.main">
                            {performanceMetrics.avgLoss.toFixed(1)}%
                          </Typography>
                        }
                      />
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemText
                        primary="Best Trade"
                        secondary={
                          <Typography variant="h6" color="success.main">
                            +{performanceMetrics.bestTrade.toFixed(1)}%
                          </Typography>
                        }
                      />
                    </ListItem>
                    <Divider />
                    <ListItem>
                      <ListItemText
                        primary="Worst Trade"
                        secondary={
                          <Typography variant="h6" color="error.main">
                            {performanceMetrics.worstTrade.toFixed(1)}%
                          </Typography>
                        }
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Monthly Performance Trend
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={monthlyPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="return"
                  stroke="#00ff88"
                  fill="#00ff8830"
                  name="Monthly Return %"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          {/* Trade Analysis */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Win/Loss Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsPieChart>
                  <Pie
                    data={tradeDistributionData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                  >
                    {tradeDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RechartsPieChart>
              </ResponsiveContainer>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Strategy Contribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsPieChart>
                  <Pie
                    data={strategyPerformance}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="totalReturn"
                    nameKey="name"
                    label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                  >
                    {strategyPerformance.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={`hsl(${index * 60}, 70%, 50%)`} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RechartsPieChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Alert severity="info">
              <Typography variant="subtitle2" gutterBottom>
                Performance Insights:
              </Typography>
              <ul>
                <li>Silver Bullet strategy shows the highest win rate at 73.9%</li>
                <li>Order Block strategy has the lowest max drawdown at -1.8%</li>
                <li>Overall portfolio maintains a strong profit factor of {performanceMetrics.profitFactor.toFixed(2)}</li>
                <li>Risk-adjusted returns (Sharpe: {performanceMetrics.sharpeRatio.toFixed(2)}) indicate solid performance</li>
              </ul>
            </Alert>
          </Box>
        </TabPanel>
      </Card>
    </Box>
  );
};

export default PerformanceAnalytics;
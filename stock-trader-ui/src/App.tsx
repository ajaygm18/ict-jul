import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  useTheme,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  AccountBalance as AccountBalanceIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayArrowIcon,
  Notifications as NotificationsIcon,
  Assessment as AssessmentIcon,
  ShowChart as ShowChartIcon,
} from '@mui/icons-material';

// Import components
import Dashboard from './components/Dashboard/Dashboard';
import StockAnalysis from './components/PatternAnalysis/StockAnalysis';
import TradingView from './components/Charts/TradingView';
import Portfolio from './components/Portfolio/Portfolio';
import StrategyBacktesting from './components/Backtesting/StrategyBacktesting';
import RealTimeAlerts from './components/Alerts/RealTimeAlerts';
import PerformanceAnalytics from './components/Analytics/PerformanceAnalytics';

const DRAWER_WIDTH = 240;

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const theme = useTheme();

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Stock Analysis', icon: <AnalyticsIcon />, path: '/analysis' },
    { text: 'Trading Charts', icon: <ShowChartIcon />, path: '/charts' },
    { text: 'Strategy Backtesting', icon: <PlayArrowIcon />, path: '/backtesting' },
    { text: 'Performance Analytics', icon: <AssessmentIcon />, path: '/analytics' },
    { text: 'Real-Time Alerts', icon: <NotificationsIcon />, path: '/alerts' },
    { text: 'Portfolio', icon: <AccountBalanceIcon />, path: '/portfolio' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  const drawer = (
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          ICT Trader
        </Typography>
      </Toolbar>
      <List>
        {menuItems.map((item) => (
          <ListItem button key={item.text} component="a" href={item.path}>
            <ListItemIcon sx={{ color: theme.palette.text.primary }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { sm: `${DRAWER_WIDTH}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            ICT Stock Trading AI Agent
          </Typography>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: DRAWER_WIDTH }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
        }}
      >
        <Toolbar />
        <Container maxWidth="xl">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analysis" element={<StockAnalysis />} />
            <Route path="/charts" element={<TradingView />} />
            <Route path="/backtesting" element={<StrategyBacktesting />} />
            <Route path="/analytics" element={<PerformanceAnalytics />} />
            <Route path="/alerts" element={<RealTimeAlerts />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/settings" element={<div>Settings Coming Soon</div>} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App;
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert,
  Switch,
  FormControlLabel,
  FormGroup,
  Divider,
  IconButton,
  Badge,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
} from '@mui/material';
import {
  Notifications,
  NotificationImportant,
  TrendingUp,
  TrendingDown,
  Timeline,
  Speed,
  Delete,
  Add,
  VolumeUp,
} from '@mui/icons-material';

interface Alert {
  id: string;
  type: 'pattern' | 'price' | 'volume' | 'strategy';
  symbol: string;
  title: string;
  message: string;
  timestamp: Date;
  priority: 'low' | 'medium' | 'high';
  isRead: boolean;
  data?: any;
}

interface AlertRule {
  id: string;
  name: string;
  symbol: string;
  type: 'pattern' | 'price' | 'volume' | 'strategy';
  condition: string;
  value?: number;
  isActive: boolean;
  soundEnabled: boolean;
  patterns?: string[];
}

const RealTimeAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([
    {
      id: '1',
      name: 'AAPL Order Block Formation',
      symbol: 'AAPL',
      type: 'pattern',
      condition: 'Order Block Detected',
      isActive: true,
      soundEnabled: true,
      patterns: ['order_block', 'breaker_block'],
    },
    {
      id: '2',
      name: 'Silver Bullet Window',
      symbol: 'SPY',
      type: 'strategy',
      condition: 'Silver Bullet Setup',
      isActive: true,
      soundEnabled: false,
    },
    {
      id: '3',
      name: 'TSLA Price Alert',
      symbol: 'TSLA',
      type: 'price',
      condition: 'Price Above',
      value: 250,
      isActive: true,
      soundEnabled: true,
    },
  ]);

  const [alertSettings, setAlertSettings] = useState({
    soundEnabled: true,
    browserNotifications: true,
    patternAlerts: true,
    priceAlerts: true,
    volumeAlerts: true,
    strategyAlerts: true,
  });

  const [openDialog, setOpenDialog] = useState(false);
  const [newRule, setNewRule] = useState<Partial<AlertRule>>({
    type: 'pattern',
    isActive: true,
    soundEnabled: false,
  });

  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as any });

  // Simulate real-time alerts
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate incoming alerts based on active rules
      const activeRules = alertRules.filter(rule => rule.isActive);
      
      if (activeRules.length > 0 && Math.random() > 0.8) {
        const randomRule = activeRules[Math.floor(Math.random() * activeRules.length)];
        const newAlert: Alert = {
          id: `alert_${Date.now()}`,
          type: randomRule.type,
          symbol: randomRule.symbol,
          title: randomRule.name,
          message: generateAlertMessage(randomRule),
          timestamp: new Date(),
          priority: getPriority(randomRule.type),
          isRead: false,
          data: generateMockData(randomRule.type),
        };

        setAlerts(prev => [newAlert, ...prev.slice(0, 19)]); // Keep last 20 alerts

        // Play sound if enabled
        if (alertSettings.soundEnabled && randomRule.soundEnabled) {
          playAlertSound();
        }

        // Show browser notification
        if (alertSettings.browserNotifications && 'Notification' in window) {
          new Notification(`ICT Alert: ${newAlert.title}`, {
            body: newAlert.message,
            icon: '/favicon.ico',
          });
        }
      }
    }, 5000); // Check every 5 seconds

    return () => clearInterval(interval);
  }, [alertRules, alertSettings]);

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  const generateAlertMessage = (rule: AlertRule): string => {
    switch (rule.type) {
      case 'pattern':
        return `ICT pattern detected: ${rule.condition} on ${rule.symbol}`;
      case 'strategy':
        return `Strategy setup: ${rule.condition} triggered on ${rule.symbol}`;
      case 'price':
        return `Price alert: ${rule.symbol} ${rule.condition} $${rule.value}`;
      case 'volume':
        return `Volume spike detected on ${rule.symbol}`;
      default:
        return `Alert triggered for ${rule.symbol}`;
    }
  };

  const getPriority = (type: string): 'low' | 'medium' | 'high' => {
    switch (type) {
      case 'strategy':
        return 'high';
      case 'pattern':
        return 'medium';
      default:
        return 'low';
    }
  };

  const generateMockData = (type: string) => {
    switch (type) {
      case 'pattern':
        return {
          patternType: 'Order Block',
          confidence: Math.random() * 0.3 + 0.7,
          timeframe: '15m',
          price: Math.random() * 200 + 100,
        };
      case 'strategy':
        return {
          strategy: 'Silver Bullet',
          entryPrice: Math.random() * 200 + 100,
          stopLoss: Math.random() * 190 + 95,
          takeProfit: Math.random() * 210 + 105,
          riskReward: Math.random() * 2 + 1,
        };
      case 'price':
        return {
          currentPrice: Math.random() * 200 + 100,
          change: Math.random() * 10 - 5,
          changePercent: Math.random() * 5 - 2.5,
        };
      default:
        return {};
    }
  };

  const playAlertSound = () => {
    // In a real implementation, you'd play an actual sound file
    if (alertSettings.soundEnabled) {
      const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+P1mmkeDDOMz/LNdycEJ4nK8dCALAUKbbnvx5I9CJST6+V2NwwWksH13dBaWA6+' );
      audio.play().catch(() => {
        // Ignore autoplay restrictions
      });
    }
  };

  const markAsRead = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, isRead: true } : alert
    ));
  };

  const deleteAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const toggleRule = (ruleId: string) => {
    setAlertRules(prev => prev.map(rule =>
      rule.id === ruleId ? { ...rule, isActive: !rule.isActive } : rule
    ));
  };

  const deleteRule = (ruleId: string) => {
    setAlertRules(prev => prev.filter(rule => rule.id !== ruleId));
  };

  const addNewRule = () => {
    if (!newRule.name || !newRule.symbol) {
      setSnackbar({
        open: true,
        message: 'Please fill in all required fields',
        severity: 'error',
      });
      return;
    }

    const rule: AlertRule = {
      id: `rule_${Date.now()}`,
      name: newRule.name!,
      symbol: newRule.symbol!.toUpperCase(),
      type: newRule.type!,
      condition: newRule.condition || 'Pattern Detected',
      value: newRule.value,
      isActive: newRule.isActive!,
      soundEnabled: newRule.soundEnabled!,
      patterns: newRule.patterns,
    };

    setAlertRules(prev => [...prev, rule]);
    setOpenDialog(false);
    setNewRule({ type: 'pattern', isActive: true, soundEnabled: false });
    setSnackbar({
      open: true,
      message: 'Alert rule added successfully',
      severity: 'success',
    });
  };

  const getAlertIcon = (alert: Alert) => {
    switch (alert.type) {
      case 'pattern':
        return <Timeline color={alert.priority === 'high' ? 'error' : 'primary'} />;
      case 'strategy':
        return <NotificationImportant color="error" />;
      case 'price':
        return alert.data?.change > 0 ? <TrendingUp color="success" /> : <TrendingDown color="error" />;
      case 'volume':
        return <Speed color="warning" />;
      default:
        return <Notifications />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      default:
        return 'default';
    }
  };

  const unreadCount = alerts.filter(alert => !alert.isRead).length;

  return (
    <Box>
      <Box display="flex" alignItems="center" justifyContent="between" mb={3}>
        <Typography variant="h4" gutterBottom>
          ICT Real-Time Alerts
        </Typography>
        <Badge badgeContent={unreadCount} color="error">
          <Notifications fontSize="large" />
        </Badge>
      </Box>

      <Grid container spacing={3}>
        {/* Alert Settings */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Alert Settings
              </Typography>
              
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={alertSettings.soundEnabled}
                      onChange={(e) => setAlertSettings(prev => ({ 
                        ...prev, 
                        soundEnabled: e.target.checked 
                      }))}
                    />
                  }
                  label="Sound Alerts"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={alertSettings.browserNotifications}
                      onChange={(e) => setAlertSettings(prev => ({ 
                        ...prev, 
                        browserNotifications: e.target.checked 
                      }))}
                    />
                  }
                  label="Browser Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={alertSettings.patternAlerts}
                      onChange={(e) => setAlertSettings(prev => ({ 
                        ...prev, 
                        patternAlerts: e.target.checked 
                      }))}
                    />
                  }
                  label="ICT Pattern Alerts"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={alertSettings.strategyAlerts}
                      onChange={(e) => setAlertSettings(prev => ({ 
                        ...prev, 
                        strategyAlerts: e.target.checked 
                      }))}
                    />
                  }
                  label="Strategy Alerts"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={alertSettings.priceAlerts}
                      onChange={(e) => setAlertSettings(prev => ({ 
                        ...prev, 
                        priceAlerts: e.target.checked 
                      }))}
                    />
                  }
                  label="Price Alerts"
                />
              </FormGroup>
            </CardContent>
          </Card>

          {/* Alert Rules */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
                <Typography variant="h6">
                  Alert Rules
                </Typography>
                <Button
                  startIcon={<Add />}
                  onClick={() => setOpenDialog(true)}
                  size="small"
                >
                  Add Rule
                </Button>
              </Box>

              <List>
                {alertRules.map((rule) => (
                  <ListItem key={rule.id}>
                    <ListItemIcon>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={rule.isActive}
                            onChange={() => toggleRule(rule.id)}
                            size="small"
                          />
                        }
                        label=""
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={rule.name}
                      secondary={`${rule.symbol} - ${rule.condition}`}
                    />
                    <Box display="flex" alignItems="center">
                      {rule.soundEnabled && <VolumeUp fontSize="small" sx={{ mr: 1 }} />}
                      <IconButton size="small" onClick={() => deleteRule(rule.id)}>
                        <Delete fontSize="small" />
                      </IconButton>
                    </Box>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Live Alerts */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live Alerts ({alerts.length})
              </Typography>
              
              {alerts.length === 0 ? (
                <Alert severity="info">
                  No alerts yet. Configure alert rules to start receiving notifications.
                </Alert>
              ) : (
                <List>
                  {alerts.map((alert) => (
                    <React.Fragment key={alert.id}>
                      <ListItem
                        sx={{
                          backgroundColor: alert.isRead ? 'transparent' : 'action.hover',
                          borderRadius: 1,
                          mb: 1,
                        }}
                        onClick={() => markAsRead(alert.id)}
                      >
                        <ListItemIcon>
                          {getAlertIcon(alert)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box display="flex" alignItems="center" gap={1}>
                              <Typography variant="subtitle2">
                                {alert.title}
                              </Typography>
                              <Chip
                                label={alert.symbol}
                                size="small"
                                variant="outlined"
                              />
                              <Chip
                                label={alert.priority}
                                size="small"
                                color={getPriorityColor(alert.priority) as any}
                                variant="outlined"
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="textSecondary">
                                {alert.message}
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {alert.timestamp.toLocaleTimeString()}
                              </Typography>
                            </Box>
                          }
                        />
                        <Box display="flex" alignItems="center">
                          {!alert.isRead && (
                            <Chip label="New" size="small" color="primary" sx={{ mr: 1 }} />
                          )}
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteAlert(alert.id);
                            }}
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </Box>
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Add Rule Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add New Alert Rule</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                label="Rule Name"
                value={newRule.name || ''}
                onChange={(e) => setNewRule(prev => ({ ...prev, name: e.target.value }))}
                fullWidth
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="Symbol"
                value={newRule.symbol || ''}
                onChange={(e) => setNewRule(prev => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
                fullWidth
                required
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Alert Type</InputLabel>
                <Select
                  value={newRule.type}
                  label="Alert Type"
                  onChange={(e) => setNewRule(prev => ({ ...prev, type: e.target.value as any }))}
                >
                  <MenuItem value="pattern">ICT Pattern</MenuItem>
                  <MenuItem value="strategy">Strategy Setup</MenuItem>
                  <MenuItem value="price">Price Alert</MenuItem>
                  <MenuItem value="volume">Volume Alert</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Condition"
                value={newRule.condition || ''}
                onChange={(e) => setNewRule(prev => ({ ...prev, condition: e.target.value }))}
                fullWidth
                placeholder="e.g., Order Block Detected, Price Above, etc."
              />
            </Grid>
            {newRule.type === 'price' && (
              <Grid item xs={12}>
                <TextField
                  label="Price Value"
                  type="number"
                  value={newRule.value || ''}
                  onChange={(e) => setNewRule(prev => ({ ...prev, value: Number(e.target.value) }))}
                  fullWidth
                />
              </Grid>
            )}
            <Grid item xs={12}>
              <FormGroup row>
                <FormControlLabel
                  control={
                    <Switch
                      checked={newRule.isActive}
                      onChange={(e) => setNewRule(prev => ({ ...prev, isActive: e.target.checked }))}
                    />
                  }
                  label="Active"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={newRule.soundEnabled}
                      onChange={(e) => setNewRule(prev => ({ ...prev, soundEnabled: e.target.checked }))}
                    />
                  }
                  label="Sound Alert"
                />
              </FormGroup>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={addNewRule} variant="contained">Add Rule</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default RealTimeAlerts;
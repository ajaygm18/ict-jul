import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
} from '@mui/material';
import { AccountBalance } from '@mui/icons-material';

const Portfolio: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        ICT Portfolio Management
      </Typography>
      
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <AccountBalance sx={{ mr: 2, fontSize: 40 }} />
            <Box>
              <Typography variant="h5" gutterBottom>
                Portfolio Tracking Coming Soon
              </Typography>
              <Typography variant="body1" color="textSecondary">
                ICT-based portfolio management and performance tracking
              </Typography>
            </Box>
          </Box>
          
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Planned Portfolio Features:
            </Typography>
            <ul>
              <li>ICT strategy performance tracking</li>
              <li>Trade journal with ICT concept tagging</li>
              <li>Risk management dashboard</li>
              <li>Position sizing calculator</li>
              <li>Drawdown analysis and controls</li>
              <li>Strategy backtesting results</li>
              <li>Performance analytics by ICT concept</li>
              <li>Risk/reward optimization</li>
              <li>Compounding models and projections</li>
            </ul>
          </Alert>
          
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
            This section will integrate with the ICT Risk Management concepts (31-39) 
            to provide comprehensive portfolio and risk management capabilities.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Portfolio;
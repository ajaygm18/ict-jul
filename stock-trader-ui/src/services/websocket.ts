import React from 'react';
import { create } from 'zustand';

// WebSocket connection states
export enum WebSocketState {
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  ERROR = 'ERROR',
}

// Message types for WebSocket communication
export enum MessageType {
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  STOCK_DATA = 'stock_data',
  ICT_PATTERN = 'ict_pattern',
  MARKET_UPDATE = 'market_update',
  ALERT = 'alert',
  HEARTBEAT = 'heartbeat',
}

interface WebSocketMessage {
  type: MessageType;
  data?: any;
  timestamp?: number;
}

interface Subscription {
  symbol: string;
  type: 'stock_data' | 'ict_patterns' | 'alerts';
  timeframe?: string;
}

interface WebSocketStore {
  // Connection state
  state: WebSocketState;
  socket: WebSocket | null;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  
  // Subscriptions
  subscriptions: Subscription[];
  
  // Data
  stockData: Record<string, any>;
  ictPatterns: Record<string, any[]>;
  alerts: any[];
  marketData: any;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  subscribe: (subscription: Subscription) => void;
  unsubscribe: (symbol: string, type: string) => void;
  sendMessage: (message: WebSocketMessage) => void;
  
  // Internal
  setConnectionState: (state: WebSocketState) => void;
  handleMessage: (event: MessageEvent) => void;
  handleError: (event: Event) => void;
  handleClose: (event: CloseEvent) => void;
  reconnect: () => void;
}

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export const useWebSocketStore = create<WebSocketStore>((set, get) => ({
  // Initial state
  state: WebSocketState.DISCONNECTED,
  socket: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 5,
  subscriptions: [],
  stockData: {},
  ictPatterns: {},
  alerts: [],
  marketData: {},

  // Connect to WebSocket
  connect: () => {
    const { socket, state } = get();
    
    if (socket && state === WebSocketState.CONNECTED) {
      return; // Already connected
    }

    set({ state: WebSocketState.CONNECTING });

    try {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        set({ 
          socket: ws, 
          state: WebSocketState.CONNECTED, 
          reconnectAttempts: 0 
        });
        
        // Resubscribe to existing subscriptions
        const { subscriptions } = get();
        subscriptions.forEach(subscription => {
          get().sendMessage({
            type: MessageType.SUBSCRIBE,
            data: subscription,
          });
        });
      };

      ws.onmessage = get().handleMessage;
      ws.onerror = get().handleError;
      ws.onclose = get().handleClose;

      set({ socket: ws });
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      set({ state: WebSocketState.ERROR });
    }
  },

  // Disconnect from WebSocket
  disconnect: () => {
    const { socket } = get();
    
    if (socket) {
      socket.close(1000, 'User disconnected');
      set({ socket: null, state: WebSocketState.DISCONNECTED });
    }
  },

  // Subscribe to data updates
  subscribe: (subscription: Subscription) => {
    const { subscriptions, socket, state } = get();
    
    // Check if already subscribed
    const exists = subscriptions.find(
      sub => sub.symbol === subscription.symbol && sub.type === subscription.type
    );
    
    if (exists) {
      return;
    }

    // Add to subscriptions
    set({ subscriptions: [...subscriptions, subscription] });

    // Send subscription message if connected
    if (socket && state === WebSocketState.CONNECTED) {
      get().sendMessage({
        type: MessageType.SUBSCRIBE,
        data: subscription,
      });
    }
  },

  // Unsubscribe from data updates
  unsubscribe: (symbol: string, type: string) => {
    const { subscriptions, socket, state } = get();
    
    const filteredSubscriptions = subscriptions.filter(
      sub => !(sub.symbol === symbol && sub.type === type)
    );
    
    set({ subscriptions: filteredSubscriptions });

    // Send unsubscribe message if connected
    if (socket && state === WebSocketState.CONNECTED) {
      get().sendMessage({
        type: MessageType.UNSUBSCRIBE,
        data: { symbol, type },
      });
    }
  },

  // Send message to WebSocket
  sendMessage: (message: WebSocketMessage) => {
    const { socket, state } = get();
    
    if (socket && state === WebSocketState.CONNECTED) {
      const messageWithTimestamp = {
        ...message,
        timestamp: Date.now(),
      };
      
      socket.send(JSON.stringify(messageWithTimestamp));
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  },

  // Set connection state
  setConnectionState: (state: WebSocketState) => {
    set({ state });
  },

  // Handle incoming messages
  handleMessage: (event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      switch (message.type) {
        case MessageType.STOCK_DATA:
          set(state => ({
            stockData: {
              ...state.stockData,
              [message.data.symbol]: message.data,
            },
          }));
          break;

        case MessageType.ICT_PATTERN:
          set(state => ({
            ictPatterns: {
              ...state.ictPatterns,
              [message.data.symbol]: [
                ...(state.ictPatterns[message.data.symbol] || []),
                message.data.pattern,
              ].slice(-50), // Keep last 50 patterns
            },
          }));
          break;

        case MessageType.MARKET_UPDATE:
          set({ marketData: message.data });
          break;

        case MessageType.ALERT:
          set(state => ({
            alerts: [message.data, ...state.alerts].slice(0, 100), // Keep last 100 alerts
          }));
          break;

        case MessageType.HEARTBEAT:
          // Respond to heartbeat
          get().sendMessage({ type: MessageType.HEARTBEAT });
          break;

        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  },

  // Handle WebSocket errors
  handleError: (event: Event) => {
    console.error('WebSocket error:', event);
    set({ state: WebSocketState.ERROR });
  },

  // Handle WebSocket close
  handleClose: (event: CloseEvent) => {
    console.log('WebSocket closed:', event.code, event.reason);
    set({ socket: null, state: WebSocketState.DISCONNECTED });
    
    // Attempt to reconnect if not manually closed
    if (event.code !== 1000 && get().reconnectAttempts < get().maxReconnectAttempts) {
      get().reconnect();
    }
  },

  // Reconnect to WebSocket
  reconnect: () => {
    const { reconnectAttempts, maxReconnectAttempts } = get();
    
    if (reconnectAttempts >= maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Exponential backoff, max 30s
    
    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
    
    set({ reconnectAttempts: reconnectAttempts + 1 });
    
    setTimeout(() => {
      get().connect();
    }, delay);
  },
}));

// Hook for easy component usage
export const useWebSocket = () => {
  const store = useWebSocketStore();
  
  return {
    // State
    connectionState: store.state,
    isConnected: store.state === WebSocketState.CONNECTED,
    stockData: store.stockData,
    ictPatterns: store.ictPatterns,
    alerts: store.alerts,
    marketData: store.marketData,
    
    // Actions
    connect: store.connect,
    disconnect: store.disconnect,
    subscribe: store.subscribe,
    unsubscribe: store.unsubscribe,
    
    // Utilities
    getStockData: (symbol: string) => store.stockData[symbol],
    getICTPatterns: (symbol: string) => store.ictPatterns[symbol] || [],
  };
};

// Auto-connect hook for components that need real-time data
export const useAutoWebSocket = (subscriptions: Subscription[] = []) => {
  const webSocket = useWebSocket();
  
  React.useEffect(() => {
    // Connect when component mounts
    webSocket.connect();
    
    // Subscribe to requested data
    subscriptions.forEach(subscription => {
      webSocket.subscribe(subscription);
    });
    
    // Cleanup on unmount
    return () => {
      subscriptions.forEach(subscription => {
        webSocket.unsubscribe(subscription.symbol, subscription.type);
      });
    };
  }, []);
  
  return webSocket;
};
# Advanced Strategy Features

This document explains the advanced features that can be used in strategy JSON files.

## Automatic Take Profit Handling

The system automatically determines whether to use full take profit or partial take profit based on the number of TP levels specified:

- **Single TP Level**: Treated as full take profit (entire position exits at that level)
- **Multiple TP Levels**: Automatically converted to partial take profit with equal distribution

## Dynamic Position Sizing

### `amount`
- **Type**: String expression
- **Description**: Defines a dynamic position sizing formula based on market conditions
- **Example**: `"FIXED_AMOUNT * (1 / (ATR14 / PRICE)) * (RSI14 / 50)"` - Larger positions when volatility is low and momentum is strong
- **Available Variables**: `FIXED_AMOUNT`, `PRICE`, `ATR14`, `RSI14`, `VOLUME`, `VMA20`, and any other technical indicator

##

## Average Down/Up Features

### `avg_down`
- **Type**: String expression or Array of string expressions
- **Description**: Defines the price level(s) at which the strategy will automatically add to the position (average down)
- **Example (single)**: `"ENTRY_PRICE * 0.95"` - Averages down when price drops 5% from entry price
- **Example (multiple)**: `["ENTRY_PRICE * 0.98", "ENTRY_PRICE * 0.95", "ENTRY_PRICE * 0.92"]` - Multiple averaging down opportunities
- **Limit**: Maximum number of thresholds specified in array

### `avg_up`
- **Type**: String expression or Array of string expressions
- **Description**: Defines the price level(s) at which the strategy will automatically add to the position (average up)
- **Example (single)**: `"ENTRY_PRICE * 1.03"` - Averages up when price rises 3% from entry price
- **Example (multiple)**: `["ENTRY_PRICE * 1.02", "ENTRY_PRICE * 1.04", "ENTRY_PRICE * 1.06"]` - Multiple averaging up opportunities
- **Limit**: Maximum number of thresholds specified in array

## Partial Take Profit

### `partial_tp`
- **Type**: Array of string expressions (optional)
- **Description**: Defines multiple take profit levels for partial exits
- **Example**: `["ENTRY_PRICE * 1.02", "ENTRY_PRICE * 1.05", "ENTRY_PRICE * 1.08"]`
- **Note**: Can be omitted if using automatic TP handling. Also, if `tp` array has multiple levels, they will be automatically converted to partial TP.

### `partial_tp_ratios`
- **Type**: Array of floats (optional)
- **Description**: Defines the portion of position to exit at each partial TP level
- **Example**: `[0.3, 0.3, 0.4]` - Exit 30% at first TP, 30% at second TP, 40% at third TP
- **Note**: If omitted, position will be distributed equally across all partial TP levels

## Example Strategy

### Full Take Profit (Single TP Level)
```json
{
  "name": "FULL_TP_STRATEGY",
  "entry": [
    "RSI14 < 30",
    "VOLUME > VMA20"
  ],
  "tp": [
    "PRICE + (ATR14 * 3)"
  ],
  "sl": [
    "PRICE - (ATR14 * 2)"
  ],
  "max_hold_days": 10
}
```
In this example, since there's only one TP level, the entire position exits at that level.

### Partial Take Profit (Multiple TP Levels)
```json
{
  "name": "PARTIAL_TP_STRATEGY",
  "entry": [
    "RSI14 < 30",
    "VOLUME > VMA20"
  ],
  "tp": [
    "ENTRY_PRICE * 1.01",
    "ENTRY_PRICE * 1.02",
    "ENTRY_PRICE * 1.03"
  ],
  "sl": [
    "PRICE - (ATR14 * 2)"
  ],
  "max_hold_days": 10
}
```
In this example, since there are multiple TP levels, they are automatically converted to partial TP with equal distribution (33.33% at each level).

### Multi-Level Averaging Strategy
```json
{
  "name": "MULTI_AVG_STRATEGY",
  "entry": [
    "RSI14 < 30",
    "VOLUME > VMA20"
  ],
  "tp": [
    "PRICE + (ATR14 * 3)"
  ],
  "sl": [
    "PRICE - (ATR14 * 2)"
  ],
  "avg_down": [
    "ENTRY_PRICE * 0.98",
    "ENTRY_PRICE * 0.95",
    "ENTRY_PRICE * 0.92"
  ],
  "avg_up": [
    "ENTRY_PRICE * 1.02",
    "ENTRY_PRICE * 1.04",
    "ENTRY_PRICE * 1.06"
  ],
  "max_hold_days": 10
}
```
In this example, the strategy will average down at three different price levels and average up at three different price levels, providing fine-grained position management.

### Dynamic Amount Strategy
```json
{
  "name": "DYNAMIC_AMOUNT_STRATEGY",
  "entry": [
    "RSI14 < 30",
    "VOLUME > VMA20"
  ],
  "tp": [
    "PRICE + (ATR14 * 3)"
  ],
  "sl": [
    "PRICE - (ATR14 * 2)"
  ],
  "amount": "FIXED_AMOUNT * (1 / (ATR14 / PRICE)) * (RSI14 / 50)",
  "ts": "2%",
  "max_hold_days": 10
}
```
In this example, the strategy will use dynamic position sizing based on volatility and momentum, taking larger positions when volatility is low and momentum is strong, with a 2% trailing stop for risk management.

## Stop Loss

### `sl`
- **Type**: String expression or Array of string expressions
- **Description**: Defines the price level(s) at which to exit a losing position
- **Example (single)**: `"PRICE - (ATR14 * 2)"` - Exit when price drops 2x ATR below entry
- **Example (multiple)**: `["ENTRY_PRICE * 0.98", "ENTRY_PRICE * 0.95", "ENTRY_PRICE * 0.90"]` - Multiple stop loss levels for progressive risk management

## Trailing Stop

### `ts`
- **Type**: String (percentage format)
- **Description**: Dynamic stop loss that follows price upward and triggers when price drops from peak
- **Example**: `"2%"` - Exit when price drops 2% from highest price reached since entry
- **Priority**: Takes precedence over regular stop loss levels when configured

## Available Variables in Expressions

The following variables can be used in threshold expressions:
- `PRICE` or `ENTRY_PRICE` - Entry price
- `ATR14` - Average True Range (14 period)
- `FIXED_AMOUNT` - Fixed position size (1,000,000 IDR)
- `RSI14` - Relative Strength Index (14 period)
- `VOLUME` - Current trading volume
- `VMA20` - 20-period volume moving average
- `HIGH` - Current bar high price (available for trailing stop calculations)
- `PERCENT_SPIKE` - Number of times price increased >= 1% in the last 30 days
- `VOLUME_SPIKE` - Current volume divided by 30-day average volume
- `RVOL` - Relative Volume (current volume divided by 10-day average volume)
- Any technical indicator from the engine (MA20, RSI14, etc.)

## Spike Indicators

### `PERCENT_SPIKE`
- **Type**: Integer
- **Description**: Number of times the stock price increased by >= 1% in the last 30 days
- **Example**: `"PERCENT_SPIKE > 0"` - Stock has shown strong positive moves recently

### `VOLUME_SPIKE`
- **Type**: Float
- **Description**: Current trading volume divided by the 30-day average volume
- **Example**: `"VOLUME_SPIKE > 3"` - Today's volume is 3x the 30-day average

### `RVOL`
- **Type**: Float
- **Description**: Relative Volume - Current trading volume divided by the 10-day average volume
- **Example**: `"RVOL > 2"` - Today's volume is 2x the 10-day average

## How It Works

1. **Entry**: Standard entry signal based on configured conditions
2. **Position Sizing**: Calculates position size based on dynamic amount formula
3. **Monitoring**: Continuously checks for averaging opportunities, partial TP levels, trailing stops, and multiple SL levels
4. **Averaging Down**: When price drops below threshold, uses dynamic amount formula for additional position size
5. **Averaging Up**: When price rises above threshold, uses dynamic amount formula for additional position size
6. **Partial TP**: As price reaches each TP level, exits specified portion of position
7. **Trailing Stop**: Dynamically adjusts stop loss level as price moves in favor, exits when price drops from peak
8. **Multiple SL**: Checks multiple stop loss levels and exits at the first breached level
9. **Exit**: Final exit when remaining position hits TP, trailing stop, any SL level, or max hold days reached
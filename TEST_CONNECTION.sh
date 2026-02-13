#!/bin/bash
# Connection Test Script for Autocar
# Run this on the autocar to test connectivity to PC

PC_IP="192.168.56.1"
PC_PORT="5000"

echo "=========================================="
echo "Testing Connection to PC"
echo "=========================================="
echo "PC IP: $PC_IP"
echo "PC Port: $PC_PORT"
echo ""

# Test 1: Ping
echo "Test 1: Ping to PC..."
if ping -c 3 $PC_IP > /dev/null 2>&1; then
    echo "✓ Ping successful"
else
    echo "✗ Ping failed - PC is not reachable"
    echo "  Check: Are PC and autocar on the same network?"
    exit 1
fi
echo ""

# Test 2: Port connectivity
echo "Test 2: Testing port $PC_PORT..."
if timeout 5 bash -c "echo > /dev/tcp/$PC_IP/$PC_PORT" 2>/dev/null; then
    echo "✓ Port $PC_PORT is open"
else
    echo "✗ Port $PC_PORT is closed or unreachable"
    echo "  Check: Is PC webhook receiver running?"
    echo "  Check: Is firewall blocking port $PC_PORT?"
    exit 1
fi
echo ""

# Test 3: HTTP endpoint
echo "Test 3: Testing webhook endpoint..."
response=$(curl -s -w "\n%{http_code}" --max-time 5 http://$PC_IP:$PC_PORT/health 2>/dev/null)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "200" ]; then
    echo "✓ Webhook endpoint is responding"
    echo "  Response: $body"
else
    echo "✗ Webhook endpoint failed (HTTP $http_code)"
    echo "  Check: Is PC webhook receiver running?"
    exit 1
fi
echo ""

echo "=========================================="
echo "All tests passed! Connection is working."
echo "=========================================="

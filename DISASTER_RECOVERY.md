# 🔧 Pragmatic Disaster Recovery & Operations Guide

## Overview

This is a **practical, no-nonsense** disaster recovery plan for a small team. No enterprise complexity, just what you actually need to keep the service running.

## 🎯 Recovery Targets (Realistic)

- **RTO (Recovery Time Objective)**: 2 hours
- **RPO (Recovery Point Objective)**: 24 hours
- **Availability Target**: 99.5% (allows ~3.5 hours downtime/month)

## 📊 System Architecture

```
[Load Balancer (nginx)]
       ↓
[API Servers x2] ←→ [Redis Cache] ←→ [PostgreSQL]
       ↓
[CV Service]
```

## 🚨 Common Failure Scenarios & Solutions

### 1. API Server Crash

**Symptoms:**
- 502 errors from nginx
- Health check failures

**Quick Fix:**
```bash
# Check server status
docker-compose ps

# Restart failed container
docker-compose restart api1  # or api2

# If persistent, check logs
docker-compose logs --tail=100 api1

# Scale up if needed
docker-compose up -d --scale api1=2
```

### 2. Database Connection Exhaustion

**Symptoms:**
- "too many connections" errors
- Slow API responses

**Quick Fix:**
```bash
# Check connection count
docker exec -it postgres psql -U simgen -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
docker exec -it postgres psql -U simgen -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE state = 'idle'
  AND state_change < now() - interval '10 minutes';"

# Restart API servers to reset connection pools
docker-compose restart api1 api2
```

### 3. Redis Memory Full

**Symptoms:**
- "OOM" errors in Redis logs
- Cache misses increase

**Quick Fix:**
```bash
# Check memory usage
docker exec -it redis redis-cli INFO memory

# Clear cache if needed
docker exec -it redis redis-cli FLUSHDB

# Or selectively remove old keys
docker exec -it redis redis-cli --scan --pattern "cv:*" | xargs docker exec -it redis redis-cli DEL
```

### 4. WebSocket Sessions Stuck

**Symptoms:**
- Clients can't connect
- "Session limit reached" errors

**Quick Fix:**
```bash
# Check session count
curl http://localhost/health/detailed | jq '.websocket_sessions'

# Clear stale sessions from Redis
docker exec -it redis redis-cli DEL "sessions:all"

# Restart WebSocket managers
docker-compose restart api1 api2
```

### 5. CV Service Overload

**Symptoms:**
- Timeouts on sketch analysis
- High memory usage

**Quick Fix:**
```bash
# Check CV service health
curl http://localhost:8001/health

# Restart with memory limit
docker-compose stop cv_service
docker update --memory="4g" --memory-swap="4g" cv_service
docker-compose start cv_service

# Or scale horizontally
docker-compose up -d --scale cv_service=2
```

## 📝 Backup Procedures

### Daily Backup Script

```bash
#!/bin/bash
# backup.sh - Run daily at 2 AM via cron

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 1. Database backup
docker exec postgres pg_dump -U simgen simgen | gzip > $BACKUP_DIR/db.sql.gz

# 2. Redis snapshot (optional, for cache warming)
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb $BACKUP_DIR/redis.rdb

# 3. Configuration files
tar czf $BACKUP_DIR/config.tar.gz \
  .env \
  docker-compose.yml \
  nginx/

# 4. Keep only last 7 days
find /backups -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR"
```

### Quick Restore

```bash
#!/bin/bash
# restore.sh - Restore from backup

BACKUP_DATE=$1  # Format: YYYYMMDD

# 1. Stop services
docker-compose down

# 2. Restore database
gunzip < /backups/$BACKUP_DATE/db.sql.gz | \
  docker exec -i postgres psql -U simgen simgen

# 3. Restore Redis (optional)
docker cp /backups/$BACKUP_DATE/redis.rdb redis:/data/dump.rdb
docker exec redis redis-cli SHUTDOWN NOSAVE
docker-compose restart redis

# 4. Start services
docker-compose up -d

echo "Restore completed from $BACKUP_DATE"
```

## 🔍 Monitoring Checklist

### Every 5 Minutes (Automated)
```bash
# Simple monitoring script
#!/bin/bash

# Check health endpoints
for endpoint in health health/ready health/live; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/$endpoint)
  if [ $STATUS -ne 200 ]; then
    echo "ALERT: /$endpoint returned $STATUS"
    # Send alert (email, Slack, etc.)
  fi
done

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
  echo "ALERT: Disk usage at $DISK_USAGE%"
fi

# Check memory
MEM_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
if [ $MEM_USAGE -gt 85 ]; then
  echo "ALERT: Memory usage at $MEM_USAGE%"
fi
```

### Daily Checks (Manual)
- [ ] Review error logs: `docker-compose logs --since 24h | grep ERROR`
- [ ] Check backup completion
- [ ] Review resource usage trends
- [ ] Clear old logs: `find /var/log -name "*.log" -mtime +7 -delete`

## 🚀 Emergency Procedures

### Complete System Down

1. **Don't Panic** - Follow the steps

2. **Quick Diagnosis** (2 minutes)
```bash
# Is Docker running?
docker ps

# Is the network up?
ping 8.8.8.8

# Is disk full?
df -h

# Is memory exhausted?
free -h
```

3. **Nuclear Option** (5 minutes)
```bash
# Complete restart
docker-compose down
docker system prune -f
docker-compose up -d
```

4. **Rollback if Needed** (15 minutes)
```bash
# Revert to last known good version
git checkout [last-good-commit]
docker-compose build
docker-compose up -d
```

### Performance Degradation

1. **Identify Bottleneck**
```bash
# Check CPU
docker stats --no-stream

# Check slow queries
docker exec -it postgres psql -U simgen -c "
  SELECT query, calls, mean_exec_time
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 10;"

# Check Redis slow log
docker exec -it redis redis-cli SLOWLOG GET 10
```

2. **Quick Optimizations**
```bash
# Increase connection pool
docker exec -it api1 sh -c 'export DATABASE_POOL_SIZE=50'

# Clear caches
docker exec -it redis redis-cli FLUSHDB

# Restart with more resources
docker update --cpus="4" --memory="8g" api1
docker-compose restart api1
```

## 📋 Runbook for Common Tasks

### Deploying Updates (Zero Downtime)

```bash
# 1. Build new version
docker-compose build api1

# 2. Start new instance
docker-compose up -d --scale api1=2 --no-recreate

# 3. Wait for health check
sleep 30
curl http://localhost/health

# 4. Remove old instance
docker-compose stop api1
docker-compose rm api1
docker-compose up -d api1

# 5. Repeat for api2
```

### Adding a New API Server

```bash
# 1. Update docker-compose.yml
# Add api3 service configuration

# 2. Update nginx config
# Add api3:8000 to upstream

# 3. Deploy
docker-compose up -d api3
docker-compose restart nginx
```

### Emergency Cache Clear

```bash
# Clear everything (nuclear option)
docker exec -it redis redis-cli FLUSHALL

# Clear specific patterns
docker exec -it redis redis-cli --scan --pattern "cv:*" | \
  xargs docker exec -it redis redis-cli DEL

# Clear LLM cache only
docker exec -it redis redis-cli --scan --pattern "llm:*" | \
  xargs docker exec -it redis redis-cli DEL
```

## 🔐 Security Incident Response

### Suspected Breach

1. **Isolate immediately**
```bash
# Block external access
iptables -I INPUT -p tcp --dport 80 -j DROP
iptables -I INPUT -p tcp --dport 443 -j DROP
```

2. **Preserve evidence**
```bash
# Capture logs
docker-compose logs > incident_$(date +%s).log
```

3. **Rotate credentials**
```bash
# Change database password
docker exec -it postgres psql -U postgres -c "ALTER USER simgen PASSWORD 'new_password';"

# Update .env file
sed -i 's/old_password/new_password/g' .env

# Restart services
docker-compose down
docker-compose up -d
```

## 📞 Escalation Path

1. **Level 1**: On-call developer (handles 90% of issues)
   - Check health endpoints
   - Restart failed services
   - Clear caches

2. **Level 2**: Senior developer (handles complex issues)
   - Database problems
   - Performance tuning
   - Code rollbacks

3. **Level 3**: Team lead (major incidents)
   - Complete outages
   - Data loss scenarios
   - Security incidents

## 🎯 Key Metrics to Track

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| API Response Time | >500ms p95 | Check database queries |
| Error Rate | >1% | Check logs, scale up |
| Memory Usage | >85% | Restart services, add memory |
| Disk Usage | >80% | Clean logs, expand disk |
| Cache Hit Rate | <60% | Check cache configuration |
| WebSocket Sessions | >1000 | Scale horizontally |

## 📚 Useful Commands Reference

```bash
# View real-time logs
docker-compose logs -f api1

# Check container resource usage
docker stats

# Enter container shell
docker exec -it api1 /bin/sh

# Database query
docker exec -it postgres psql -U simgen -c "SELECT * FROM simulations LIMIT 10;"

# Redis monitor
docker exec -it redis redis-cli MONITOR

# Port check
netstat -tulpn | grep LISTEN

# Process tree
ps auxf

# Disk I/O
iotop

# Network connections
ss -tulwn
```

## ✅ Pre-Deployment Checklist

- [ ] Backups completed successfully
- [ ] Health endpoints responding
- [ ] Resource usage normal (<70%)
- [ ] No critical errors in last 24h
- [ ] Load test passed
- [ ] Rollback plan ready
- [ ] Team notified

## 🎮 Testing Disaster Recovery

**Monthly drill (30 minutes):**

1. **Simulate API failure**
```bash
docker-compose stop api1
# Verify nginx redirects to api2
curl http://localhost/health
docker-compose start api1
```

2. **Simulate database failure**
```bash
docker-compose stop postgres
# Check error handling
curl http://localhost/api/v1/test
docker-compose start postgres
```

3. **Restore from backup**
```bash
# Use staging environment
./restore.sh 20250115
# Verify data integrity
```

---

## Remember: Keep It Simple

- **Most problems** are solved by restarting the service
- **Second most common** is clearing the cache
- **Third** is checking disk space
- **Everything else** is in the logs

When in doubt: `docker-compose restart` and check the logs.

**"It's not DNS. There's no way it's DNS. It was DNS."** - Every DevOps Engineer
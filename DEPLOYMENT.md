# SimGen AI Production Deployment Guide

Complete guide for deploying SimGen AI in production environments with Docker containers.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd simulation-mujoco
   ```

2. **Configure Environment**
   ```bash
   cp .env.prod.template .env.prod
   # Edit .env.prod with your configuration
   ```

3. **Deploy**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **CPU**: 4+ cores recommended for optimal performance
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 50GB+ free space for Docker images and data
- **GPU**: NVIDIA GPU recommended for MuJoCo rendering (optional)

### Software Dependencies
- **Docker**: 20.10.0 or higher
- **Docker Compose**: 2.0.0 or higher
- **Git**: For cloning the repository
- **curl**: For health checks

Install Docker and Docker Compose:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## ⚙️ Configuration

### Environment Variables

Copy the template and fill in your values:
```bash
cp .env.prod.template .env.prod
```

**Required Variables:**
```env
# Database
DB_PASSWORD=your_secure_database_password_here

# Redis
REDIS_PASSWORD=your_secure_redis_password_here

# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Security
SECRET_KEY=your_super_secret_key_here_minimum_32_characters
API_KEYS=your_api_key_1,your_api_key_2,admin_key_here

# Domain Configuration
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
NEXT_PUBLIC_API_URL=https://yourdomain.com/api
NEXT_PUBLIC_WS_URL=wss://yourdomain.com/ws
```

### SSL Certificate Setup

For HTTPS, place your SSL certificates in the `ssl/` directory:
```bash
mkdir -p ssl
# Copy your certificates
cp /path/to/your/fullchain.pem ssl/
cp /path/to/your/privkey.pem ssl/
```

For development/testing, generate self-signed certificates:
```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/privkey.pem -out ssl/fullchain.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## 🚢 Deployment Options

### Option 1: Automated Deployment (Recommended)

Use the deployment script for a fully automated deployment:
```bash
chmod +x deploy.sh
./deploy.sh
```

The script will:
- Validate environment configuration
- Create necessary directories
- Backup existing data
- Build and deploy services
- Perform health checks
- Show deployment status

### Option 2: Manual Deployment

For more control over the deployment process:

1. **Create directories:**
   ```bash
   mkdir -p logs storage cache ssl
   ```

2. **Build services:**
   ```bash
   docker-compose -f docker-compose.prod.yml build
   ```

3. **Start databases:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d postgres redis
   ```

4. **Wait for databases (30 seconds):**
   ```bash
   sleep 30
   ```

5. **Start application:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d backend frontend nginx
   ```

6. **Optional: Start monitoring:**
   ```bash
   docker-compose -f docker-compose.prod.yml --profile monitoring up -d
   ```

### Option 3: Development with Production Config

For testing production configuration in development:
```bash
docker-compose -f docker-compose.prod.yml up
```

## 🔍 Monitoring and Health Checks

### Built-in Health Endpoints

- **Basic Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed`
- **Kubernetes Ready**: `GET /ready`
- **Kubernetes Live**: `GET /live`
- **System Metrics**: `GET /metrics` (requires authentication)

### Monitoring Stack (Optional)

Enable monitoring by setting `ENABLE_MONITORING=true` in your environment:

```env
ENABLE_MONITORING=true
GRAFANA_PASSWORD=your_grafana_admin_password
```

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090 (metrics collection)
- **Grafana**: http://localhost:3001 (dashboards and alerting)

### Service Status

Check deployment status:
```bash
./deploy.sh status
```

View logs:
```bash
# All services
./deploy.sh logs

# Specific service
./deploy.sh logs backend
```

## 🔧 Management Commands

### Deployment Management
```bash
# Deploy application
./deploy.sh deploy

# Check status
./deploy.sh status

# View logs
./deploy.sh logs [service_name]

# Restart services
./deploy.sh restart [service_name]

# Stop all services
./deploy.sh stop

# Rollback deployment
./deploy.sh rollback
```

### Docker Compose Commands
```bash
# View running services
docker-compose -f docker-compose.prod.yml ps

# Scale backend service
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# View resource usage
docker stats

# Access container shell
docker-compose -f docker-compose.prod.yml exec backend bash
```

## 🔒 Security Considerations

### Network Security
- All services run in isolated Docker network
- Only necessary ports are exposed
- Nginx acts as reverse proxy with security headers

### Database Security
- PostgreSQL uses SCRAM-SHA-256 authentication
- Database passwords are environment-variable based
- Data volumes are persistent and secure

### Application Security
- JWT-based authentication with secure secret keys
- Rate limiting on API endpoints
- Input validation and sanitization
- CORS protection with configurable origins

### SSL/TLS Configuration
- Modern TLS 1.2/1.3 protocols only
- Strong cipher suites
- HSTS headers enabled
- Self-signed certificates for development

## 📊 Performance Optimization

### Resource Limits
Services are configured with resource limits:
- **Backend**: 4GB RAM, 2 CPU cores
- **Frontend**: 1GB RAM, 0.5 CPU cores
- **PostgreSQL**: 2GB RAM, 1 CPU core
- **Redis**: 512MB RAM, 0.5 CPU cores

### Scaling Options

**Horizontal Scaling:**
```bash
# Scale backend replicas
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# Load balancer will distribute requests
```

**Vertical Scaling:**
Edit resource limits in `docker-compose.prod.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4.0'
```

### Database Optimization
- PostgreSQL configuration optimized for AI workloads
- Connection pooling enabled
- Automatic vacuum and analyze
- Performance monitoring enabled

## 🚨 Troubleshooting

### Common Issues

**1. Service won't start:**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs [service_name]

# Check environment variables
docker-compose -f docker-compose.prod.yml config
```

**2. Database connection errors:**
```bash
# Wait for database to be ready
sleep 30

# Check database health
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U simgen_user
```

**3. SSL certificate issues:**
```bash
# Verify certificate files
ls -la ssl/
openssl x509 -in ssl/fullchain.pem -text -noout
```

**4. Performance issues:**
```bash
# Check resource usage
docker stats

# Check system resources
free -h
df -h
```

### Log Locations

Application logs are stored in:
- **Application logs**: `./logs/`
- **Nginx logs**: `./logs/nginx/`
- **Container logs**: `docker-compose logs [service]`

### Health Check Failures

If health checks fail:
1. Check service logs
2. Verify environment variables
3. Ensure all required services are running
4. Check network connectivity

## 🔄 Backup and Recovery

### Automated Backups

The deployment script automatically creates backups before deployment:
```bash
# Backups are stored in
ls -la ./backups/
```

### Manual Backup

Create manual backup:
```bash
# Stop services
docker-compose -f docker-compose.prod.yml stop

# Create backup
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz postgres_data redis_data storage

# Restart services
docker-compose -f docker-compose.prod.yml start
```

### Recovery

Restore from backup:
```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data
tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

## 📈 Monitoring and Alerting

### Key Metrics to Monitor

1. **Application Health**
   - Response time
   - Error rate
   - Request throughput

2. **System Resources**
   - CPU usage
   - Memory usage
   - Disk space

3. **Database Performance**
   - Connection count
   - Query performance
   - Lock contention

4. **Cache Performance**
   - Hit rate
   - Memory usage
   - Eviction rate

### Setting Up Alerts

Configure alerts in Grafana or external monitoring:
- Response time > 5 seconds
- Error rate > 5%
- CPU usage > 80%
- Memory usage > 90%
- Disk usage > 85%

## 🌐 Production Checklist

Before going live:

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Domain name configured
- [ ] Database passwords changed from defaults
- [ ] API keys added to environment
- [ ] Health checks passing
- [ ] Monitoring enabled
- [ ] Backup strategy implemented
- [ ] Load testing completed
- [ ] Security review completed
- [ ] Documentation updated

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review application logs
3. Consult the monitoring dashboards
4. Create an issue in the repository

## 🔗 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
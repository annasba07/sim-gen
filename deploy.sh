#!/bin/bash

# SimGen AI Production Deployment Script
# Handles secure deployment with proper validation and rollback capabilities

set -euo pipefail

# Configuration
PROJECT_NAME="simgen-ai"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

# Create required directories
setup_directories() {
    log "Creating required directories..."
    mkdir -p logs storage cache ssl monitoring/grafana/{dashboards,datasources}
    chmod 755 logs storage cache
}

# Validate environment
validate_environment() {
    log "Validating environment configuration..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found. Copy .env.prod.template to .env.prod and configure it."
    fi
    
    # Check required environment variables
    source "$ENV_FILE"
    
    required_vars=(
        "DB_PASSWORD"
        "REDIS_PASSWORD" 
        "ANTHROPIC_API_KEY"
        "OPENAI_API_KEY"
        "SECRET_KEY"
        "API_KEYS"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set in $ENV_FILE"
        fi
    done
    
    success "Environment validation passed"
}

# Check Docker and Docker Compose
check_docker() {
    log "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi
    
    success "Docker environment ready"
}

# Backup existing data
backup_data() {
    if [[ -d "postgres_data" ]] || [[ -d "redis_data" ]]; then
        log "Creating backup of existing data..."
        
        backup_timestamp=$(date +%Y%m%d_%H%M%S)
        backup_path="$BACKUP_DIR/backup_$backup_timestamp"
        
        mkdir -p "$backup_path"
        
        if [[ -d "postgres_data" ]]; then
            cp -r postgres_data "$backup_path/"
            log "PostgreSQL data backed up to $backup_path/postgres_data"
        fi
        
        if [[ -d "redis_data" ]]; then
            cp -r redis_data "$backup_path/"
            log "Redis data backed up to $backup_path/redis_data"
        fi
        
        success "Data backup completed: $backup_path"
    else
        log "No existing data to backup"
    fi
}

# Build and deploy services
deploy_services() {
    log "Building and deploying services..."
    
    # Build images
    log "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build --no-cache
    
    # Start core services first
    log "Starting database services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres redis
    
    # Wait for databases to be ready
    log "Waiting for databases to be ready..."
    sleep 30
    
    # Start application services
    log "Starting application services..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d backend frontend nginx
    
    # Start monitoring (optional)
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        log "Starting monitoring services..."
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile monitoring up -d
    fi
    
    success "Services deployed successfully"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Health check attempt $attempt/$max_attempts..."
        
        # Check backend health
        if curl -f -s http://localhost/health &> /dev/null; then
            success "Application is healthy and responding"
            return 0
        fi
        
        log "Waiting for application to be ready..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    echo
    log "Service URLs:"
    echo "  Frontend: http://localhost (or https://localhost if SSL configured)"
    echo "  Backend API: http://localhost/api"
    echo "  Health Check: http://localhost/health"
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        echo "  Prometheus: http://localhost:9090 (internal only)"
        echo "  Grafana: http://localhost:3001 (internal only)"
    fi
    echo
}

# Rollback function
rollback() {
    warning "Rolling back deployment..."
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    
    # Restore from latest backup if available
    latest_backup=$(ls -t "$BACKUP_DIR" 2>/dev/null | head -n1)
    if [[ -n "$latest_backup" ]]; then
        log "Restoring from backup: $latest_backup"
        
        if [[ -d "$BACKUP_DIR/$latest_backup/postgres_data" ]]; then
            rm -rf postgres_data
            cp -r "$BACKUP_DIR/$latest_backup/postgres_data" .
        fi
        
        if [[ -d "$BACKUP_DIR/$latest_backup/redis_data" ]]; then
            rm -rf redis_data  
            cp -r "$BACKUP_DIR/$latest_backup/redis_data" .
        fi
        
        success "Rollback completed"
    else
        warning "No backup found for rollback"
    fi
}

# Main deployment function
main() {
    log "Starting SimGen AI production deployment..."
    
    # Trap to handle failures
    trap 'error "Deployment failed. Check logs for details."' ERR
    
    setup_directories
    validate_environment
    check_docker
    backup_data
    deploy_services
    health_check
    show_status
    
    success "🚀 SimGen AI deployment completed successfully!"
    log "Access your application at http://localhost"
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        log "Monitoring dashboards available at http://localhost:3001 (Grafana)"
    fi
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
        ;;
    "stop")
        log "Stopping all services..."
        docker-compose -f "$COMPOSE_FILE" down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting services..."
        docker-compose -f "$COMPOSE_FILE" restart "${2:-}"
        success "Services restarted"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|logs|stop|restart}"
        echo "  deploy   - Deploy the application (default)"
        echo "  rollback - Rollback to previous version"
        echo "  status   - Show service status"
        echo "  logs     - Show service logs (optionally specify service name)"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart services (optionally specify service name)"
        exit 1
        ;;
esac
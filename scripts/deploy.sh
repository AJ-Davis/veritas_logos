#!/bin/bash

# Veritas Logos API Deployment Script
# This script handles the complete deployment process for the API Gateway

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-production}
PROJECT_NAME="veritas-logos"
BACKUP_DIR="./backups"
LOG_FILE="./logs/deployment.log"

# Ensure logs directory exists
mkdir -p logs
mkdir -p $BACKUP_DIR

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a $LOG_FILE
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a $LOG_FILE
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a $LOG_FILE
}

# Check if running as root (not recommended)
check_user() {
    if [ "$EUID" -eq 0 ]; then
        warn "Running as root is not recommended for deployment"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check required dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    log "All dependencies satisfied"
}

# Load environment variables
load_environment() {
    log "Loading environment configuration..."
    
    # Check for .env file
    if [ -f ".env.${DEPLOYMENT_ENV}" ]; then
        log "Loading .env.${DEPLOYMENT_ENV}"
        set -a
        source ".env.${DEPLOYMENT_ENV}"
        set +a
    elif [ -f ".env" ]; then
        log "Loading .env"
        set -a
        source ".env"
        set +a
    else
        warn "No environment file found. Using default values."
    fi
    
    # Check required environment variables
    required_vars=("ANTHROPIC_API_KEY")
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Generate JWT secret if not provided
    if [ -z "$JWT_SECRET_KEY" ]; then
        export JWT_SECRET_KEY=$(openssl rand -hex 32)
        log "Generated new JWT secret key"
    fi
    
    log "Environment loaded successfully"
}

# Backup existing data
backup_data() {
    log "Creating backup..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_path="${BACKUP_DIR}/backup_${timestamp}"
    mkdir -p "$backup_path"
    
    # Backup database if it exists
    if [ -f "veritas_logos.db" ]; then
        cp "veritas_logos.db" "${backup_path}/veritas_logos.db"
        log "Database backed up"
    fi
    
    # Backup logs if they exist
    if [ -d "logs" ]; then
        cp -r logs "${backup_path}/logs"
        log "Logs backed up"
    fi
    
    log "Backup created at: $backup_path"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build main application image
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        log "Docker images built successfully"
    else
        error "Failed to build Docker images"
        exit 1
    fi
}

# Run security checks
security_check() {
    log "Running security checks..."
    
    # Check for secrets in code
    if command -v git &> /dev/null; then
        log "Checking for potential secrets in git history..."
        # This is a basic check - in production, use tools like git-secrets or truffleHog
        if git log --all --grep="password\|secret\|key" --oneline | head -5; then
            warn "Found potential secrets in git history. Please review."
        fi
    fi
    
    # Check file permissions
    log "Checking file permissions..."
    find . -type f -perm 777 2>/dev/null | while read file; do
        warn "File with 777 permissions found: $file"
    done
    
    log "Security checks completed"
}

# Deploy application
deploy() {
    log "Starting deployment..."
    
    # Stop existing containers
    docker-compose down --remove-orphans
    
    # Remove unused Docker resources
    docker system prune -f
    
    # Start services
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        log "Services started successfully"
    else
        error "Failed to start services"
        exit 1
    fi
}

# Health check
health_check() {
    log "Performing health checks..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        info "Health check attempt $attempt/$max_attempts"
        
        if curl -f -s "http://localhost:8000/health" > /dev/null; then
            log "Health check passed!"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Post-deployment tasks
post_deploy() {
    log "Running post-deployment tasks..."
    
    # Show service status
    info "Service status:"
    docker-compose ps
    
    # Show logs for any failed services
    failed_services=$(docker-compose ps --services --filter "status=exited")
    if [ -n "$failed_services" ]; then
        warn "Failed services detected: $failed_services"
        for service in $failed_services; do
            info "Logs for $service:"
            docker-compose logs --tail=50 "$service"
        done
    fi
    
    # Display useful information
    info "Deployment completed successfully!"
    info "API Gateway: http://localhost:8000"
    info "Health Check: http://localhost:8000/health"
    info "API Documentation: http://localhost:8000/docs"
    info "Monitoring (Prometheus): http://localhost:9090"
    info "Monitoring (Grafana): http://localhost:3001"
    
    log "Post-deployment tasks completed"
}

# Rollback function
rollback() {
    error "Rolling back deployment..."
    
    # Stop current deployment
    docker-compose down
    
    # Find latest backup
    latest_backup=$(ls -t $BACKUP_DIR | head -1)
    if [ -n "$latest_backup" ]; then
        info "Restoring from backup: $latest_backup"
        
        # Restore database
        if [ -f "${BACKUP_DIR}/${latest_backup}/veritas_logos.db" ]; then
            cp "${BACKUP_DIR}/${latest_backup}/veritas_logos.db" "./veritas_logos.db"
        fi
        
        # Restore logs
        if [ -d "${BACKUP_DIR}/${latest_backup}/logs" ]; then
            rm -rf logs
            cp -r "${BACKUP_DIR}/${latest_backup}/logs" "./logs"
        fi
        
        log "Backup restored"
    else
        warn "No backup found for rollback"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove old backups (keep last 5)
    if [ -d "$BACKUP_DIR" ]; then
        ls -t $BACKUP_DIR | tail -n +6 | xargs -r -I {} rm -rf "${BACKUP_DIR}/{}"
        log "Old backups cleaned up"
    fi
    
    # Remove old Docker images
    docker image prune -f
    
    log "Cleanup completed"
}

# Main deployment flow
main() {
    log "Starting Veritas Logos API deployment..."
    
    # Trap errors and attempt rollback
    trap 'error "Deployment failed. Attempting rollback..."; rollback; exit 1' ERR
    
    check_user
    check_dependencies
    load_environment
    backup_data
    security_check
    build_images
    deploy
    
    if health_check; then
        post_deploy
        cleanup
        log "Deployment completed successfully!"
    else
        error "Deployment failed health check"
        rollback
        exit 1
    fi
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health")
        health_check
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "stop")
        docker-compose down
        ;;
    "restart")
        docker-compose restart
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health|logs|status|stop|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Full deployment (default)"
        echo "  rollback - Rollback to previous version"
        echo "  health   - Check service health"
        echo "  logs     - Show service logs"
        echo "  status   - Show service status"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        exit 1
        ;;
esac 
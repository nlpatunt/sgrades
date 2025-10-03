#!/bin/bash

# S-GRADES Apache Frontend Setup Script

echo "Setting up S-GRADES Frontend on Apache..."
echo ""
echo "Select deployment environment:"
echo "1. dev-mac (macOS development environment)"
echo "2. prod-ubuntu (Ubuntu production environment)"
echo ""
read -p "Enter your choice (1 or 2): " ENVIRONMENT_CHOICE

case $ENVIRONMENT_CHOICE in
    1)
        ENVIRONMENT="dev-mac"
        echo "Selected: dev-mac"
        ;;
    2)
        ENVIRONMENT="prod-ubuntu"
        echo "Selected: prod-ubuntu"
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac

echo ""

# Check if running as root for system-wide installation
INSTALL_SYSTEM=false
APACHE_SITES_DIR=""
APACHE_DOC_ROOT=""
APACHE_SERVICE=""
USE_SUBDIR=true

if [[ "$ENVIRONMENT" == "prod-ubuntu" ]]; then
    # Production Ubuntu setup
    APACHE_SITES_DIR="/etc/apache2/sites-available"
    APACHE_DOC_ROOT="/var/www/html"
    APACHE_SERVICE="apache2"
    USE_SUBDIR=false
    echo "Configuring for production Ubuntu environment..."
elif [[ "$ENVIRONMENT" == "dev-mac" ]]; then
    # Development macOS setup
    USE_SUBDIR=true
    # Check if we're running as root (sudo) - brew won't work
    if [[ $EUID -eq 0 ]]; then
        # When running as root, check for Apache installation by looking for the binary
        if [[ -f "/opt/homebrew/bin/httpd" ]] || [[ -f "/usr/local/bin/httpd" ]] || command -v httpd &>/dev/null; then
            APACHE_SITES_DIR="/opt/homebrew/etc/httpd/other"
            APACHE_DOC_ROOT="/opt/homebrew/var/www"
            APACHE_SERVICE="httpd"
        else
            echo "Apache not found. Please install Apache first:"
            echo "   brew install httpd"
            echo "Note: Install Apache without sudo, then re-run this script with sudo"
            exit 1
        fi
    else
        # Running as normal user, can use brew
        if brew list httpd &>/dev/null || brew list apache2 &>/dev/null; then
            APACHE_SITES_DIR="/opt/homebrew/etc/httpd/other"
            APACHE_DOC_ROOT="/opt/homebrew/var/www"
            APACHE_SERVICE="httpd"
        else
            echo "Apache not found. Please install Apache first:"
            echo "   brew install httpd"
            exit 1
        fi
    fi
    echo "Configuring for development macOS environment..."
else
    echo "Unsupported environment: $ENVIRONMENT"
    exit 1
fi

# Check if Apache is installed
if ! command -v httpd &> /dev/null && ! command -v apache2 &> /dev/null; then
    echo "Apache is not installed. Please install Apache first."
    exit 1
fi


setup_system_wide() {
    echo "Setting up system-wide Apache configuration..."

    # Check if running with sufficient privileges
    if [[ $EUID -ne 0 ]]; then
        echo "System-wide setup requires sudo privileges."
        echo "   Re-run with: sudo ./setup_apache_frontend.sh"
        exit 1
    fi

    # Determine target directory based on environment
    if [[ "$USE_SUBDIR" == "true" ]]; then
        # dev-mac: use subdirectory
        TARGET_DIR="$APACHE_DOC_ROOT/sgrades"
        mkdir -p "$TARGET_DIR"
        echo "Installing to subdirectory: $TARGET_DIR"
    else
        # prod-ubuntu: use base directory
        TARGET_DIR="$APACHE_DOC_ROOT"
        echo "Installing to base directory: $TARGET_DIR"
    fi

    # Copy frontend files
    echo "Copying frontend files..."
    cp -r app/frontend/* "$TARGET_DIR/"

    # Set proper permissions
    if [[ "$ENVIRONMENT" == "prod-ubuntu" ]]; then
        chown -R www-data:www-data "$TARGET_DIR" 2>/dev/null || \
        echo "Could not set www-data permissions (this may be OK)"
    else
        chown -R _www:_www "$TARGET_DIR" 2>/dev/null || \
        echo "Could not set _www permissions (this may be OK)"
    fi

    # Copy Apache configuration - use appropriate config file for environment
    if [[ -d "$APACHE_SITES_DIR" ]]; then
        if [[ "$ENVIRONMENT" == "prod-ubuntu" ]]; then
            # Production: Do not copy config file, use existing one
            echo "Production environment detected - skipping Apache config file copy"
            echo "Ensure Apache config is already present at $APACHE_SITES_DIR/sgrades.conf"
        else
            # Use development configuration
            cp apache_config/sgrades.conf "$APACHE_SITES_DIR/"
            echo "Apache development configuration copied to $APACHE_SITES_DIR/sgrades.conf"

            # Update paths in configuration file for dev-mac
            sed -i.bak "s|/opt/homebrew/var/www|$APACHE_DOC_ROOT|g" "$APACHE_SITES_DIR/sgrades.conf"
            echo "Updated document root path in configuration"
        fi
    else
        echo "Apache sites directory not found. Manual configuration needed."
        if [[ "$ENVIRONMENT" == "prod-ubuntu" ]]; then
            echo "   Ensure sgrades.conf exists in your Apache configuration directory"
        else
            echo "   Copy apache_config/sgrades.conf to your Apache configuration directory"
        fi
    fi

    # Enable required Apache modules (Ubuntu/Debian)
    if command -v a2enmod &> /dev/null; then
        echo "Enabling required Apache modules..."
        a2enmod rewrite proxy proxy_http headers expires deflate ssl
        a2ensite sgrades.conf

        # Test Apache configuration
        if apache2ctl configtest; then
            echo "Apache configuration is valid"

            # Restart Apache
            systemctl reload $APACHE_SERVICE
            echo "Apache reloaded"
        else
            echo "Apache configuration test failed"
            exit 1
        fi
    fi

    echo ""
    echo "✓ Frontend files installed to $TARGET_DIR"
    echo "✓ Environment: $ENVIRONMENT"
}

setup_system_wide
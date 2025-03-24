#!/usr/bin/env bash

# Create log directory if it doesn't exist
if [ ! -d "/var/log/supervisor" ]; then
    echo "INFO: Creating supervisor log directory"
    mkdir -p /var/log/supervisor
fi

# Execute supervisord with the specified config
echo "INFO: Starting supervisord"
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

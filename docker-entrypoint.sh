#!/usr/bin/env bash
#
# This script checks /servers for any new subfolders. If it finds
# a subfolder whose requirements have not yet been installed,
# it installs them, then finally runs mcpo (or whatever command is passed).
# In the event that /servers is empty, all default mcp servers are copied and the example config file is used.

set -e  # Exit immediately if a command exits with a non-zero status

# If /servers is empty, clone the mcp servers repo and copy its "src" directory
DIR="/servers"
if [ ! -d "$DIR" ]; then
    echo "$DIR does not exist. Creating it..."
    mkdir -p "$DIR"
fi
if [ -d "$DIR" ] && [ -z "$(ls -A "$DIR")" ]; then
    # Servers directory is empty
    echo "$DIR is empty, cloning example mcp servers..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/modelcontextprotocol/servers.git /servers/tmp
    git -C /servers/tmp sparse-checkout init --cone
    git -C /servers/tmp sparse-checkout set src/time src/fetch
    mv /servers/tmp/src/* /servers/
    echo "listing /servers dir"
    ls /servers
    echo "listing /servers subdirs"
    ls /servers/*
    rm -rf /servers/tmp 
    echo "Cloned and copied servers into /servers."
    if [ -f "/app/config.json" ]; then
      echo "Config supplied but there are no servers! Reverting to example config and backing up current config."
      mv -f /app/config.json /app/backup.config.json
    else
      echo "No config supplied, using example config."
    fi
    cp /usr/local/bin/example.config.json /app/config.json
else
    # Servers directory is not empty
    echo "$DIR exists and is not empty. Doing nothing."
fi

# Directory to track which /servers/ subfolders we've installed
INSTALLED_DIR="/usr/local/bin/installed_servers"

# Ensure the tracking directory exists
mkdir -p "$INSTALLED_DIR"

# Temporary file to combine new requirements
NEW_REQS_FILE=$(mktemp)

# For each subfolder in /servers, if not yet installed, add its requirements
if [ -d /servers ]; then
    for folder in /servers/*; do
        if [ -d "$folder" ]; then
            subfolder="$(basename "$folder")"
            # Install the actual server
            echo "$folder" >> "$NEW_REQS_FILE"
            # If we haven't installed this subfolder yet
            if [ ! -f "$INSTALLED_DIR/$subfolder" ]; then
                # If it has a requirements.txt, add to the combined file
                if [ -f "$folder/requirements.txt" ]; then
                    for l in "$folder/requirements.txt"; do (cat "${l}"; echo) >> "$NEW_REQS_FILE"; done
                fi

                # Mark this subfolder as installed
                touch "$INSTALLED_DIR/$subfolder"
            fi
        fi
    done
fi

# If NEW_REQS_FILE is non-empty, install everything in one go
if [ -s "$NEW_REQS_FILE" ]; then
    pip install -r "$NEW_REQS_FILE"
fi

# Clean up
rm "$NEW_REQS_FILE"

# Finally, execute the command that was passed in (defaults to "mcpo --config $CONFIG_PATH" from the Dockerfile)
exec "$@"

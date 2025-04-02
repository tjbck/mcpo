#!/usr/bin/env bash
#
# This script checks /servers for any new subfolders (custom mcp servers). If it finds
# a subfolder with python requirements.txt that have not yet been installed,
# it installs them, then finally runs mcpo (or whatever command is passed).
# /servers dir also allows adding custom executables, etc. on the fly.
# TO DO: Add support for offline node modules.

set -e  # Exit immediately if a command exits with a non-zero status

DIR="/servers"
if [ ! -d "$DIR" ]; then
    echo "$DIR does not exist. Creating it..."
    mkdir -p "$DIR"
fi

if [ -d "$DIR" ] && [ -z "$(ls -A "$DIR")" ]; then
    # Servers directory is empty
    echo "$DIR is empty"
fi

if ! [ -f "/app/config.json" ]; then
      echo "No config supplied, using example config"
      cp /usr/local/bin/example.config.json /app/config.json
else
      echo "Config file exists."
fi

# Directory to track which /servers/ subfolders we've installed
INSTALLED_DIR="/usr/local/bin/installed_servers"

# Ensure the tracking directory exists
mkdir -p "$INSTALLED_DIR"

# Temporary file to combine new requirements
NEW_REQS_FILE=$(mktemp)

# For each subfolder in /servers, if not yet installed, add its requirements
if [ -d "$DIR" ]; then
    for folder in "$DIR/*"; do
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

# If NEW_REQS_FILE is non-empty, install new python dependencies
if [ -s "$NEW_REQS_FILE" ]; then
    pip install -r "$NEW_REQS_FILE"
fi

# Clean up
rm "$NEW_REQS_FILE"

# Finally, execute the command that was passed in (defaults to "mcpo --config $CONFIG_PATH" from the Dockerfile)
exec "$@"
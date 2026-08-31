#!/bin/bash

FOLDER="$1"
ANGLE="${2:-90}"

if [ -z "$FOLDER" ]; then
    echo "Usage: ./rotate_videos.sh /path/to/video_folder [90|180|270]"
    exit 1
fi

python3 outputs/rotate_videos.py \
    --folder "$FOLDER" \
    --angle "$ANGLE"
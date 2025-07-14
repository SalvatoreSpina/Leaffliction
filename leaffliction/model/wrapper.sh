#!/bin/bash

# Base directory
BASE_DIR="splitted/datasets"

# Loop through each subdirectory (Apples and Grapes)
for PARENT_DIR in "$BASE_DIR"/*/; do
  # Get the parent directory name (Apples or Grapes)
  PARENT_NAME=$(basename "$PARENT_DIR")

  # Loop through the training and validation directories inside each parent
  for SUB_DIR in "$PARENT_DIR"*/; do
    # Get the subdirectory name (training or validation)
    SUB_NAME=$(basename "$SUB_DIR")

    # Create a new directory with the parent name inside the subdirectory
    NEW_DIR="$SUB_DIR/$PARENT_NAME"
    
    # Only create the new directory if it does not exist
    if [ ! -d "$NEW_DIR" ]; then
      mkdir -p "$NEW_DIR"
    fi

    # Move the contents of the subdirectory into the new directory, excluding the new parent-named directory
    find "$SUB_DIR" -mindepth 1 -maxdepth 1 -not -name "$PARENT_NAME" -exec mv {} "$NEW_DIR" \;
  done
done

echo "Wrapping completed!"

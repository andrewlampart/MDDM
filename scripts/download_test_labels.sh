#!/bin/bash
# Download full test labels from DAIC-WOZ official server

BASE_URL="https://dcapswoz.ict.usc.edu/wwwdaicwoz"
OUTPUT_DIR="/mnt/c/Users/GUMed/Documents/MDDM/DAIC-WOZ-Dataset"

echo "=========================================="
echo "Downloading DAIC-WOZ test labels"
echo "=========================================="

# Download full_test_split.csv (contains PHQ8 labels for test set)
echo "Downloading full_test_split.csv..."
wget -O "${OUTPUT_DIR}/full_test_split.csv" "${BASE_URL}/full_test_split.csv"

if [ $? -eq 0 ]; then
    echo "✅ Successfully downloaded full_test_split.csv"
    echo ""
    echo "File location: ${OUTPUT_DIR}/full_test_split.csv"
    echo ""
    echo "First few lines:"
    head -5 "${OUTPUT_DIR}/full_test_split.csv"
else
    echo "❌ Download failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="

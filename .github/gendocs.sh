
find . -name "*.md" -type f -exec m2r {} \;
find . -name "*.rst" -type f -exec mv {} docs/ \;
find work_dirs -type d -path "*/round_*/teacher_outputs/*/results" | while read dir; do
    find "$dir" -maxdepth 1 -name "*.json" -delete
    echo "Deleted jsons in $dir"
done
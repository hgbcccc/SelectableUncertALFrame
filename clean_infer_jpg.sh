find work_dirs -type d -path "*/round_*/teacher_outputs/*/visualize" | while read dir; do
    find "$dir" -maxdepth 1 -name "*.jpg" -delete
    echo "Deleted jpgs in $dir"
done
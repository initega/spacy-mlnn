#!/usr/bin/env sh

for dir in "$1"/*
do
    # echo "$dir"
    for file in "$dir"/*
    do
        entity="$(basename $dir)"
        word="$(basename "$file" .txt | tr '_' ' ')"
        echo "// $word"
        sh text2traindata.sh "$file" "$word" "$entity"
    done
done

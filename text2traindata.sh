#!/usr/bin/env sh

[ -z "$1" -o -z "$2" -o -z "$3" ] && echo "Missing parameters" 1>&2 && exit 1

file="$1" # Filename
name="$2" # Name that will be identified with the next entity
entity="$3" # Entity to associate words with

normalize(){
    awk '$0 != "" {printf "%s, ",$0} $0 == "" {printf "\n"}' $1 | \
        sed 's/-, //g' | sed 's|,||g'
}

content="$(normalize "$file")"

echo "$content" | while read -r line
do
    awk_command="{print match(\$0, \"$name\")}"
    text="\"$line\""

    location="$(echo "$text" | awk "$awk_command")"

    if [ "$location" -ne 0 ]; then
        begin="$(( $location - 2))"
        end="$(( $begin + $(printf "$name" | wc -c) ))"
        coords="($begin, $end, \"$entity\")"
    fi

    
    entity_spec="{\"entities\": [$coords]}"

    printf "%b\n" \
"    (
        $text,
        $entity_spec
    ),"

    coords=""
done

#!/bin/bash
filename='commit_list'
rm -f asv_commit_list
while read line; do
    commit=$(echo $line | cut -f1 -d ' ')
    url=$(echo $line | cut -f2 -d ' ')
    echo $commit >> asv_commit_list
done < $filename
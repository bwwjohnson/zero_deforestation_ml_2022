#!/bin/bash


# get list of paths for each classsification
awk -F"," '{if($1==0) print $5}' train.csv > 0_paths
awk -F"," '{if($1==1) print $5}' train.csv > 1_paths
awk -F"," '{if($1==2) print $5}' train.csv > 2_paths

rm -r *_img
mkdir 0_img
mkdir 1_img
mkdir 2_img

sed $'s/\r$//'  0_paths > 0_pathsnew
sed $'s/\r$//'  1_paths > 1_pathsnew
sed $'s/\r$//'  2_paths > 2_pathsnew

LINES=`cat 0_pathsnew`
for line in $LINES; do
cp  $line 0_img
done

LINES=`cat 1_pathsnew`
for line in $LINES; do
cp  $line 1_img
done
LINES=`cat 2_pathsnew`
for line in $LINES; do
cp  $line 2_img
done
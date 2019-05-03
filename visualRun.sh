#!/usr/bin/env bash

python visualize.py --model resnet50 --attention no
python visualize.py --model se_resnet50 --attention channel
python visualize.py --model bam_resnet50 --attention joint
python visualize.py --model bam_resnet50 --attention spatial
python visualize.py --model bam_resnet50 --attention channel
python visualize.py --model cam_resnet50 --attention joint
python visualize.py --model cam_resnet50 --attention spatial
python visualize.py --model cam_resnet50 --attention channel

python visualize.py --model resnet34 --attention no
python visualize.py --model se_resnet34 --attention channel
python visualize.py --model bam_resnet34 --attention joint
python visualize.py --model bam_resnet34 --attention spatial
python visualize.py --model bam_resnet34 --attention channel
python visualize.py --model cam_resnet34 --attention joint
python visualize.py --model cam_resnet34 --attention spatial
python visualize.py --model cam_resnet34 --attention channel
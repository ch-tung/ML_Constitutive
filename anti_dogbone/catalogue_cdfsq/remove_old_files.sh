#!/bin/bash

filepath="output_files/"
find $filepath -name "*.xdmf" -type f -mmin +60 -delete
find $filepath -name "*.h5" -type f -mmin +60 -delete
find $filepath -name "_pre*" -type f -mmin +60 -delete

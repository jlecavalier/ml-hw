#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
	python logreg.py --step .15 --passes $i >> ./results/passes_equals_$i
done
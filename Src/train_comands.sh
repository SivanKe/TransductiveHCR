#!/usr/bin/env bash

train.py --do-test-vat True --vat-epsilon 0.5 --vat-xi 1e-6 --vat-sign True --vat-ratio 10. \
--output-dir '../Output/transductive_vat' --do-lr-step True
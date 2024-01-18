The interpolator can be tested via

```
./uiuc-test \
    ../../etc/uiuc/equation_of_state.csv.gz \
    --downsample 5 \
    --figtype png \
    --figtype pdf \
    --tag uiuc-equation_of_state \
    --time-execution
```

```
./uiuc-test-new \
    ../../etc/uiuc/equation_of_state.csv.gz \
    --downsample 10
    --num-burnin 500 \
    --num-samples 10000 \
    --num-walkers 100 \
    --tag uiuc-equation_of_state \
    -v

./uiuc-test-new-corner \
    uiuc-test-new-samples_uiuc-equation_of_state.hdf \
    -v
```



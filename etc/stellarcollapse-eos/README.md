These EoS tables were taken from: https://stellarcollapse.org/equationofstate.html


```
./regrid \
    LS180_234r_136t_50y_analmu_20091212_SVNr26.h5 \
    regrid-LS180_50logenergy.hdf \
    --grid 50 17.5 33.0 \
    --plot \
    --verbose

./regrid2 \
    LS180_234r_136t_50y_analmu_20091212_SVNr26.h5 \
    regrid2-LS180_50logenergy.hdf \
    --grid 50 16.0 17.5 33.0 \
    --plot \
    --verbose
```


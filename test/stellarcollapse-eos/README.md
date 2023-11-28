A sandbox to test interpolation of EoS tables from stellarcollapse.org with the goal of producing estimates for the temperature on a regularly spaced grid of (number density, energy density, electron fraction).

The basic script can be run via

```
./sc-eos-test \
    ../../etc/stellarcollapse-eos/LS180_234r_136t_50y_analmu_20091212_SVNr26.h5 \
    --outpath resampled-LS180_234r_136t_50y_analmu_20091212_SVNr26.h5 \
    --downsample 10 \
    --ye-num-points 20 \
    --baryon-density-num-points 20 \
    --energy-density-num-points 20 \
    --verbose \
    --time-execution
```

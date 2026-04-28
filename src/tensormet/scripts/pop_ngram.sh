python -m tensormet.scripts.population \
          --dataset 3gram \
          --top-ks 1000,2000,4000,6000,10000 \
          --cols-to-build w1,w2,w3 \
          --shared-factors 0-1,1-2
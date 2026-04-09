python -m tensormet.scripts.population \
          --dataset frame_based \
          --top-ks 1000,2000,4000,6000,10000 \
          --cols-to-build frame_name,target,arg1,arg2,arg3 \
          --shared-factors 1-2,2-3,3-4
#!/bin/bash 

if [ -d /Users/samuel/Desktop/BNL/puck_visualization ]; then
   export PUCK_ROOT=/Users/samuel/Desktop/BNL/puck_visualization
fi

puck_visualization.sh puck_C_004_dewarFull.jpg
puck_visualization.sh puck_C_emptyPuck_94.6pct_filling.jpg
puck_visualization.sh puck_C_emptyPuck_96.8pct_emptying.jpg
puck_visualization.sh puck_C_fullPuck_ALScaps_95.7pct_filling.jpg
puck_visualization.sh puck_C_fullPuck_B5Rcaps_94pct_filling.jpg
puck_visualization.sh puck_C_fullPuck_gunMetalCaps_96.5pct_filling.jpg
puck_visualization.sh puck_C_fullPuck_silverCaps_96.4_emptying.jpg
puck_visualization.sh puck_C_tiltedCaps_ALScaps_94.5pct_emptying.jpg
puck_visualization.sh puck_C_tiltedCaps_B5Rcaps_95pct_emptying.jpg
puck_visualization.sh puck_C_tiltedCaps_gunMetalCaps_92pct_filling.jpg
puck_visualization.sh puck_C_tiltedCaps_gunMetalCaps_94.8pct_filling.jpg
puck_visualization.sh puck_C_tiltedCaps_silverCaps_96.2pct_fillOff.jpg
#!/bin/bash

qsub unweighted_geometry_xe.sh
qsub unweighted_geometry_scst.sh
qsub unweighted_geometry_scst_10.sh
qsub unweighted_geometry_scst_25.sh
qsub unweighted_geometry_scst_50.sh

qsub weighted_geometry_xe.sh
qsub weighted_geometry_scst.sh
qsub weighted_geometry_scst_10.sh
qsub weighted_geometry_scst_25.sh
qsub weighted_geometry_scst_50.sh

# qsub unweighted_semantic_xe.sh
# qsub unweighted_semantic_scst.sh
# qsub unweighted_semantic_scst_10.sh
# qsub unweighted_semantic_scst_25.sh
# qsub unweighted_semantic_scst_50.sh

sleep 5
qstat
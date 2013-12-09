#!/bin/bash

cd MMM1DColumn
qsub p1.pbs
qsub p2.pbs
qsub p4.pbs
qsub p8.pbs
qsub p16.pbs
qsub p32.pbs
qsub p64.pbs
qsub p128.pbs
cd ..

cd MMM1DRow
qsub p1.pbs
qsub p2.pbs
qsub p4.pbs
qsub p8.pbs
qsub p16.pbs
qsub p32.pbs
qsub p64.pbs
qsub p128.pbs
cd ..

cd MMM2D
qsub p1.pbs
qsub p4.pbs
qsub p9.pbs
qsub p16.pbs
qsub p25.pbs
qsub p36.pbs
qsub p49.pbs
qsub p64.pbs
qsub p81.pbs
qsub p100.pbs
qsub p121.pbs
qsub p144.pbs
cd ..


cd MMM3D
qsub p1.pbs
qsub p8.pbs
qsub p27.pbs
qsub p64.pbs
qsub p125.pbs
cd ..

cd MMM25D
qsub c2p8.pbs
qsub c2p18.pbs
qsub c2p32.pbs
qsub c2p50.pbs
qsub c2p72.pbs
qsub c2p98.pbs
qsub c2p128.pbs
qsub c3p27.pbs
qsub c3p48.pbs
qsub c3p75.pbs
qsub c3p108.pbs
qsub c3p147.pbs
qsub c4p64.pbs
qsub c4p100.pbs
qsub c4p144.pbs
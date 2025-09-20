Machine learning for high energy physics pattern recognition for `pfRICH` particle identification.

The pfRICH ROOT files need to be sanitized using the `readTree.C ` macro which is run as follows (set file paths appropriately):
`root -l "readTree.C(\"rawdata/Kplus_5GeV_etaneg2.0_phirand_10000ev.root\",\"datafiles/Kplus_5GeV_etaneg2.0_phirand_10Kevts_4ML.root\",5,\"Kplus\")"`

import ROOT as r
import numpy as np


tfile = r.TFile.Open("correctionSiHitAnalysish.root")
tree = tfile.Get("ntuples/SiHGTD") 

pdgid = []
prodVtx_x = []
prodVtx_y = []
prodVtx_z = []

for event in tree:
    pdgid.append(list(event.HGTD_pdgid))
    prodVtx_y.append(list(event.HGTD_prodVtx_y))
    prodVtx_x.append(list(event.HGTD_prodVtx_x))
    prodVtx_z.append(list(event.HGTD_prodVtx_z))

print(pdgid[1])
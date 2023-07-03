import ROOT as r
import fast_library as Lbr
import Tools_library as Tools
from pyjet import ClusterSequenceArea
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from pyjet import cluster, DTYPE_PTEPM
import fastjet
import pytest
import math
import pprint
from tqdm import tqdm
vector = pytest.importorskip("vector")
import sys



data = np.loadtxt('DBSCAN_donnees.txt')


BIB_eff_list = data[:, 0].tolist()
top_eff_list = data[:, 1].tolist()
rayon_list = data[:, 2].tolist()
c_list = data[:, 3].tolist()

c_list2 = np.array(c_list)





canvas = r.TCanvas("canvas", "Nb True", 1200, 600)
graph = r.TGraph(len(top_eff_list), np.array(top_eff_list), np.array(BIB_eff_list))
graph.SetLineColor(r.kMagenta+2)
graph.SetLineWidth(2)
graph.GetXaxis().SetTitle("False Efficiency")
graph.GetYaxis().SetTitle("True Efficiency")

graph.GetXaxis().SetRangeUser(0, 0.5)  
graph.GetYaxis().SetRangeUser(70, 100)

canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.05)
canvas.SetTopMargin(0.05)
canvas.SetBottomMargin(0.15)
canvas.Update()

canvas.SetTitle("Track Efficiency ")
graph.Draw()


canvas.Draw()
canvas.SaveAs("FastJet_DBSCAN_Efficiency.png")








sys.exit()
canvas = r.TCanvas("canvas", "Nb True", 1200, 600)
graph = r.TGraph(len(rayon_list), np.array(rayon_list), np.array(eff_list))
graph.SetLineColor(r.kMagenta+2)
graph.SetLineWidth(2)
graph.GetXaxis().SetTitle("Radius")
graph.GetYaxis().SetTitle("Number of differentiable Cluster (%)")

graph.GetXaxis().SetRangeUser(0, 0.5)  
graph.GetYaxis().SetRangeUser(70, 100)

canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.05)
canvas.SetTopMargin(0.05)
canvas.SetBottomMargin(0.15)
canvas.Update()

canvas.SetTitle("Track Efficiency ")
graph.Draw()


canvas.Draw()
canvas.SaveAs("DBSCAN_Eff_Rayon_10event2.png")


sys.exit()



canvas = r.TCanvas("canvas", "Hit in clusters", 1200, 600)
graph = r.TGraph(len(rayon_list), np.array(rayon_list), np.array(c_list2))
graph.SetLineColor(r.kMagenta+2)
graph.SetLineWidth(2)
graph.GetXaxis().SetTitle("Radius")
graph.GetYaxis().SetTitle("Number of hit in cluster (%)")

graph.GetXaxis().SetRangeUser(0., 0.5)  
graph.GetYaxis().SetRangeUser(89, 101)
canvas.SetTitle("Hits Efficiency ")
canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.05)
canvas.SetTopMargin(0.05)
canvas.SetBottomMargin(0.15)
canvas.Update()



graph.Draw()



canvas.Draw()
canvas.SaveAs("DB_SCAN_hit_Rayon_10event3.png")


sys.exit()

























canvas = r.TCanvas("canvas", "Nb True", 1200, 600)
graph = r.TGraph(len(rayon_list), np.array(rayon_list), np.array(eff_list))
graph.SetLineColor(r.kMagenta+2)
graph.SetLineWidth(2)
graph.GetXaxis().SetTitle("Radius")
graph.GetYaxis().SetTitle("Number of differentiable Cluster (%)")

graph.GetXaxis().SetRangeUser(0.1, 0.75)  
graph.GetYaxis().SetRangeUser(0, 100)

canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.05)
canvas.SetTopMargin(0.05)
canvas.SetBottomMargin(0.15)
canvas.Update()



graph.Draw()


legend = r.TLegend(0.25, 0.65, 0.35, 0.85)
legend.SetBorderSize(0)
legend.SetTextSize(0.03)
legend.SetFillColor(0)
legend.AddEntry(graph,'Tracker and', "l")
legend.Draw()

canvas.Draw()
canvas.SaveAs("Eff_Rayon_10event2.png")
from unittest.mock import sentinel
import numpy as np
from keywords import *
from functools import partial
from functionalLib import compose
import fitz
from PyPDF2 import PdfMerger

from auxiliary import *


def getPathsForClusteringAnalysis(pathsList, models_dict, iniMode, maxIter, distMeasure):
    pdfs_path = get_partial_PDFs_path(models_dict)
    basePath = pdfs_path + '_'.join([iniMode, str(maxIter), distMeasure])
    clusteringHeader = basePath + '_.pdf'
    clusterHeaderImage = pdfs_path + clusteringComparisionHeaderPdf
    clustersImage = basePath + '_clusters_.pdf'
    mseImage = basePath + '_plot_comparision_mse_.pdf'
    silhoutteImage = basePath + '_plot_comparision_silhouette_.pdf'
    pointsForTestHeaderImage = pdfs_path + points_to_test_comparision_header
    pointsForTestImage = basePath + '_points_to_test_.pdf'
    ksImage = basePath + '_ks_plot_KS_test_.pdf'
    pathsList += [clusteringHeader, 
                  clusterHeaderImage, 
                  clustersImage, 
                  mseImage, 
                  silhoutteImage, 
                  pointsForTestHeaderImage, 
                  pointsForTestImage, 
                  ksImage]


def getRaportHeaderPaths(pathsList, models_dict):
    basePath = get_partial_PDFs_path(models_dict)
    raportHeader = basePath + 'raport_header.pdf'
    originalSet = basePath + 'original_set_.pdf'
    originalSubsetComparision = basePath + 'original_subsets_comparision_.pdf'
    originalSetSummary = basePath + 'originalSetSummary_.pdf'
    originalSet1Summary = basePath + 'originalSet1Summary_.pdf'
    originalSet2Summary = basePath + 'originalSet2Summary_.pdf'
    transformedSet1Summary = basePath + 'transformedSet1Summary_.pdf'
    transformedSet2Summary = basePath + 'transformedSet2Summary_.pdf'
    pathsList += [raportHeader, 
                  originalSet,
                  originalSubsetComparision,
                  originalSetSummary, 
                  originalSet1Summary, 
                  originalSet2Summary, 
                  transformedSet1Summary, 
                  transformedSet2Summary]


def createReport(models_dict, partialPDFsPaths):
    parameters = models_dict[parametersDict]
    destinationFolder = parameters[raportDestination]
    raportPath = destinationFolder + parameters[name]
    merger = PdfMerger()

    for pdf in partialPDFsPaths:
        merger.append(pdf)

    merger.write(raportPath)
    merger.close()
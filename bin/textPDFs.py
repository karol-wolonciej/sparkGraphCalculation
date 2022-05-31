import numpy as np
import matplotlib.pyplot as plt
from auxiliary import *
from keywords import *
from fpdf import FPDF 


singleCell = lambda pdf, w, h, txt, align: pdf.cell(w=w, h=h, txt=txt, align=align)
multiCell = lambda pdf, w, h, txt, align: pdf.multi_cell(w=w, h=h, txt=txt, align=align)


def writeLine(format, orientation, fontType, size, cellWith, cellHeight, align, cellType, txt, path):
    pdf = FPDF(orientation=orientation, format=format)
    pdf.add_page()
    pdf.set_font(fontType, size=size)
    pdf.set_auto_page_break(auto=False)
    cellType(pdf=pdf, w=cellWith, h=cellHeight, txt=txt, align=align)
    pdf.output(path)

cellWith = 304

writeUsualLine = partial(writeLine, format=(20, cellWith), orientation='landscape', fontType='Arial', size=20, cellWith=cellWith, cellHeight=5, align='L', cellType=singleCell)
writeHeader = partial(writeLine, format=(60, cellWith), orientation='landscape', fontType='Arial', size=60, cellWith=cellWith, cellHeight=30, align='C', cellType=singleCell)
writeMultipleLines = partial(writeLine, format=(75, cellWith), orientation='landscape', fontType='Arial', size=10, cellWith=cellWith, cellHeight=5, align='L', cellType=multiCell)


raportHeaderPath = lambda parameters: parameters[partialPDFsPath] + 'raport_header.pdf'
getSubHeaderPath = lambda parameters, *params: parameters[partialPDFsPath] + '_'.join(params)


def createRaportHeader(models_dict):
    parameters = models_dict[parametersDict]
    headerText = "Report " + parameters[name]
    path = raportHeaderPath(parameters)
    writeHeader(txt=headerText, path=path)


def createSubHeaders(models_dict, iniMode, maxIter, distMeasure):
    parameters = models_dict[parametersDict]
    iniMode, maxIter, distMeasure = compose(tuple, map)(str, (iniMode, maxIter, distMeasure))
    subHeaderText = 'Initialization Mode: ' + iniMode + ', maxIteration: ' + maxIter + ', distance measure: ' + distMeasure
    path = getSubHeaderPath(parameters, iniMode, maxIter, distMeasure, pdf_extension)
    writeUsualLine(txt=subHeaderText, path=path)


def createSummary(summary_keyword, models_dict):
    summaryHeader = summariesDescribtion[summary_keyword] + '\n\n'
    summaryText = summaryHeader + models_dict[summary_keyword]
    path = get_path_to_partial_pdf(models_dict, summary_keyword)
    writeMultipleLines(txt=summaryText, path=path)
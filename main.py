import csv
import camelot
import argparse

import huggingface_hub

TRANSFORMERS_OFFLINE = 1
from pikepdf import Pdf
from flair.data import Sentence
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import resolve1
from segtok.segmenter import split_single
from tqdm import tqdm
import easyocr
from pdf2image import convert_from_path  # For scanned PDFs
from collections import OrderedDict
from pdfminer import pdfpage
import cv2
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO, BytesIO
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

from flair.visual.ner_html import render_ner_html, HTML_PAGE, TAGGED_ENTITY, PARAGRAPH
import tempfile
import webbrowser

#####################
# ONLY FOR DEMO USE #
#####################
import warnings

warnings.filterwarnings("ignore")


#####################
# ONLY FOR DEMO USE #
#####################

def pdf2img(pdf, name, pagenums=None):
    """
    Takes PDF page and converts it to png image for running OCR
    :param pdf: Pdf location
    :param name: Pdf name for image file
    :param pagenums: page number to be converted
    :return: save pages as jpeg images
    """

    # print(len(PDF))
    global total_pages
    pages = convert_from_path(pdf, 300, poppler_path=r"C:\poppler-0.68.0\bin", timeout=10000, first_page=pagenums,
                              last_page=pagenums)

    # Counter to store images of each page of PDF to image
    image_counter = 1
    #
    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> pdfname_1.jpg
        # PDF page 2 -> pdfname_2.jpg
        # PDF page 3 -> pdfname_3.jpg
        # . . . .
        # PDF page n -> pdfname_n.jpg
        filename = f'{name}_' + str(image_counter) + ".png"

        # Save the image of the page in system
        page.save(r'C:/Data/Output/OCR/images/' + filename)
        # print('Saved page number ' + str(image_counter))
        # Increment the counter to update filename
        image_counter = image_counter + 1
    total_pages = image_counter


def table_extraction(pdf, name, page, type):
    """
    Find any lattice tables in the page and convert its rows to syntactically sane text sentences for NER run
    :param pdf: Location of PDF file
    :param name: Name of PDF file
    :param page: Page numbers to find tables in
    :param type: If OCR or regular PDF page
    :return: Text string that contains row vise sentences
    """
    if type == 'lattice':
        try:
            tables, text = camelot.read_pdf(pdf, flavor='lattice', strip_text='\n', pages=str(page),
                                            backend="poppler", split_text=True, process_background=False,
                                            copy_text=['h', 'v'], line_scale=60)
            # layout_kwargs={'char_margin': 1, 'line_margin': 0.2,
            #                'boxes_flow': 1})  # Text based page
        except ZeroDivisionError:  # if (bbox_intersection_area(ba, bb) / bbox_area(ba)) > 0.8: ZeroDivisionError:
            # float division by zero
            print('Zero Division Error')
            return None
    else:
        tables, text = camelot.read_pdf(pdf, flavor='lattice_ocr', pages=str(page))  # OCR based page

    tables.export(f'C:/Data/Output/{name} tables.csv', f='txt',
                  compress=True)  # json, excel, html, markdown, sqlite, txt
    # print(tables.export(r'C:\Data\Output\tables\table.txt', f='txt'))
    tablesfin, item, dic, header, table_line = [], '', {}, 0, []
    for table in text:
        for row_index, row in enumerate(table):
            items = []
            for col_index, col in enumerate(row):
                dic.setdefault(f'Col{col_index}', [])
                if row_index <= header:
                    if table[row_index][col_index] not in dic.get(f'Col{col_index}', ''):
                        dic[f'Col{col_index}'].append(table[row_index][col_index])
                if table[row_index][col_index] in dic.get(f'Col{col_index}', ''):
                    header = row_index
                else:
                    head = ' '.join(dic.get(f'Col{col_index}', ''))
                    item = f'{head} - {table[row_index][col_index]},'
                items.append(item)
            row_line = ' '.join(items)
            table_line.append(row_line)
        tablesfin.append(table_line)

    # for table in tablesfin:
    # print(f'\n ------ TABLES ------\n {table}\n')

    if tablesfin:
        return tablesfin
    else:
        return None


def img_ocr(location, filename):  # For Image/Scanned PDF to text
    """
    Opens PNG image (single page) and runs OCR model to extract text
    :param location: Location of PNG image
    :param filename: Name of PNG image
    :return: Text extracted from scanned image (string)
    """
    total_text = ''
    for page in range(1, total_pages):  # tqdm(range(1, total_pages), desc='Converting images to text. . .'):
        loc = f'{location}/{filename}_{page}.png'
        image = cv2.imread(loc)
        reader = easyocr.Reader(['en'],
                                recog_network='custom_example')  # recog_network='custom_example' this needs to run only once to load the model into memory
        try:
            result = reader.readtext(loc, height_ths=0.2,
                                     ycenter_ths=0.3, width_ths=0.5, paragraph=True, decoder='wordbeamsearch',
                                     y_ths=0.2,
                                     x_ths=50)
        except Exception as e:
            result = ''  # Exception occured.
            print(e)
        # paragraph=True)  # , rotation_info=[90, 180, 270], y_ths=1, x_ths=0.09, height_ths=0.5, ycenter_ths=0.5, width_ths=0.5
        cv2.startWindowThread()
        for (bbox, text) in result:  # , prob
            # display the OCR'd text and associated probability
            # print("[INFO] {:.4f}: {}".format(prob, text))
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # cleanup the text and draw the box surrounding the text along
            # with the OCR'd text itself
            cv2.rectangle(image, tl, br, (0, 0, 255), 4)
            cv2.putText(image, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 90, 200), 5)

        file = open(f"C:/Data/Output/{filename}_OCR.txt", 'a')
        for (bbox, text) in result:  # , prob
            total_text += str(text) + '\n'
            file.write(str(text))
            file.write('\n')
        file.close()

        # Show the output image
        '''
        cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
        cv2.imshow("PDF Output", image)
        cv2.waitKey(2)
        '''

    return str(total_text)


def read_dic():
    """
    Read dictionary mappings for converting entity names to Normal text
    :return: None
    """
    global key_mappings
    key_mappings = {}
    with open('./Outputs/EntityDic.text', 'r') as f:
        for line in f:
            key, value = line.split(',')
            key_mappings.fromkeys(key)
            key_mappings[key] = value


def ner(pdf, titles, im_loc, page_limits=(0, 0)):
    """
    Takes PDF runs it page by page to extract its text using OCR or text extraction to run NER model and save its output as text summary and temp HTML render
    :param page_limits: start and end page number for extraction
    :param pdf: PDF file location
    :param titles: Name of PDF file
    :param im_loc: Location for saving PNG image in case of scanned page
    :return: None
    """

    # with Pdf.open(pdf, allow_overwriting_input=True) as pdf_f:  # To decrypt Permission pdfs
    #     pdf_f.save()
    data = ''  # Data string variable to save all text data of PDF
    tagger = SequenceTagger.load(
        r'E:\PycharmProjects\DL\Doc_IMG-OCR\trainer\resources\taggers\2048_Aug2\final-model.pt')  # 2048layers_std
    # print(tagger)
    open(f"C:/Data/Output/{titles}_OCR.txt", "w").close()  # Clear/Wipe OCR txt file
    open(f"C:/Data/Output/{titles} tables.csv", "w").close()  # Clear/Wipe if there is older version of table.csv
    rsrcmgr = PDFResourceManager()
    retstr = BytesIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=15, line_margin=2, boxes_flow=1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    fp = open(pdf, 'rb')
    parser = PDFParser(fp)  # For getting total pages
    document = PDFDocument(parser)  # For getting total pages
    total_pages = resolve1(document.catalog['Pages'])['Count']  # For making progress bar
    pbar = tqdm(total=total_pages, desc='Reading PDF')

    ##############
    # Prediction #
    ##############
    start = page_limits[0]
    end = page_limits[1]
    if start == 0:  # if no page range defined set last page as last page of PDF
        end = total_pages
    for pagenum, page in enumerate(pdfpage.PDFPage.get_pages(fp, check_extractable=True)):
        pagenum += 1  # To start page count from 1
        if start <= pagenum <= end:
            interpreter.process_page(page)
            if len(retstr.getvalue()) < 50:
                # print(f'>> OCR PAGE >>{retstr.getvalue()} <<<<<<< Page number: {pagenum + 1}<<<<< ! ! ! ')
                # Page is OCR only
                pdf2img(pdf, titles, pagenums=pagenum)  # Convert page to image
                data += img_ocr(im_loc, titles)  # Get OCR from converted image
                try:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice_ocr')  # Run OCR based table extraction

                except Exception as e:
                    page_tables = None
                    print(f'Page  {pagenum} \n!!> {e} .')
                    continue
            else:
                try:

                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice')  # Returns list of tables in the specified page
                except IndexError:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice_ocr')
                except Exception as e:
                    page_tables = table_extraction(pdf, titles, pagenum,
                                                   'lattice_ocr')
                    print(f'\nPage  {pagenum} !!> {e} .')

                data += retstr.getvalue().decode('ascii',
                                                 'ignore')  # add extracted text from bytesIO to data variable
                data = data.replace('\x0c', ' ')  # Remove useless character

            ##################################################################
            # NEW LOGIC                                                      #
            # Append table NER extractions to table csv file after each page #
            ##################################################################
            '''
            TABLE APPEND NER
            try:    # If page has a large diagram, catch Out of memory exception of CUDA
                if page_tables:
                    tok_table_lines, tok_line, extraction = [], None, []
                    for table in page_tables:
                        for line in table:
                            tok_table_lines.append(Sentence(line, use_tokenizer=True))
                    for tok_line in tok_table_lines:
                        tagger.predict(tok_line)
                    with open(f"C:/Data/Output/{titles} tables.csv", 'a', newline='', encoding="utf-8") as f:
                        for tok_line in tok_table_lines:
                            for entity in tok_line.get_spans('ner', min_score=threshold):
                                if str(entity.tag) != 'tenderid' and entity.tag != 'marking' and entity.tag != '<unk>':
                                    # print(f'-- Adding table extraction to CSV file --')
                                    extraction.append(f'"{entity.text} , {entity.tag}"')
                        if extraction:
                            f.write("-------------------------- , ---------------------- \n")
                            f.write('"Attribute , Type"')
                            f.write("\n")
                            res = list(OrderedDict.fromkeys(extraction))
                            for tags in res:
                                f.write(tags)
                                f.write("\n")
                            f.write("\n -------------------------- , ---------------------- \n")



                retstr.truncate(0)
                retstr.seek(0)
            except RuntimeError:
                print(f'Too Large page CUDA OFM error')
                retstr.truncate(0)
                retstr.seek(0)
            '''
        retstr.truncate(0)  # Clear byte stream for new page extraction
        retstr.seek(0)
        pbar.update(1)
    pbar.close()
    print('///////// File Read Complete //////////// ')
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(data)

    for num, sentence in enumerate(tqdm(sentences, desc=f'Predicting labels . . .')):
        try:  # If page has a large diagram, catch Out of memory exception of CUDA
            tagger.predict(sentence)
        except RuntimeError:
            print(f'Too Large page CUDA OFM error')
            continue

    ##################
    #   LOG OUTPUT   #
    ##################
    logfile = f'C:/Data/Output/{titles}_summary.txt'
    sen = {}  # Dictionary for removing duplicate sentences
    dic = {}  # Dictionary for saving output under Cable Name
    misc = {}  # Dictionary for packing and marking tags
    with open(logfile, 'w', newline='', encoding="utf-8") as f:
        print('Writing values to file. . . ')
        print(f'////////////////////////////////////////////////////////////////////////////////')
        print(f'//////////////////  E X T R A C T I O N    R E S U L T  ///////////////////////')
        print(f'-------------------------------------------------------------------------------')
        print(f'//  Text ,   Entity Tag ,  Confidence percentage   //')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        f.writelines(f'//////////////////  E X T R A C T I O N    R E S U L T  //////////////////////// \n')
        f.writelines(f'//////////////////////////////////////////////////////////////////////////////// \n')
        cable_name = None
        cable_list = [
            'cable', 'line', 'covered conductors'
        ]  # 'cable', 'lt', 'lt cable', 'cables']
        forbidden = [
            'applicable', 'standard', 'standards', 'accessory', 'accessories', 'pipe',
            'pipe',
            'applicable standard', 'applicable standards', 'system', 'switch',
            'station', 'circuit', 'isolator', 'hdpe', 'mccb', 'breaker', 'pole', 'duct', 'fence', 'meter',
            'switchgear', 'bus', 'control', 'loose','lugs'
            'transformer', 'surge', 'insulator', 'ring', 'smoke', 'lug', 'ABBREVIATION'
        ]

        #########################################################
        #        CUSTOMISATION FOR TENDER SPECIFICATIONS        #
        #########################################################

        cable_flag = 1
        for sentence in sentences:
            sen.setdefault(sentence.to_plain_string(),
                           [])  # Create list initialised dictionary where Key = sentence
            for entity in sentence.get_spans('ner', min_score=threshold):
                if str(entity.tag) == 'cableItype' and entity.score > 0.8:
                    for y in cable_list:
                        if entity.text.lower().find(y) != -1 and len(entity) > 1:
                            cable_flag = 1
                            cable_name = entity.text.upper()
                            break
                    continue
                else:
                    continue

            for entity in sentence.get_spans('ner', min_score=threshold):
                if str(entity.tag) == 'cableItype' and entity.score > 0.8:  # Check all entities if in Forbidden list
                    print(f'- - - - Cable {entity.text.upper()}- - - - ')
                    for x in forbidden:  # Filtering results of Cable Type
                        if entity.text.lower().find(x) != -1 and len(entity.text) == len(x):
                            print(f'X X X X X Cable Type Rejected {entity.text.upper()} > {x.upper()} X X X X X')
                            cable_flag = 0
                            break
                    break
                else:
                    continue

            if cable_flag == 1 and cable_name is not None:  # If cable is present in sentence

                for entity in sentence.get_spans('ner', min_score=threshold):
                    if entity.tag != 'cableItype' and str(entity.tag) != 'tenderid' and str(
                            entity.tag) != 'standard' and entity.tag != '<unk>':

                        if entity.tag in ['marking', 'packing'] and len(
                                entity) > 2:  # Removing entity output and less than 2 word entities from final text for markings
                            misc.setdefault(entity.tag, [])
                            misc[entity.tag].append(
                                f'{sentence.to_plain_string()}')  # Adding specific formatted line to final text file
                            print(f'= = = = = Cable Type  {cable_name.upper()} = = = = =')
                            print(
                                f'// =={entity.text}  ====  {entity.tag} :::: {(round(entity.score, 4) * 100)}% :::://')  # Debugging/CLI output
                            continue
                        elif (500 > len(sentence.to_plain_string()) > len(
                                entity.text) + 4) and entity.tag != 'marking' and entity.tag != 'packing':
                            sen[sentence.to_plain_string()].append(
                                # Adding to sentence dictionary to avoid multiple same sentences
                                f'Tag: >> {entity.text}, {entity.tag}')

                            # f.writelines(f'> {entity.text}, {entity.tag}-[{(round(entity.score, 4) * 100)}%] \n')
                            # f.writelines(f'>> {sentence.to_original_text()}, {entity.tag} \n\n')
                            print(f'{sentence.get_spans()}'
                                f'// =={entity.text}  ====  {entity.tag} ::CONF:: {(round(entity.score, 4) * 100)}% :::://')  # Debugging/CLI output

                if len(sen[sentence.to_plain_string()]) > 0:  # Only add sentence to Cable dictionary if it has Entities
                    dic.setdefault(cable_name, [])  # Initialise blank value list in cable type dictionary
                    dic[cable_name].append(sentence.to_plain_string())

        ##############################
        # CABLE LEVEL FILTRATION    #
        ##############################
        for k, values in dic.items():
            f.writelines(f'______________________________________________________________________\n')
            f.writelines(f'CABLE TYPE: {k}')
            f.writelines(f'\n______________________________________________________________________\n\n')
            for line in values:
                tags = sen[line]
                unq_tags = list(OrderedDict.fromkeys(tags))  # To remove multiple same Keys from different similar sentences
                for tag in unq_tags:
                    f.writelines(f'{tag}\n')
                f.writelines(f'Sentence: {line}\n\n')
        f.writelines(f'X----------------------------------X-----------------------------------X \n')

        print(f'|___________________________________END OF FILE___________________________________________|')

        # sorted_dic = OrderedDict(sorted(final.items()))
        # for k, v in sorted_dic.items():
        #     if len(v) > 0:
        #         f.writelines(f'CABLE TYPE: {k} \n\n')
        #         res = list(OrderedDict.fromkeys(v))  # To remove multiple same Keys from different similar sentences
        #         for tags in res:
        #             f.writelines(f'{tags}')
        #
        #         f.writelines(f'X----------------------------------X-------------------------------X \n')

        for k, v in misc.items():
            if len(v) > 0:
                res = list(OrderedDict.fromkeys(v))  # Remove multiple same Keys from different similar sentences
                f.writelines(f'\n______________________________________________________________________\n')
                f.writelines(f'{str(k).upper()}')
                f.writelines(f'\n______________________________________________________________________\n')
                for count, tags in enumerate(res):
                    f.writelines(f'\nSentence {count + 1} : {tags}\n')

        f.writelines(f'\nX----------------------------------X-----------------------------------X \n')

    colors = {

        "default": "#FF40A3",
        "O": "#ddd",
    }
    actual = render_ner_html(sentences, title=titles, colors=colors)

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(actual)
    webbrowser.open(url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='threshold', help='Default = 0.8  , Limit: [0,1]', type=float,
                        default=0.8)
    parser.add_argument('-s', dest='s_page', help='Default = 0  , Limit: [0-99999]', type=int,
                        default=0)
    parser.add_argument('-e', dest='e_page', help='Default = 0  , Limit: [0-99999]', type=int,
                        default=0)
    parser.add_argument('-f', dest='pdfname', help='Name of PDF to be processed', type=str)
    args = parser.parse_args()

    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'//////////////////     INTELLIGENT - DOCUMENT - PROCESSOR    ///////////////////')
    print(f'///////////////////             HAVELLS NEW TECH             ///////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'////////////////////////////////////////////////////////////////////////////////')
    print(f'-------------------------------------------------------------------------------')
    print(f'|::::::::::::::::   Threshold Confidence : {args.threshold}   :::::::::::::::::|')
    print(f'|::::::::::::::::     PDF to be evaluated : {str(args.pdfname).upper()}    :::::::::::::::::|')
    print(f'|::::::::::::::::     Page limits  : {args.s_page}  - {args.e_page}  :::::::::::::::::|')
    print(f'|______________________________________________________________________________|')

    threshold = args.threshold
    pdfname = args.pdfname
    pages = (args.s_page, args.e_page)
    # pdfname = 'PGCIL'#PGCIL BSES TENDER EIL Specs BHEL
    PDF_file = f'C:/Data/test/{pdfname}.pdf'
    img_loc = r'C:/Data/Output/OCR/images'

    # pdf2img(PDF_file, pdfname)
    # img_ocr(img_loc, pdfname)
    # searchable_ocr(img_loc)  # For converting image to text embedded PDF
    ner(PDF_file, pdfname, img_loc, pages)

"""
CLI command :
                E:\PycharmProjects\DL\venv\scripts\python.exe E:\PycharmProjects\DL\Doc_IMG-OCR\main.py -c 0.7 -f TEND00012790 //
"""

import csv

import easyocr
import pdf2image
from flair.data import Sentence
from pdf2image import convert_from_path  # For scanned PDFs
from PIL import Image
from pdfminer import high_level # Convert text PDF to text
from pdfminer import pdfpage
import cv2
import pytesseract
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
import nltk
from flair.models import SequenceTagger

def pdf2img(PDF):
    # print(len(PDF))
    pages = convert_from_path(PDF, 500, poppler_path=r"C:\poppler-21.09.0\Library\bin", first_page=150,
                              last_page=160, timeout=10000)
    #
    # # Counter to store images of each page of PDF to image
    image_counter = 1
    #
    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> page_1.jpg
        # PDF page 2 -> page_2.jpg
        # PDF page 3 -> page_3.jpg
        # ....
        # PDF page n -> page_n.jpg
        filename = "page" + str(image_counter) + ".png"

        # Save the image of the page in system
        page.save(r'C:\Users\33669\PycharmProjects\OCR\pdf2img\ ' + filename)
        print('Saved page number ' + str(image_counter))
        # Increment the counter to update filename
        image_counter = image_counter + 1


def searchable_ocr(img):  # From image to searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(img, config='--oem 1', extension='pdf')
    print(pytesseract.image_to_string(img))
    with open(r'C:\Users\33669\PycharmProjects\OCR\outputs\Searchable.pdf', 'w+b') as f:
        f.write(pdf)


def img_ocr(loc):  # For Image/Scanned PDF to text
    image = cv2.imread(loc)
    reader = easyocr.Reader(['en'],
                            recog_network='custom_example')  # , recog_network='custom_example' this needs to run only once to load the model into memory
    result = reader.readtext(img_loc,
                             paragraph=True)  # , rotation_info=[90, 180, 270], y_ths=1, x_ths=0.09, height_ths=0.5, ycenter_ths=0.5, width_ths=0.5
    cv2.startWindowThread()
    for (bbox, text) in result:  # , prob
        # display the OCR'd text and associated probability
        # print("[INFO] {:.4f}: {}".format(prob, text))
        print(text)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 90, 200), 8)

    file = open(r"C:\Users\33669\PycharmProjects\OCR\outputs\OCR_OUT.txt", 'w')
    for (bbox, text) in result:  # , prob
        file.write(str(text))
        file.write('\n')
    file.close()
    # show the output image
    cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
    cv2.imshow("PDF Output", image)
    cv2.waitKey(0)


def text_pdf(pdf):  # For PDF that is in text selectable format char_margin=30, line_margin=2, boxes_flow=1
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams(char_margin=30, line_margin=2, boxes_flow=1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    caching = True
    pagenos = set()
    fp = open(pdf, 'rb')
    writer = csv.writer(open(r"H:\Code\Doc_IMG-OCR\Tender_Text1.csv", 'w', newline=''))
    #file = open(r"H:\Code\Doc_IMG-OCR\Tender_Text1.txt", 'a')
    #file.truncate(0)
    for pagenumber, page in enumerate(pdfpage.PDFPage.get_pages(fp, check_extractable=True)):
        print(pagenumber)
        if pagenumber == 150:
            interpreter.process_page(page)
            data = retstr.getvalue()
            sent_text = nltk.sent_tokenize(data)
            for sentence in sent_text:
                tokenized_text = nltk.word_tokenize(sentence)
                print(tokenized_text)
                for word in tokenized_text:
                    writer.writerow([word])
                writer.writerow('\n')
                #file.write(str(tokenized_text))
    #file.close()

def ner(file):
    sentences = []
    tagger = SequenceTagger.load('H:/Code/Doc_IMG-OCR/trainer/resources/taggers/ner-flert/final-model.pt')
    with open(file, 'r') as read_obj:
        for line in read_obj:
            sentence = Sentence(line)
            tagger.predict(sentence)
            for entity in sentence.get_spans('ner'):
                print(entity)


if __name__ == '__main__':
    PDF_file = r'H:\Code\Data\EIL Specs.pdf'
    img_loc = r'C:\Users\33669\PycharmProjects\OCR\pdf2img\K2.jpg'
    filename = 'PGCIL'
    # img_ocr(img_loc)
    # pdf2img(PDF_file)
    # searchable_ocr(img_loc) # For converting image to text embedded PDF
    #text_pdf(PDF_file)
    ner('H:/Code/Doc_IMG-OCR/trainer/ner/test/'+filename +'.csv')

'''
H:\Code\Doc_IMG-OCR\lib\Scripts\python.exe H:/Code/Doc_IMG-OCR/main.py
2021-11-25 14:16:41,580 loading file H:/Code/Doc_IMG-OCR/trainer/resources/taggers/ner-flert/final-model.pt
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [19]: "contractor"   [− Labels: conductor (0.9558)]
Span [23]: "contractor"   [− Labels: conductor (0.9426)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [19]: "contractor"   [− Labels: conductor (0.9558)]
Span [23]: "contractor"   [− Labels: conductor (0.9426)]
Span [2]: "xlpe"   [− Labels: insulation (0.9984)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [44,45,46]: "pvc insulated cables"   [− Labels: insualtion (0.729)]
Span [51,52,53]: "xlpe insulated cables"   [− Labels: insualtion (0.7563)]
Span [26]: "pvc"   [− Labels: insulation (0.6504)]
Span [2,3]: "xlpe insulated"   [− Labels: insulation (0.9897)]
Span [2,3]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [19]: "stranding"   [− Labels: conductor (0.9528)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [19]: "contractor"   [− Labels: conductor (0.9558)]
Span [23]: "contractor"   [− Labels: conductor (0.9426)]
Span [2]: "xlpe"   [− Labels: insulation (0.9984)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [44,45,46]: "pvc insulated cables"   [− Labels: insualtion (0.729)]
Span [51,52,53]: "xlpe insulated cables"   [− Labels: insualtion (0.7563)]
Span [26]: "pvc"   [− Labels: insulation (0.6504)]
Span [2,3]: "xlpe insulated"   [− Labels: insulation (0.9897)]
Span [2,3]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [19]: "stranding"   [− Labels: conductor (0.9528)]
Span [7,8]: "pvc insulated"   [− Labels: insualtion (0.9988)]
Span [1]: "xlpe"   [− Labels: insulation (0.9981)]
Span [3]: "xlpe"   [− Labels: insulation (0.9794)]
Span [3]: "xlpe"   [− Labels: insulation (0.9991)]
Span [12,13]: "extruded pvc"   [− Labels: inner (0.5225)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [12,13]: "aluminium wires"   [− Labels: armouring (0.7419)]
Span [17]: "xlpe"   [− Labels: insulation (0.9988)]
Span [1]: "pvc"   [− Labels: insulation (0.4401)]
Span [7]: "insulated"   [− Labels: insualtion (0.5475)]
Span [5]: "stranded"   [− Labels: conductor (0.9962)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [13,14]: "extruded pvc"   [− Labels: inner (0.7767)]
Span [6]: "extruded"   [− Labels: inner (0.3093)]
Span [7]: "pvc"   [− Labels: insulation (0.5888)]
Span [1,2,3]: "pvc control cables"   [− Labels: inner (0.4062)]
Span [7]: "insulated"   [− Labels: insualtion (0.9887)]
Span [5]: "stranded"   [− Labels: conductor (0.9958)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [7]: "pvc"   [− Labels: insulation (0.5043)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [19]: "contractor"   [− Labels: conductor (0.9558)]
Span [23]: "contractor"   [− Labels: conductor (0.9426)]
Span [2]: "xlpe"   [− Labels: insulation (0.9984)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [44,45,46]: "pvc insulated cables"   [− Labels: insualtion (0.729)]
Span [51,52,53]: "xlpe insulated cables"   [− Labels: insualtion (0.7563)]
Span [26]: "pvc"   [− Labels: insulation (0.6504)]
Span [2,3]: "xlpe insulated"   [− Labels: insulation (0.9897)]
Span [2,3]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [19]: "stranding"   [− Labels: conductor (0.9528)]
Span [7,8]: "pvc insulated"   [− Labels: insualtion (0.9988)]
Span [1]: "xlpe"   [− Labels: insulation (0.9981)]
Span [3]: "xlpe"   [− Labels: insulation (0.9794)]
Span [3]: "xlpe"   [− Labels: insulation (0.9991)]
Span [12,13]: "extruded pvc"   [− Labels: inner (0.5225)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [12,13]: "aluminium wires"   [− Labels: armouring (0.7419)]
Span [17]: "xlpe"   [− Labels: insulation (0.9988)]
Span [1]: "pvc"   [− Labels: insulation (0.4401)]
Span [7]: "insulated"   [− Labels: insualtion (0.5475)]
Span [5]: "stranded"   [− Labels: conductor (0.9962)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [13,14]: "extruded pvc"   [− Labels: inner (0.7767)]
Span [6]: "extruded"   [− Labels: inner (0.3093)]
Span [7]: "pvc"   [− Labels: insulation (0.5888)]
Span [1,2,3]: "pvc control cables"   [− Labels: inner (0.4062)]
Span [7]: "insulated"   [− Labels: insualtion (0.9887)]
Span [5]: "stranded"   [− Labels: conductor (0.9958)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [7]: "pvc"   [− Labels: insulation (0.5043)]
Span [56,57]: "xlpe insulated"   [− Labels: insualtion (0.9989)]
Span [41,42]: "xlpe insulated"   [− Labels: insualtion (0.9988)]
Span [22]: "contractor"   [− Labels: conductor (0.5463)]
Span [5,6]: "xlpe insulated"   [− Labels: insualtion (0.9596)]
Span [18]: "voltage"   [− Labels: marking (0.5689)]
Span [20,21]: "xlpe insulated"   [− Labels: insualtion (0.9988)]
Span [3,4]: "xlpe insulated"   [− Labels: insulation (0.9851)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9991)]
Span [10,11]: "pvc insulated"   [− Labels: insualtion (0.9992)]
Span [3]: "stranded"   [− Labels: conductor (0.9965)]
Span [3,4,5,6,7]: "cable size cable type 1"   [− Labels: marking (0.7834)]
Span [4]: "xlpe"   [− Labels: insulation (0.999)]
Span [5]: "xlpe"   [− Labels: insulation (0.999)]
Span [6]: "xlpe"   [− Labels: insulation (0.9989)]
Span [7]: "xlpe"   [− Labels: insulation (0.9988)]
Span [4]: "xlpe"   [− Labels: insulation (0.9982)]
Span [6]: "xlpe"   [− Labels: insulation (0.9979)]
Span [7]: "xlpe"   [− Labels: insulation (0.9979)]
Span [5]: "pvc"   [− Labels: inner (0.3307)]
Span [3]: "pvc"   [− Labels: insulation (0.6103)]
Span [19]: "contractor"   [− Labels: conductor (0.9558)]
Span [23]: "contractor"   [− Labels: conductor (0.9426)]
Span [2]: "xlpe"   [− Labels: insulation (0.9984)]
Span [4,5]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [44,45,46]: "pvc insulated cables"   [− Labels: insualtion (0.729)]
Span [51,52,53]: "xlpe insulated cables"   [− Labels: insualtion (0.7563)]
Span [26]: "pvc"   [− Labels: insulation (0.6504)]
Span [2,3]: "xlpe insulated"   [− Labels: insulation (0.9897)]
Span [2,3]: "pvc insulated"   [− Labels: insualtion (0.9989)]
Span [19]: "stranding"   [− Labels: conductor (0.9528)]
Span [7,8]: "pvc insulated"   [− Labels: insualtion (0.9988)]
Span [1]: "xlpe"   [− Labels: insulation (0.9981)]
Span [3]: "xlpe"   [− Labels: insulation (0.9794)]
Span [3]: "xlpe"   [− Labels: insulation (0.9991)]
Span [12,13]: "extruded pvc"   [− Labels: inner (0.5225)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [12,13]: "aluminium wires"   [− Labels: armouring (0.7419)]
Span [17]: "xlpe"   [− Labels: insulation (0.9988)]
Span [1]: "pvc"   [− Labels: insulation (0.4401)]
Span [7]: "insulated"   [− Labels: insualtion (0.5475)]
Span [5]: "stranded"   [− Labels: conductor (0.9962)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [13,14]: "extruded pvc"   [− Labels: inner (0.7767)]
Span [6]: "extruded"   [− Labels: inner (0.3093)]
Span [7]: "pvc"   [− Labels: insulation (0.5888)]
Span [1,2,3]: "pvc control cables"   [− Labels: inner (0.4062)]
Span [7]: "insulated"   [− Labels: insualtion (0.9887)]
Span [5]: "stranded"   [− Labels: conductor (0.9958)]
Span [6]: "armoured"   [− Labels: conductor (0.4698)]
Span [7]: "pvc"   [− Labels: insulation (0.5043)]
Span [56,57]: "xlpe insulated"   [− Labels: insualtion (0.9989)]
Span [41,42]: "xlpe insulated"   [− Labels: insualtion (0.9988)]
Span [22]: "contractor"   [− Labels: conductor (0.5463)]
Span [5,6]: "xlpe insulated"   [− Labels: insualtion (0.9596)]
Span [18]: "voltage"   [− Labels: marking (0.5689)]
Span [20,21]: "xlpe insulated"   [− Labels: insualtion (0.9988)]
Span [13,14]: "galvanized steel"   [− Labels: armouring (0.7288)]
Span [24,25]: "extruded pvc"   [− Labels: inner (0.6097)]
Span [3]: "xlpe"   [− Labels: insulation (0.9992)]
Span [24,25]: "xlpe insulated"   [− Labels: insulation (0.9835)]
Span [42]: "xlpe"   [− Labels: insulation (0.9424)]

Process finished with exit code 0
'''
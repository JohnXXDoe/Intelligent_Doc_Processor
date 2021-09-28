import easyocr
import pdf2image
from pdf2image import convert_from_path  # For scanned PDFs
from PIL import Image
import PyPDF2  # Convert text PDF to text
import cv2

PDF_file = r"C:\Users\33669\PycharmProjects\OCR\pdf2img\test.pdf"
#print(len(PDF_file))
# pages = convert_from_path(PDF_file, 500, poppler_path=r"C:\poppler-21.09.0\Library\bin", first_page=22, last_page=27, timeout=10000)
#
# # Counter to store images of each page of PDF to image
# image_counter = 1
#
# # Iterate through all the pages stored above
# for page in pages:
#     # Declaring filename for each page of PDF as JPG
#     # For each page, filename will be:
#     # PDF page 1 -> page_1.jpg
#     # PDF page 2 -> page_2.jpg
#     # PDF page 3 -> page_3.jpg
#     # ....
#     # PDF page n -> page_n.jpg
#     filename = "page" + str(image_counter) + ".jpg"
#
#     # Save the image of the page in system
#     page.save(r'C:\Users\33669\PycharmProjects\OCR\pdf2img\ ' + filename)
#     print('Saved page number ' + str(image_counter))
#     # Increment the counter to update filename
#     image_counter = image_counter + 1

img_loc = 'C:/Users/33669/PycharmProjects/OCR/pdf2img/4.png'
model = r'C:\Users\33669\PycharmProjects\OCR\classification\Lib\site-packages\easyocr\model\english_g2.pth'
image = cv2.imread(img_loc)
reader = easyocr.Reader(['en'], model_storage_directory=model)  # this needs to run only once to load the model into memory
result = reader.readtext(img_loc, paragraph=False, mag_ratio=2, y_ths=0.05, x_ths=1.5, width_ths=1.5)#, rotation_info=[90, 180, 270]
cv2.startWindowThread()
for (bbox, text, prob) in result:
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

file = open("OCR_OUT.txt", "a")
for (bbox, text, prob) in result:
    file.write(str(text))
    file.write('\n')
file.close()
# show the output image
cv2.namedWindow('PDF Output', cv2.WINDOW_NORMAL)
cv2.imshow("PDF Output", image)
cv2.waitKey(0)

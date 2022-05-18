# Doc_IMG-OCR
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS (RE ADD AFTER GOING PUBLIC)-->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JohnXXDoe/Intelligent_Doc_Processor">
    <img src="images/5860.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Intelligent Document Processor</h3>

  <p align="center">
    Read large scale PDF documents and extract important entities (Lots of customisations too!)
    <br />
    <a href="https://github.com/JohnXXDoe/Intelligent_Doc_Processor"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/JohnXXDoe/Intelligent_Doc_Processor/tree/master/trainer">Training sub-directory</a>
    ·
    <a href="https://github.com/JohnXXDoe/Intelligent_Doc_Processor/tree/master/Outputs">Sample Outputs</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#info">Information</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#roadmap">Road Map</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Project was created out of a need to process large tender documents and creating a requirements sheet after going thorugh the large PDF (80-900 pages). This task can be eased with the help of a custom NER.
This project aims to reduce and ultimately eleminate the need to go through such tenders manually to find its tags.

Currently this program is trained to extract ~17 Tags which range in their average word length.

<p align="right">(<a href="#top">back to top</a>)</p>


### Customized rejection filter to classify and remove unnecessary extractions
[![Rejection Screen Shot][rejection]]
### Built With

* [Camelot](https://github.com/camelot-dev)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [Fair](https://github.com/flairNLP/flair)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Download this project and change paths in main.py, place the visualNER file in the library bin folder for Flair.

### Prerequisites

Camelot
EasyOCR
Flair
pdfminer.six
poppler

### Info

1. To run on default settings i.e no custom NER no custom OCR model open main.py and run it via CLI, pass the following command in CLI to get a HTML + txt file output of NER.
   ```sh
   Your_path\python.exe Project_Path\Doc_IMG-OCR\main.py -c 0.7 -f EIL
   ```
   -c between [0,1] for minimum threashold confidace, -f to define the name of file.pdf to be run.
   The file can be either serchable PDF or a scanned PDF or mix of both, the program takes into account individual pages one by one to check if its OCR or scanable.

2. To trian NER model - Create a corpus first by using NERDatabase.py which takes in PDF+txt file as input and produces a B-I-O tagged txt document that is ready for Flair to take in as a training copus.
    Run the training.py with the copus to train a NER model. You can also choose a pretrianed Language model/Word Embedding (word2vec, Roberta etc.)
    Use trained model to run NER output for custom tags.

3. To trian OCR model - Run the OCR only mode by modifying main.py's main function.
    Run this OCR function on a scanned PDF with word snipped saving ON.
    Create a corpus using the extracted word images and manual tagging and maintaing a .csv file.
    
4. The output of this program generates 2 files - HTML with CSS rendering of the PDF and marked entities to visualise the PDF after marking NER tags. Second, a consise txt file containing the following-
    Extracted tag, Tag name, Confidence
    Sentence that contains the tag
    
    To remove multiple sentences being mentioned again and maintain a dictionary(map) to sort tags sentence wise.
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

This project can function as a baseline for creating your custom NER document processor that can take in PDFs in any form and output custom tag identifications.
For getting a decent NER recognition atleast ~100 sentences should be provided per NER tag that needs to be trained.
The OCR engine that is using EasyOCR works really well and with a little finetune it even detects some confusing chars.

After the addition of table detection this project will be able to keep data intact which are in tables instead of loosing its format and ultimatly its context.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Add Changelog
- [-] Add Screenshots
- [ ] Add Training explanation
- [ ] Decide License

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

To be added.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Utkarsh Kharayat - utkarshk.co.16@nsit.ac.in

Project Link: [https://github.com/your_username/repo_name](https://github.com/JohnXXDoe/Doc_IMG-OCR/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Open source libraries used to make this project possible:

* [Camelot](https://github.com/camelot-dev)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [Fair](https://github.com/flairNLP/flair)
* [PDFminerSIX](https://github.com/pdfminer/pdfminer.six)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/JohnXXDoe/Doc_IMG-OCR/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/JohnXXDoe/Doc_IMG-OCR/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/JohnXXDoe/Doc_IMG-OCR/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/JohnXXDoe/Doc_IMG-OCR/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/JohnXXDoe/Doc_IMG-OCR/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/utkarsh-kharayat-23068b179
[product-screenshot]: images/IDP.PNG
[rejection]: images/Rejection.PNG

# DigiLut

# Code for the 5th solution for the DigiLut challenge.

https://app.trustii.io/leaderboard-v2/1526

Dataset description:

_**Trustii.io and the Foch Hospital are launching a new computer vision competition with a ¬25,000 prize pool:  Detecting pathological regions (A lesions) on transbronchial biopsies from lung transplant patients**_

The DigiLut Challenge, a collaborative project between Trustii.io and Foch Hospital and supported by the Health Data Hub and Bpifrance, aims to develop a medical decision support algorithm to diagnose graft rejection episodes in lung transplant patients by detecting pathological regions (A lesions) on transbronchial biopsies.

Lung transplantation is the only treatment available for certain types of respiratory failure. The success of transplantation depends on the occurrence of acute rejection episodes (type A rejection), which can lead to irreversible graft rejection and ultimately the patient's death.

Foch Hospital, France's leading lung transplant center and a top institution in Europe, has performed over 1,000 transplants, including more than 70 annually over the past three years, with an active file of almost 600 patients. The hospital is also a leader in lung transplant research, focusing on ex-vivo reconditioning and rejection.

Digital pathology, a sub-field of pathology that focuses on data management based on information generated from digitized specimen slides, is revolutionizing conventional anatomo-pathology. This evolution is seen in day-to-day practice (training interns, sharing expertise) and with the prospect of diagnostic aid tools based on Artificial Intelligence (AI). In the future, AI tools are expected to improve the quality and reliability of diagnoses.

![image](https://github.com/user-attachments/assets/09b6a607-d204-40aa-90f1-e84c288e3bee)


### 1/ Core Idea Behind DigiLut Challenge

**Data Flow for Graft Rejection Diagnosis Using Digital Pathology:**

*   **\- Current Situation:** Multiple pathways and significant disconnection for in vitro and clinical data, hampering integrated interpretation.
*   **\- DigiLut Challenge Objective:** Develop an algorithm capable of generating region-of-interest boxes for type A lesions on new unannotated slides, using both annotated and unannotated data provided.

### 2/ Data

The anonymized database constructed from digitized biopsy slides includes annotations of the zones of interest, created by an international panel of expert pathologists. This database is made available to competitors who will be challenged to create an algorithm to detect graft rejection and its severity.

_Data Scientists can find an additional dataset that can help them perform transfer learning, which maps the presence or absence of lesion for each of  the lesions where:_

           _**- 1** corresponds to the **presence of at least one lesion** in at least one of the 8 levels of the given image (on the slide \_a.tif or on the slide \_b.tif or on both)._

*             _**- 0** corresponds to the **absence of a lesion** in the given image (neither of the slides \_a.tif nor \_b.tif)_

  
 _As a reminder, **not all the images have been annotated in this challenge (25%):**_

_The images that contain at least one bounding box contain at least one lesion (= graft rejection)_

_The images that do not present bounding box can contain rejection zones (lesion = 1) or no rejection zones at all (no lesion = 0)_  

Data is available in the JupyterHub environment, where you can either download it or work on JupyterHub to process it (generate crops) before downloading to your own local development environment.

![image](https://github.com/user-attachments/assets/4966794a-c6e8-484c-92a9-5ad8fceaca85)

Data challenge evaluation strategy

**Data folder contains:**

     **\-** **images**: Contains all .tif files, including annotated and unannotated images.  
               o  These images are intended for model training.  
               o  Most of the images are couples (XXXX\_a.tif and XXXX\_b.tif), or solo images (YYYY\_a.tif) and represent one patient's transbronchial biopsy  
               o  Annotated images are for training the model, while unannotated images can optionally be used for transfer learning.  
               o  You are free to generate crops from the base .tif files and preprocess the images as you see fit.

*        **- train.csv:**
*                  **o Description:** Annotated .tif files intended for training.
*                  **o Columns:**
    *                                 **¤ filename:** Name of the .tif file.
    *                                 **¤ max\_x, max\_y:** Maximum x and y coordinates of the Bounding Boxes (resolutions of the .tif).

                              **¤** _**Coordinates of a bounding box**_ are encoded with four values in pixels: \[x1,y1,x2,y2\]\[x1, y1, x2, y2\]\[x1,y1,x2,y2\] where:

*                                                              **¤   x1** and **y1** are the coordinates of the top-left corner of the bounding box
*                                                              **¤   x2** and **y2** are the coordinates of the bottom-right corner of the bounding box

  Below an example of the coordinates :
![image](https://github.com/user-attachments/assets/3639b149-5c24-4bda-a1c4-7dcb7205f059)

*        **- validation.csv:**
*                  **o Description:** Validation set including .tif files intended for validation.
    *                                 **¤** Includes annotated .tif validation images as well as part of the training data for cross-validation.
    *                                 **¤** Files include .tif annotated cross-wise by one or more pathologists, for which Bounding Boxes need to be predicted.
*                  **o Columns:**
    *                                 **¤ filename:** Name of the .tif file.
    *                                 **¤ Coordinates of a bounding box are encoded with four values in pixels: \[x\_1, y\_1, x\_2, y\_2\]**  where x\_1 and y\_1 are coordinates of the top-left corner of the bounding box and x\_2 and y\_2 are coordinates of bottom-right corner of the bounding box.
    *                                 **¤ max\_x, max\_y:** Maximum x and y coordinates of the Bounding Boxes (resolutions of the .tif).
    *                                 **¤ trustii\_id:** Unique identifier for each record.

  
**Use of external data:** Not allowed. If used, the candidate will be excluded.

This description offers an overview of the Images, train.csv, and validation.csv files, their contents, and their use in the training and validation of the model.

### 3/ Evaluation

_Participants will be evaluated based on the accuracy of their model's predictions on a private test set, cross-annotated by multiple pathologists. The final ranking will be based on the scores of the private leaderboard._

### 4/ Platform - Training Environment

**Data retention on Jupyter environment:**

All files created by a user on the Jupyter environment will be deleted after 48 hours of inactivity. It's important to note that inactivity means that no connection has been made to the environment during this period. Therefore, if you connect to the environment every day, your files will remain persistent.

**JupyterHub Navigator requirement:**

*   Firefox : 127
*   Chrome : 125.0.6422.144
*   Safari : 17
*   Edge : 125

**JupyterLab Notebooks:**

The dataset will be accessible ONLY via the trustii.io JupyterHub environment. Participants will have a dedicated and persistent file system (13GB RAM, 6CPU, and 521GB storage per team). The environment supports real-time collaboration, allowing team members to work on the same notebooks and data. Note: The environment must not be used for training, as there is no GPU. Also, after 48 hours of inactivity, team/user files will be removed.

**Programming Languages:**

The preferred language for the challenge is Python, widely used in data science and machine learning.

**Libraries:**

For data preprocessing, any open-source Python library is permitted (OpenCV, SciKit-Image, numpy, etc.). The Trustii.io JupyterHub environment comes with libraries pre-installed, and additional libraries can be installed as needed by the users.

### 5/ Submission Requirements

Participants will submit their work following a submission template provided by Trustii.io, available in the JupyterHub environment. Only submissions that include the completed zip file (submission template) will be considered. The Trustii.io team will review and run inference on each winning model. Check out the README in the provided submission template in JupyterHub.

### 6/ Innovative Approaches

Participants may use various strategies for utilizing unannotated data, including:

1.  **Transfer Learning:** Pre-train a model on the unannotated dataset, then refine it on the annotated dataset.
2.  **Clustering:** Identify groups of images or regions within images that share similar features using clustering techniques.
3.  **Anomaly Detection:** Train an anomaly detection model on unannotated data to identify deviations suggesting lesions.
4.  **Autoencoders:** Use autoencoders to learn compressed representations of images, which can then be used as input features for a classification or detection model.
5.  **Data Augmentation:** Apply transformations to create an augmented dataset, increasing variability and quantity of training data.
6.  **Ensemble Learning:** Train several models on different subsets of unannotated data, then combine them to improve detection performance.

###   
7/ Timing

The challenge will run from June 14th until August14th, with results announced in September.

### 8/ Partners

This challenge is organized by Foch Hospital in partnership with the Health Data Hub, financed by Bpifrance, and sponsored by the "Grand Défi: Improvement of medical diagnoses through Artificial Intelligence" program led by the French Secretariat General for Investment.

### 9/ Evaluation Metrics

**Challenge Context:**

In this challenge, lesion detection models will be evaluated using metrics that balance precision and recall, with particular attention to minimizing false negatives. The chosen metrics are the Generalized Intersection over Union (GIoU) and the F2 Score.

**GIoU Calculation:**

The Generalized Intersection over Union (GIoU) provides a robust measure of the intersection between the predicted object and the actual object.

Formula:

![image](https://github.com/user-attachments/assets/92188e37-e595-4d73-8801-89412e4ac7ae)

where ![image](https://github.com/user-attachments/assets/f839e6ff-6c23-437e-906f-c1bf7b6dab0d) is the enclosing area and ![image](https://github.com/user-attachments/assets/f0c36720-ce93-454f-ae85-63fc0221f90b) is the union area.
  

**F2 Score Evaluation:**

The F2 Score places more weight on recall than on precision.

Formula:
![image](https://github.com/user-attachments/assets/e93cdb2d-4ad3-46dc-9407-6bbdc15ac231)





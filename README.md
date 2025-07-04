# Predicting Student Dropout Risk Using Demographics and Online Activity 
## Datasource
The data source is from https://analyse.kmi.open.ac.uk/#open-dataset

## Required CSV files
After downloading the dataset, create a **data** folder in repo and copy the following 3 CSV files:
1. StudentInfo
2. StudentRegistration
3. StudentVle

## Generate and save models
Create a **models** folder Go to the existing **notebooks** folder to run the data-analysis. Three models should be generated and saved in the folder.

## Docker commands

docker build -t your-image-name .
docker run -p 5000:5000 your-image-name
docker run -d --name  your-image-name -p 5000:5000 your-app-name
docker stop your-app-name

OR

docker-compose up --build
docker-compose down

The webpage should be displayed at:
http://127.0.0.1:5000

## References
1. J. Kuzilek, M. Hlosta, and Z. Zdrahal. 2017. Open University Learning Analytics Dataset. Scientific Data 4, 170171. DOI: https://doi.org/10.1038/sdata.2017.171. CC‑BY 4.0. Available: https://analyse.kmi.open.ac.uk/open_dataset
# Docker file for analysis of 
# portugese financial institution client
# data to predict of a customer will 
# subscribe to a term deposit or not.
# Gaurav Sinha, Feb, 2020

# use continuumio/anaconda3 as the base image 
FROM continuumio/anaconda3

# udate everything
RUN apt-get update

# install R 
RUN apt-get install r-base r-base-dev -y

# install R packages
RUN Rscript -e "install.packages('testthat')"
RUN Rscript -e "install.packages('docopt')"

# Install chromedriver
RUN apt-get update && apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /usr/bin/ \
    && rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# install python packages
RUN pip install requests==2.22.0
RUN pip install pandas==0.25.1
RUN pip install altair==4.0.0
RUN pip install seaborn==0.10.0
RUN pip install matplotlib==3.1.1
RUN pip install -U scikit-learn==0.22.1
RUN pip install numpy==1.18.1
RUN pip install tqdm==4.37.0
RUN pip install docopt==0.6.2
RUN pip install selenium==3.141.0
RUN pip install lightgbm




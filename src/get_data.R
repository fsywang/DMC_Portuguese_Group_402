'Download data from given URL and extract in given folder if .zip file
Usage:
    get_data.R [--url=<url> --out_dir=<out_dir>]
    
Options:
    -h --help  Show this screen.
    --url=<url>  URL of file [default: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip]
    --out_dir=<out_dir>  Directory to save the data [default: ../data/raw/]
' -> doc

library(docopt)
library(tools)
arguments <- docopt(doc)

filename <- gsub("^.*/", "", arguments$url)
extension <- file_ext(arguments$url)

dir.create(arguments$out_dir, showWarnings = FALSE)

downloaded_filepath <- file.path(arguments$out_dir, filename)

print('Downloading file')
download.file(arguments$url, downloaded_filepath)

if (file.exists(downloaded_filepath)){
	print(paste('File is stored in :', downloaded_filepath))
	if(extension == 'zip'){
		print('Unzipping the downloaded file')
		unzip(downloaded_filepath, exdir=arguments$out_dir)
	} else {
		print(paste('Not unzipping. The file extension is not zip. It is ', extension))
	}
} else {
	print('File has not been downloaded in some reason. Please check your internet connection')
}



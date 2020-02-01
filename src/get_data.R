'Download data from given URL and extract in given folder if .zip file
Usage:
    get_data.R [--url=<url> --out_dir=<out_dir>]
    
Options:
    -h --help  Show this screen.
    --url=<url>  URL of file [default: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip]
    --out_dir=<out_dir>  Directory to save the data [default: ./data/raw/]
' -> doc

library(docopt)
library(tools)


arguments <- docopt(doc)

main <- function () {
	
	dir.create(arguments$out_dir, showWarnings = FALSE)

	filename <- gsub("^.*/", "", arguments$url)
	filepath <- file.path(arguments$out_dir, filename)
	if(!download_data(arguments$url, arguments$out_dir, filepath)){
		return()
	}
	
	
	extension <- file_ext(filename)
	if(extension == 'zip'){
		print('Unzipping the downloaded file')
		unzip_data(filepath, arguments$out_dir)
	} else {
		print(paste('Not unzipping. The file extension is not zip. It is ', extension))
	}
}

download_data <- function(url, dir, filepath) {
	return(tryCatch({
		dir.create(dir, recursive = TRUE)
		print('Downloading file')
		download.file(url, filepath)

		if (file.exists(filepath)){
			print(paste('File is stored in :', filepath))
			return(TRUE)
		} else {
			print('File has not been downloaded for some reason. Please check your internet connection')
			return(FALSE)
		}
	}, warning=function(cond) {
            	message(paste("URL caused a warning:", url))
		message('Check the URL!!')
            	return(FALSE)
        }, error = function(cond) {
		message(paste("URL does not seem to exist:", url))
            	return(FALSE)
	}))
}

unzip_data <- function(filepath, out_dir) {
	print(paste('Unzipping file: ', filepath))
	return(tryCatch({
		unzip(filepath, exdir=out_dir)
		return(TRUE)
	}, warning=function(cond) {
		message('Cannot unzip: Check file!!')
            	return(FALSE)
        }, error = function(cond) {
		message(paste("Cannot unzip: Check file!"))
            	return(FALSE)
	})) 
}

if (sys.nframe() == 0){
  main()
}

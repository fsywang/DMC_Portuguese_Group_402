library(testthat)
source(here::here('src/get_data.R'))

test_download_function <- function() {
	non_existing_url <- 'https://archive.ics.uci.edu/ml/machine-learning-database/test.png'
	test_that("download_data() function should handle the case when the url is invalid", {
		expect_equal(download_data(non_existing_url, './test_files/test.png'), FALSE)
	})

	existing_url <- 'https://www.google.ca/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'
	existing_zip <- 'http://www.awitness.org/prophecy.zip'
	test_that("download_data() should download existing file without problems and store in the given path", {
		  expect_equal(download_data(existing_url, './test_files/google.png'), TRUE)
		  expect_equal(download_data(existing_zip, './test_files/file.zip'), TRUE)
	})

	
}

test_unzip_function <- function() {
	zip_file <- './test_files/file.zip'
	test_that("unzip_data() function should unzip .zip files to the given folder", {
		expect_equal(unzip_data(zip_file, './test_files'), TRUE)
	})
	
	nonzip_file <- './test_files/google.png'
	test_that("unzip_data() function should handle non zip files as an input", {
		expect_equal(unzip_data(nonzip_file, './test_files'), FALSE)
	})
}

clean_after_tests <- function(){
	print('Cleaning everything')
	do.call(file.remove, list(list.files("./test_files", full.names = TRUE)))
	return(TRUE)
}

test_download_function()
test_unzip_function()
clean_after_tests()


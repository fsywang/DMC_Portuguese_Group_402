import sys
sys.path.append('..')
from src import ml_analysis
import os
import shutil

def clean_after_tests():
    """
    Cleaning files generated throughout tests.
    """
    folder = './test_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def test_subfolder_creation():
    """
    Testing check_filepath() method of ml_analysis.py file
    """
    ml_analysis.check_filepath('./test_files/subdir1/')
    assert os.path.exists('./test_files/subdir1'), 'The function check_filepath should create all subfolders'

def test_non_existing_file_input():
    """
    Testing read_data_and_split() method of ml_analysis.py file
    """
    try:
        ml_analysis.read_data_and_split('./unexisting_dir/nonexisting.csv', './unexisting_dir/nonexisting.csv')
        assert False, 'This code should through an error by design of the function, which is handled in main method'
    except FileNotFoundError:
        pass

def test_csv_report():
    """
    Testing generate_csv_report() method of ml_analysis.py file
    """
    from sklearn.linear_model import LogisticRegression
    sample_arr = (LogisticRegression(), {'param1': 'best_val'}, 0.8, 0.7, 0.8, [0.5, 0.3], [0.7, 0.9])

    ml_analysis.generate_csv_and_figure_reports([sample_arr], './test_files/report.csv', './test_files/report.png')

    assert os.path.exists('./test_files/report.csv'), 'The function should successfully generate csv file'
    assert os.path.exists('./test_files/report.png'), 'The function should successfully generate png file'

test_subfolder_creation()
test_non_existing_file_input()
test_csv_report()
clean_after_tests()

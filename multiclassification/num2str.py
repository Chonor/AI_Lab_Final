
import csv
import glob

for filename in glob.glob(r'*.csv'):
    with open (filename,'r') as url:
        with open ("final"+filename,'w')as process:
            process.write (url.read().replace('1',"LOW").replace('2',"MID").replace('3',"HIG").replace('4',"LOW").replace('5',"MID").replace('6',"HIG"));
            
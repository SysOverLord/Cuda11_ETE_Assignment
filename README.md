# Cuda11_ETE_Assignment
On linux "include<windows.h>" at line 13 must be commented and "include<unistd.h>" at line 15 must be uncommented.<br/>
On linux "Sleep(1*1000)" at line 352 must be commented and "sleep(1)" at line 355 must be uncommented.<br/>
These steps are done because standard libraries on c for these operating systems are different.
